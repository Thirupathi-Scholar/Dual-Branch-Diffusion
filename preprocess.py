"""
preprocess.py — one-time resampling pass to remove the dataloader bottleneck.

The raw MAMA-MIA volumes are ~448x448x208 at 0.83x0.83x1.0 mm. Every training
step currently reloads and resamples them, which costs 3-8 s per case. Across
200 epochs and 8 ablation configurations that is the same resampling work
repeated roughly 1.6 million times.

This script does it once. Each case is resampled to the target spacing,
reoriented to RAS, cropped to the foreground bounding box, and written as a
single compressed .npz. Training then only reads and crops.

    python preprocess.py --root ~/data --out ~/data/preprocessed --workers 4

Expected effect: per-case load drops from ~5 s to well under 1 s, epoch time
from ~31 min to a few minutes, and the sweep from over a month to a few days.

Runs once, takes roughly 30-60 minutes on 4 cores. Run it in tmux.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def resample_and_crop(img_path, lab_path, spacing, margin=8, phase="1",
                      norm="zscore"):
    """
    Load, reorient to RAS, resample to `spacing`, crop to the foreground
    bounding box with a margin. Returns (image, label, meta).
    """
    import nibabel as nib
    from nibabel.orientations import (io_orientation, axcodes2ornt,
                                      ornt_transform, apply_orientation)
    from scipy.ndimage import zoom

    paths = img_path if isinstance(img_path, list) else [img_path]

    vols, zooms_in = [], None
    for p in paths:
        n = nib.load(p)
        ornt = ornt_transform(io_orientation(n.affine), axcodes2ornt(("R", "A", "S")))
        arr = apply_orientation(np.asanyarray(n.dataobj, dtype=np.float32), ornt)
        vols.append(arr)
        if zooms_in is None:
            zooms_in = tuple(float(z) for z in n.header.get_zooms()[:3])

    ln = nib.load(lab_path)
    lornt = ornt_transform(io_orientation(ln.affine), axcodes2ornt(("R", "A", "S")))
    lab = apply_orientation(np.asanyarray(ln.dataobj), lornt).astype(np.uint8)

    if phase == "sub" and len(vols) >= 2:
        vols = [vols[1] - vols[0]]

    # resample: trilinear for images, nearest for the label
    factors = tuple(zi / zo for zi, zo in zip(zooms_in, spacing))
    vols = [zoom(v, factors, order=1, prefilter=False) for v in vols]
    lab = zoom(lab, factors, order=0, prefilter=False)

    img = np.stack(vols, axis=0).astype(np.float32)     # C, D, H, W

    # align shapes if rounding differs by a voxel
    if lab.shape != img.shape[1:]:
        s = tuple(min(a, b) for a, b in zip(lab.shape, img.shape[1:]))
        img = img[:, :s[0], :s[1], :s[2]]
        lab = lab[:s[0], :s[1], :s[2]]

    # crop to foreground (nonzero image intensity), with margin
    nz = np.nonzero(img[0] > 0)
    if len(nz[0]) > 0:
        lo = [max(0, int(a.min()) - margin) for a in nz]
        hi = [min(s, int(a.max()) + margin + 1) for a, s in zip(nz, img.shape[1:])]
        img = img[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        lab = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # ---- intensity normalization ---------------------------------------
    # MAMA-MIA pools DUKE, ISPY1 and ISPY2: different scanners, no shared
    # intensity calibration. Observed per-case maxima vary by more than 20x.
    # A fixed window (the manuscript's [0,3000]) compresses low-signal cases
    # into a fraction of the range while clipping high-signal ones, so
    # per-case normalization is used instead. This matches the preprocessing
    # the dataset authors specify for their reference nnU-Net.
    body = img[0] > 0
    if norm == "zscore":
        if body.sum() > 0:
            vals = img[:, body] if img.shape[0] > 1 else img[0][body][None]
            mu, sd = float(vals.mean()), float(vals.std())
            img = (img - mu) / (sd + 1e-8) if sd > 0 else img - mu
        img = np.clip(img, -5, 5)
    elif norm == "percentile":
        if body.sum() > 0:
            lo, hi = np.percentile(img[0][body], [0.5, 99.5])
            img = np.clip(img, lo, hi)
            img = (img - lo) / (hi - lo + 1e-8)
    elif norm == "fixed":
        img = np.clip(img, 0, 3000) / 3000.0
    else:
        raise ValueError(f"unknown norm {norm!r}")

    meta = dict(spacing=list(spacing), shape=list(img.shape),
                fg_voxels=int(lab.sum()), norm=norm,
                intensity=[float(img.min()), float(img.max())])
    return img.astype(np.float32), (lab > 0).astype(np.uint8), meta


def process_one(task):
    case, img_path, lab_path, out_path, spacing, phase, norm = task
    try:
        if Path(out_path).exists():
            return (case, "skipped", 0.0, None)
        t0 = time.time()
        img, lab, meta = resample_and_crop(img_path, lab_path, tuple(spacing),
                                           phase=phase, norm=norm)
        np.savez_compressed(out_path, image=img, label=lab,
                            meta=json.dumps(meta))
        return (case, "ok", time.time() - t0, meta)
    except Exception:
        return (case, "error", 0.0, traceback.format_exc(limit=3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path.home() / "data"))
    ap.add_argument("--manifests", default=None,
                    help="default: <root>/manifests")
    ap.add_argument("--out", default=None,
                    help="default: <root>/preprocessed")
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--norm", default="zscore",
                    choices=["zscore", "percentile", "fixed"],
                    help="Intensity normalization. 'zscore' (default) is "
                         "per-case over the body region and is what the "
                         "MAMA-MIA authors specify. 'fixed' reproduces the "
                         "manuscript's [0,3000] window, which is unsuitable "
                         "for this multi-scanner cohort.")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    man_dir = Path(args.manifests).expanduser() if args.manifests else root / "manifests"
    out_dir = Path(args.out).expanduser() if args.out else root / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, splits = [], {}
    phase = "1"
    for name in ["train", "val", "test"]:
        payload = json.loads((man_dir / f"{name}.json").read_text())
        phase = payload.get("phase", "1")
        ids = []
        for r in payload["records"]:
            out_path = out_dir / f"{r['case']}.npz"
            tasks.append((r["case"], r["image"], r["label"], str(out_path),
                          args.spacing, phase, args.norm))
            ids.append(r["case"])
        splits[name] = ids

    print(f"cases:    {len(tasks)}")
    print(f"spacing:  {tuple(args.spacing)} mm")
    print(f"phase:    {phase}")
    print(f"norm:     {args.norm}")
    print(f"workers:  {args.workers}")
    print(f"output:   {out_dir}\n")

    ok = skipped = errors = 0
    times, err_log = [], {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            case, status, dt, info = fut.result()
            if status == "ok":
                ok += 1
                times.append(dt)
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                err_log[case] = info

            if i % 25 == 0 or i == len(tasks):
                el = time.time() - t_start
                rate = i / el
                eta = (len(tasks) - i) / rate / 60 if rate > 0 else 0
                print(f"  {i}/{len(tasks)}  ok={ok} skip={skipped} err={errors}"
                      f"  {rate:.1f} case/s  ETA {eta:.0f} min")

    if err_log:
        (out_dir / "errors.json").write_text(json.dumps(err_log, indent=1))
        print(f"\n{errors} case(s) failed — see {out_dir/'errors.json'}")

    # manifests pointing at the preprocessed files
    pre_man = out_dir / "manifests"
    pre_man.mkdir(exist_ok=True)
    for name, ids in splits.items():
        recs = [dict(case=c, npz=str(out_dir / f"{c}.npz"))
                for c in ids if (out_dir / f"{c}.npz").exists()]
        (pre_man / f"{name}.json").write_text(json.dumps(
            dict(phase=phase, preprocessed=True, spacing=args.spacing,
                 norm=args.norm, n=len(recs), records=recs), indent=1))
        print(f"  {name:<6} {len(recs):>5} cases")

    mins = (time.time() - t_start) / 60
    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 1e9

    print(f"\ndone in {mins:.0f} min   |   {size_gb:.1f} GB written")
    if times:
        print(f"mean {np.mean(times):.2f} s/case")
    print(f"\nnext:\n  python train.py --manifests {pre_man} --config full "
          f"--epochs 2 --workers 3 --amp")


if __name__ == "__main__":
    main()
