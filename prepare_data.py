"""
prepare_data.py — adapter for the MAMA-MIA directory layout.

The dataset ships as:

    <root>/images/<CASE>/<CASE>_0000.nii.gz    pre-contrast
                        <CASE>_0001.nii.gz     1st post-contrast
                        <CASE>_0002.nii.gz     2nd post-contrast
                        <CASE>_0003.nii.gz     3rd post-contrast
    <root>/segmentations/expert/<CASE>.nii.gz
    <root>/train_test_splits.csv               official partition

Rather than copying 95 GB into a flat layout, this builds JSON manifests that
point at the files in place.

    python prepare_data.py --root ~/data --out ~/data/manifests

Phase selection (--phase):
    1        first post-contrast only. Default. Matches the manuscript's
             single-channel 1xDxHxW input, and is where enhancement is
             strongest.
    0        pre-contrast only
    sub      first post-contrast minus pre-contrast, computed on the fly.
             Often the strongest single-channel signal for lesion contrast.
    all      all four phases stacked as channels. Deviates from the
             manuscript's stated architecture — if you use this, say so in
             the paper and set in_ch=4 when building the model.

The official split is used verbatim so results are directly comparable to the
dataset authors' published nnU-Net reference (Dice 0.7620, HD95 37.41 mm).
A validation set is carved deterministically out of the training partition;
the official test set is never touched during development.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_official_splits(root: Path):
    """
    train_test_splits.csv has two independent columns, train_split and
    test_split, each listing case IDs. They are not row-paired — a row may
    have a value in one column and not the other.
    """
    train, test = [], []
    with open(root / "train_test_splits.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            t = (row.get("train_split") or "").strip()
            s = (row.get("test_split") or "").strip()
            if t:
                train.append(t)
            if s:
                test.append(s)
    return sorted(set(train)), sorted(set(test))


def phase_files(images_dir: Path, case: str, phase: str):
    """Return the image path(s) for a case under the chosen phase policy."""
    d = images_dir / case
    p = {i: d / f"{case}_{i:04d}.nii.gz" for i in range(4)}

    if phase == "0":
        return [p[0]] if p[0].exists() else None
    if phase == "1":
        return [p[1]] if p[1].exists() else None
    if phase == "sub":
        return [p[0], p[1]] if (p[0].exists() and p[1].exists()) else None
    if phase == "all":
        present = [p[i] for i in range(4) if p[i].exists()]
        return present if len(present) >= 2 else None
    raise ValueError(f"unknown phase policy {phase!r}")


def build_records(root: Path, cases, phase: str):
    images_dir = root / "images"
    seg_dir = root / "segmentations" / "expert"

    records, missing = [], []
    for case in cases:
        seg = seg_dir / f"{case}.nii.gz"
        imgs = phase_files(images_dir, case, phase)
        if imgs is None or not seg.exists():
            missing.append(case)
            continue
        records.append(dict(
            case=case,
            image=[str(p) for p in imgs] if len(imgs) > 1 else str(imgs[0]),
            label=str(seg),
        ))
    return records, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path.home() / "data"))
    ap.add_argument("--out", default=None, help="default: <root>/manifests")
    ap.add_argument("--phase", default="1", choices=["0", "1", "sub", "all"])
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of the official TRAIN split held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser() if args.out else root / "manifests"
    out.mkdir(parents=True, exist_ok=True)

    if not (root / "train_test_splits.csv").exists():
        raise SystemExit(f"train_test_splits.csv not found under {root}")

    train_ids, test_ids = read_official_splits(root)
    print(f"official split:  {len(train_ids)} train / {len(test_ids)} test")

    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise SystemExit(f"train/test overlap in official CSV: {sorted(overlap)[:5]}")

    # deterministic validation carve-out from the training partition only
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(train_ids))
    n_val = int(len(train_ids) * args.val_frac)
    val_ids = sorted(train_ids[i] for i in idx[:n_val])
    tr_ids = sorted(train_ids[i] for i in idx[n_val:])

    all_missing = {}
    for name, ids in [("train", tr_ids), ("val", val_ids), ("test", test_ids)]:
        recs, missing = build_records(root, ids, args.phase)
        if missing:
            all_missing[name] = missing
        (out / f"{name}.json").write_text(json.dumps(
            dict(phase=args.phase, n=len(recs), records=recs), indent=1))
        print(f"  {name:<6} {len(recs):>5} cases"
              + (f"   ({len(missing)} skipped, files missing)" if missing else ""))

    if all_missing:
        (out / "missing.json").write_text(json.dumps(all_missing, indent=1))
        total = sum(len(v) for v in all_missing.values())
        print(f"\n{total} case(s) skipped for missing files — listed in "
              f"{out/'missing.json'}")
        print("Report this count in the paper; do not quietly drop cases.")

    meta = dict(
        phase_policy=args.phase,
        val_frac=args.val_frac,
        seed=args.seed,
        official_train=len(train_ids),
        official_test=len(test_ids),
        note=("Validation carved from the official train split only. "
              "The official test split is untouched during development."),
    )
    (out / "manifest_meta.json").write_text(json.dumps(meta, indent=1))

    print(f"\nmanifests written to {out}/")
    print(f"\nnext:\n  python train.py --manifests {out} --config full "
          f"--epochs 2 --workers 3 --amp")


if __name__ == "__main__":
    main()
