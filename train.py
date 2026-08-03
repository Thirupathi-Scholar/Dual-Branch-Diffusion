"""
train.py — training and ablation harness for DualDiffSeg on MAMA-MIA.

Runs one configuration, or the full eight-configuration ablation sweep with
`--ablate all`. Writes per-epoch metrics to CSV and saves the best checkpoint
by validation Dice.

    # single run
    python train.py --data /path/to/MAMA-MIA --config full --epochs 200

    # full ablation sweep
    python train.py --data /path/to/MAMA-MIA --ablate all --epochs 200

    # resume / evaluate only
    python train.py --data /path/to/MAMA-MIA --config full --eval-only \
        --checkpoint runs/full/best.pt

Expected data layout:
    <data>/images/<case_id>.nii.gz
    <data>/labels/<case_id>.nii.gz

Requires: torch, monai, nibabel, numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import models as M
import losses as L


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def fix_channels(img, n_channels):
    """
    Force a fixed channel count.

    MAMA-MIA does not supply the same number of post-contrast phases for every
    case — the cohorts differ, and some cases have three volumes where others
    have four. Batching requires a fixed channel count, so short cases are
    padded by repeating their last available phase and long ones are
    truncated.

    Repeating the last phase is the conservative choice: it adds no
    information the acquisition did not contain, and leaves the earlier
    phases (where the enhancement dynamics live) untouched.
    """
    if n_channels is None or img.shape[0] == n_channels:
        return img
    if img.shape[0] < n_channels:
        pad = np.repeat(img[-1:], n_channels - img.shape[0], axis=0)
        return np.concatenate([img, pad], axis=0)
    return img[:n_channels]


class NPZDataset(torch.utils.data.Dataset):
    """
    Loads preprocessed .npz volumes written by preprocess.py.

    Resampling, reorientation, foreground cropping and intensity normalization
    already happened offline, so this only reads the array and applies patch
    sampling plus augmentation. That is the difference between ~5 s and
    ~0.3 s per case.
    """

    def __init__(self, records, patch, train=True, seed=0, fg_prob=0.33,
                 n_channels=None):
        """
        fg_prob is the probability a TRAINING patch is centred on tumor.

        The manuscript specifies 50:50 (fg_prob=0.5). That teaches the model
        a prior in which half of all voxels are tumor, while at inference the
        true prevalence is roughly 0.1%. The resulting miscalibration produced
        median volume errors above +1000% and full-volume Dice of 0.14 despite
        patch-level Dice of 0.37. A lower fg_prob keeps lesions adequately
        represented while bringing the learned prior closer to the inference
        distribution.
        """
        self.records = records
        self.patch = tuple(patch)
        self.train = train
        self.fg_prob = fg_prob
        self.n_channels = n_channels
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def _pad(self, img, lab):
        pd, ph, pw = self.patch
        _, d, h, w = img.shape
        pad = [(0, 0), (0, max(0, pd - d)), (0, max(0, ph - h)), (0, max(0, pw - w))]
        if any(p[1] for p in pad):
            img = np.pad(img, pad)
            lab = np.pad(lab, pad[1:])
        return img, lab

    def _sample(self, img, lab, index=0):
        """
        Training: 50:50 foreground/background sampling.

        Validation: ALWAYS centre on the lesion, deterministically. A random
        patch from a 240x240x88 volume contains tumor perhaps 5% of the time,
        so random validation patches measure mostly background and produce a
        Dice figure that is both meaningless and too noisy to select a
        checkpoint on. Foreground-centred validation makes best.pt selection
        track actual segmentation quality. Note this is still patch-level and
        is NOT comparable to full-volume benchmark Dice — use evaluate.py for
        that.
        """
        pd, ph, pw = self.patch
        _, d, h, w = img.shape

        centre = None
        if self.train:
            if self.rng.random() < self.fg_prob:
                fg = np.argwhere(lab > 0)
                if len(fg):
                    centre = fg[self.rng.integers(len(fg))]
        else:
            fg = np.argwhere(lab > 0)
            if len(fg):
                centre = fg[len(fg) // 2]          # deterministic: lesion centre

        if centre is None:
            if self.train:
                z = self.rng.integers(0, max(1, d - pd + 1))
                y = self.rng.integers(0, max(1, h - ph + 1))
                x = self.rng.integers(0, max(1, w - pw + 1))
            else:
                z, y, x = (max(0, d - pd) // 2, max(0, h - ph) // 2,
                           max(0, w - pw) // 2)
        else:
            z = int(np.clip(centre[0] - pd // 2, 0, max(0, d - pd)))
            y = int(np.clip(centre[1] - ph // 2, 0, max(0, h - ph)))
            x = int(np.clip(centre[2] - pw // 2, 0, max(0, w - pw)))

        return (img[:, z:z + pd, y:y + ph, x:x + pw],
                lab[z:z + pd, y:y + ph, x:x + pw])

    def __getitem__(self, i):
        z = np.load(self.records[i]["npz"])
        img, lab = z["image"], z["label"]
        img = fix_channels(img, self.n_channels)
        img, lab = self._pad(img, lab)
        img, lab = self._sample(img, lab, index=i)

        if self.train:
            if self.rng.random() < 0.5:                      # flip along depth
                img, lab = img[:, ::-1], lab[::-1]
            # rot90 only preserves shape when the two rotated axes are equal
            # in size. The patch is typically (96, 96, 32), so rotate in the
            # first two spatial axes and never in a plane involving the
            # short axis.
            if self.patch[0] == self.patch[1]:
                k = int(self.rng.integers(4))
                if k:
                    img = np.rot90(img, k, axes=(1, 2))
                    lab = np.rot90(lab, k, axes=(0, 1))

        assert img.shape[1:] == self.patch, (
            f"patch shape {img.shape[1:]} != expected {self.patch} "
            f"for {self.records[i]['npz']}")

        return dict(
            image=torch.from_numpy(np.ascontiguousarray(img)).float(),
            label=torch.from_numpy(np.ascontiguousarray(lab)).float().unsqueeze(0),
        )


def build_transforms(train: bool, patch=(96, 96, 32), spacing=(1.5, 1.5, 2.0),
                     phase="1"):
    """
    Preprocessing pipeline matching manuscript Table 1.

    NOTE: the MAMA-MIA authors recommend z-score normalization across all DCE
    phases and isotropic 1x1x1 mm spacing for optimum performance with their
    reference nnU-Net. This pipeline follows the manuscript instead. If you
    are producing baseline comparison numbers, run each baseline under ITS OWN
    recommended preprocessing — nnU-Net in particular self-configures, and
    forcing it through this pipeline understates it.
    """
    from monai import transforms as T

    common = [
        T.LoadImaged(keys=["image", "label"]),
        T.EnsureChannelFirstd(keys=["image", "label"]),
    ]

    if phase == "sub":
        # image arrives as 2 channels (pre, post); subtract to isolate enhancement
        common.append(T.Lambdad(
            keys=["image"],
            func=lambda x: (x[1:2] - x[0:1]) if x.shape[0] >= 2 else x,
        ))

    common += [
        T.Spacingd(keys=["image", "label"], pixdim=spacing,
                   mode=("bilinear", "nearest")),
        T.Orientationd(keys=["image", "label"], axcodes="RAS"),
        T.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                               b_min=0.0, b_max=1.0, clip=True),
        T.CropForegroundd(keys=["image", "label"], source_key="image",
                          allow_smaller=True),
    ]

    if train:
        aug = [
            T.RandCropByPosNegLabeld(
                keys=["image", "label"], label_key="label",
                spatial_size=patch, pos=1, neg=1, num_samples=2,
                image_key="image", allow_smaller=True,
            ),
            T.SpatialPadd(keys=["image", "label"], spatial_size=patch),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            T.ToTensord(keys=["image", "label"]),
        ]
    else:
        # deterministic only — validation must not be augmented
        aug = [
            T.Resized(keys=["image", "label"], spatial_size=patch,
                      mode=("trilinear", "nearest")),
            T.ToTensord(keys=["image", "label"]),
        ]

    return T.Compose(common + aug)


def load_manifests(manifest_dir):
    """
    Load train/val/test manifests produced by prepare_data.py.

    Using the dataset authors' official split makes results directly
    comparable to their published nnU-Net reference. The validation set is
    carved from the official train partition only; the official test set is
    never seen during training or tuning.
    """
    import json
    d = Path(manifest_dir)
    out = {}
    for name in ["train", "val", "test"]:
        f = d / f"{name}.json"
        if not f.exists():
            raise SystemExit(f"missing {f} — run prepare_data.py first")
        payload = json.loads(f.read_text())
        out[name] = payload["records"]
        out.setdefault("_phase", payload.get("phase", "1"))
        out.setdefault("_preprocessed", payload.get("preprocessed", False))
    return out


def to_records(records):
    """Manifest entries are already in MONAI dict form."""
    return [dict(image=r["image"], label=r["label"]) for r in records]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def dice_score(logits, target, threshold=0.5, eps=1e-6):
    p = (torch.sigmoid(logits) > threshold).float().flatten(1)
    g = target.flatten(1).float()
    inter = (p * g).sum(1)
    denom = p.sum(1) + g.sum(1)
    valid = denom > 0
    if valid.sum() == 0:
        return float("nan")
    return float(((2 * inter[valid] + eps) / (denom[valid] + eps)).mean())


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------

def run_epoch(model, loader, crit, opt, device, train=True, scaler=None, amp=False):
    model.train() if train else model.eval()
    agg, dices = {}, []

    # bf16 on Ampere/Ada and newer; fp16 elsewhere. bf16 needs no loss scaling.
    amp_dtype = torch.bfloat16 if (
        amp and device == "cuda" and torch.cuda.is_bf16_supported()
    ) else torch.float16

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp):
                out = model(x)
            # loss in fp32 — Dice and the entropy term are numerically fragile at 16 bit
            out["logits"] = out["logits"].float()
            total, parts = crit(out, y, image=x)

        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                scaler.step(opt)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                opt.step()

        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        dices.append(dice_score(out["logits"], y))

    n = max(len(loader), 1)
    stats = {k: v / n for k, v in agg.items()}
    stats["dice"] = float(np.nanmean(dices)) if dices else float("nan")
    return stats


def train_one(cfg_name, args, splits, device):
    from monai.data import CacheDataset, DataLoader

    out_dir = Path(args.out) / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps(
        {k: [r["case"] for r in v] for k, v in splits.items()
         if not k.startswith("_")}, indent=1))

    phase = splits.get("_phase", "1")
    in_ch = 4 if phase == "all" else 1

    if splits.get("_preprocessed"):
        tr_ds = NPZDataset(splits["train"], args.patch, train=True,
                           seed=args.seed, fg_prob=args.fg_prob,
                           n_channels=in_ch)
        va_ds = NPZDataset(splits["val"], args.patch, train=False,
                           seed=args.seed, n_channels=in_ch)
    else:
        tr_ds = CacheDataset(to_records(splits["train"]),
                             build_transforms(True, tuple(args.patch), phase=phase),
                             cache_rate=args.cache, num_workers=args.workers)
        va_ds = CacheDataset(to_records(splits["val"]),
                             build_transforms(False, tuple(args.patch), phase=phase),
                             cache_rate=args.cache, num_workers=args.workers)

    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    va = DataLoader(va_ds, batch_size=1, num_workers=args.workers, pin_memory=True)

    model = M.build(cfg_name, in_ch=in_ch, patch=tuple(args.patch),
                    norm_kind=args.norm).to(device)
    # Build the loss kwargs, then let the per-config override REPLACE the
    # matching entry rather than being passed alongside it. Passing both
    # raises "got multiple values for keyword argument".
    loss_kw = dict(
        alpha=args.alpha, lambda_seg=args.lambda_seg,
        lambda_diff=args.lambda_diff, lambda_conf=args.lambda_conf,
        lambda_mri=args.lambda_mri,
        conf_warmup=args.conf_warmup, conf_ramp=args.conf_ramp,
        conf_floor=args.conf_floor,
    )
    overrides = M.LOSS_OVERRIDES.get(cfg_name, {})
    loss_kw.update(overrides)
    if overrides:
        print(f"  loss override: {overrides}")
    crit = L.DualDiffSegLoss(**loss_kw).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # GradScaler is only needed for fp16; bf16 has fp32 dynamic range
    use_fp16_scaler = (args.amp and device == "cuda"
                       and not torch.cuda.is_bf16_supported())
    scaler = torch.amp.GradScaler(device, enabled=use_fp16_scaler)

    print(f"\n{'='*66}\n{cfg_name}  |  {model.n_params()/1e6:.2f}M params  |  "
          f"{len(splits['train'])} train / {len(splits['val'])} val\n{'='*66}")

    log_path = out_dir / "log.csv"
    best, t0 = -1.0, time.time()

    with open(log_path, "w", newline="") as fh:
        writer = None
        for ep in range(1, args.epochs + 1):
            crit.set_epoch(ep - 1)
            trs = run_epoch(model, tr, crit, opt, device, train=True,
                            scaler=scaler, amp=args.amp)
            vas = run_epoch(model, va, crit, opt, device, train=False,
                            amp=args.amp)
            sched.step()

            row = dict(epoch=ep, lr=opt.param_groups[0]["lr"],
                       **{f"train_{k}": v for k, v in trs.items()},
                       **{f"val_{k}": v for k, v in vas.items()})
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()

            if vas["dice"] > best:
                best = vas["dice"]
                torch.save(dict(model=model.state_dict(), config=cfg_name,
                                norm_kind=args.norm, in_ch=in_ch,
                                patch=list(args.patch),
                                epoch=ep, val_dice=best, args=vars(args)),
                           out_dir / "best.pt")

            if ep % args.log_every == 0 or ep == 1:
                print(f"  ep {ep:>4}  train dice {trs['dice']:.4f}  "
                      f"val dice {vas['dice']:.4f}  (best {best:.4f})  "
                      f"loss {trs['total']:.4f}  "
                      f"H(conf) {trs.get('conf', 0):.4f} "
                      f"w {trs.get('w_conf', 0):.4g}")

    mins = (time.time() - t0) / 60
    print(f"  done — best val Dice {best:.4f} in {mins:.1f} min")
    return dict(config=cfg_name, best_val_dice=best,
                params_m=model.n_params() / 1e6, minutes=mins)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", required=True,
                    help="directory written by prepare_data.py")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--config", default="full", choices=list(M.ABLATIONS))
    ap.add_argument("--ablate", choices=["all"],
                    help="run every ablation configuration in sequence")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 32])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lambda-seg", type=float, default=1.0)
    ap.add_argument("--lambda-diff", type=float, default=0.1)
    ap.add_argument("--lambda-conf", type=float, default=0.05)
    ap.add_argument("--lambda-mri", type=float, default=0.01)
    ap.add_argument("--fg-prob", type=float, default=0.33,
                    help="probability a training patch is centred on tumor. "
                         "0.5 reproduces the manuscript's 50:50 sampling, "
                         "which miscalibrates the model badly against the "
                         "~0.1%% true prevalence at inference.")
    ap.add_argument("--conf-warmup", type=int, default=20,
                    help="epochs before the confidence penalty engages at all")
    ap.add_argument("--conf-ramp", type=int, default=30,
                    help="epochs over which it ramps to full weight")
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="disable the penalty while mean entropy is below this "
                         "(0 to disable the floor)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--force", action="store_true",
                    help="retrain configs that already have a best.pt")
    ap.add_argument("--norm", default="batch",
                    choices=["batch", "instance", "group"],
                    help="Normalization layer. 'batch' reproduces the "
                         "manuscript. 'instance' is what nnU-Net uses and "
                         "is batch-size independent — recommended below "
                         "batch 8.")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (bf16 on Ada/Ampere+). Roughly halves "
                         "VRAM use and speeds training. Recommended on <=8GB cards.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. 3D training on CPU is not viable — "
              "expect weeks per configuration. See README for compute options.")
    else:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        mode = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if args.amp else "fp32"
        print(f"GPU: {name}  |  {vram:.1f} GB  |  precision: {mode}  |  "
              f"norm: {args.norm}  |  batch: {args.batch}  |  "
              f"patch: {tuple(args.patch)}")
        if vram < 8 and not args.amp:
            print("  hint: pass --amp on a card this size; it roughly halves memory use.")

    splits = load_manifests(args.manifests)
    print(f"split: {len(splits['train'])} train / {len(splits['val'])} val / "
          f"{len(splits['test'])} test   (official MAMA-MIA split, "
          f"phase={splits.get('_phase')}"
          + (", preprocessed" if splits.get("_preprocessed") else "") + ")")

    configs = list(M.ABLATIONS) if args.ablate == "all" else [args.config]

    results = []
    for c in configs:
        ckpt = Path(args.out) / c / "best.pt"
        if ckpt.exists() and not args.force:
            import torch as _t
            info = _t.load(ckpt, map_location="cpu", weights_only=False)
            print(f"\n[skip] {c} — already trained "
                  f"(epoch {info.get('epoch','?')}, "
                  f"val Dice {info.get('val_dice', float('nan')):.4f}). "
                  f"Use --force to retrain.")
            results.append(dict(config=c,
                                best_val_dice=float(info.get("val_dice", float("nan"))),
                                params_m=float("nan"), minutes=float("nan")))
            continue
        results.append(train_one(c, args, splits, device))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    summary = Path(args.out) / "ablation_summary.csv"
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*66}\nABLATION SUMMARY\n{'='*66}")
    full = next((r["best_val_dice"] for r in results if r["config"] == "full"), None)
    for r in results:
        delta = (f"{r['best_val_dice']-full:+.4f}"
                 if full is not None and r["config"] != "full" else "—")
        print(f"{r['config']:<16}{r['best_val_dice']:>9.4f}   "
              f"delta {delta:>9}   {r['params_m']:>6.2f}M")
    print(f"\nwritten to {summary}")
    print("\nNOTE: these are validation-set numbers from a single run each. "
          "For the paper, evaluate on the held-out TEST split, repeat each "
          "configuration across seeds, and report paired significance tests.")


if __name__ == "__main__":
    main()
