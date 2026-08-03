"""
tune_threshold.py — find the decision threshold on the VALIDATION split.

Why this is needed
------------------
Training samples patches at a 50:50 foreground/background ratio to cope with
class imbalance. The model therefore learns a prior in which roughly half of
what it sees is tumor. At inference it is given whole volumes where tumor is
around 0.1% of voxels, so a 0.5 threshold over-segments enormously — observed
median volume error above +1000%.

The probabilities are miscalibrated, not uninformative. Rescaling the decision
threshold usually recovers most of the loss without retraining.

Two corrections are evaluated:
  - threshold sweep, from 0.5 up to 0.9995
  - largest connected component, discarding scattered false-positive islands

IMPORTANT: this tunes on VALIDATION. Selecting a threshold on the test split
would leak it and inflate the reported result. The chosen value is then applied
unchanged in evaluate.py.

    python tune_threshold.py --checkpoint runs_pilot4/full/best.pt \\
        --manifests ~/data/preprocessed/manifests --n 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import models as M

def build_from_ckpt(ckpt, device, default_patch=(96, 96, 32)):
    """
    Rebuild a model exactly as it was trained.

    Checkpoints record norm_kind, in_ch and patch, so evaluation never has to
    guess. Older checkpoints lack these keys and fall back to the manuscript
    defaults (BatchNorm, single channel).
    """
    import models as _M
    cfg = ckpt.get("config", "full")
    model = _M.build(
        cfg,
        in_ch=ckpt.get("in_ch", 1),
        patch=tuple(ckpt.get("patch", default_patch)),
        norm_kind=ckpt.get("norm_kind", "batch"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995]


def dice_at(prob, gt, thr):
    p = prob > thr
    inter = np.logical_and(p, gt).sum()
    denom = p.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else float("nan")


def largest_component(mask):
    """Keep only the largest connected component."""
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    if mask.sum() == 0:
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


@torch.no_grad()
def predict(model, img, device, patch, overlap=0.5):
    from monai.inferers import sliding_window_inference
    x = img.unsqueeze(0).to(device)
    logits = sliding_window_inference(
        x, roi_size=patch, sw_batch_size=2,
        predictor=lambda t: model(t)["logits"], overlap=overlap, mode="gaussian")
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="never 'test' — that would leak the threshold")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 32])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", "full")

    model, cfg = build_from_ckpt(ckpt, device, tuple(args.patch))

    print(f"checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch','?')})")
    print(f"tuning on:  {args.split} split, {args.n} cases\n")

    records = json.loads(
        (Path(args.manifests) / f"{args.split}.json").read_text())["records"]

    raw = {t: [] for t in THRESHOLDS}
    lcc = {t: [] for t in THRESHOLDS}
    vol_err = {t: [] for t in THRESHOLDS}

    for i, r in enumerate(records[:args.n]):
        z = np.load(r["npz"])
        img = torch.from_numpy(z["image"]).float()
        gt = z["label"].astype(bool)

        prob = predict(model, img, device, tuple(args.patch))

        for t in THRESHOLDS:
            pred = prob > t
            raw[t].append(dice_at(prob, gt, t))
            lcc[t].append(
                float(2 * np.logical_and(largest_component(pred), gt).sum()
                      / max(1, largest_component(pred).sum() + gt.sum())))
            vol_err[t].append(
                100.0 * (pred.sum() - gt.sum()) / max(1, gt.sum()))

        if (i + 1) % 5 == 0:
            best_now = max(THRESHOLDS, key=lambda t: np.nanmean(raw[t]))
            print(f"  {i+1}/{args.n}   best so far: thr {best_now} "
                  f"Dice {np.nanmean(raw[best_now]):.4f}", flush=True)

    print(f"\n{'threshold':>10}{'Dice':>10}{'Dice+LCC':>11}"
          f"{'median vol err %':>19}")
    print("-" * 52)
    for t in THRESHOLDS:
        print(f"{t:>10}{np.nanmean(raw[t]):>10.4f}{np.nanmean(lcc[t]):>11.4f}"
              f"{np.nanmedian(vol_err[t]):>19.1f}")

    best_raw = max(THRESHOLDS, key=lambda t: np.nanmean(raw[t]))
    best_lcc = max(THRESHOLDS, key=lambda t: np.nanmean(lcc[t]))
    d_raw, d_lcc = np.nanmean(raw[best_raw]), np.nanmean(lcc[best_lcc])

    print("\n" + "=" * 52)
    print(f"best threshold          {best_raw}   Dice {d_raw:.4f}")
    print(f"best threshold + LCC    {best_lcc}   Dice {d_lcc:.4f}")
    print(f"baseline (thr 0.5)          Dice {np.nanmean(raw[0.5]):.4f}")
    print("=" * 52)

    gain = max(d_raw, d_lcc) - np.nanmean(raw[0.5])
    print(f"\nrecovered {gain:+.4f} Dice without retraining")

    if max(d_raw, d_lcc) < 0.45:
        print("\nStill well short of the 0.7620 reference. Threshold tuning is")
        print("a patch, not a fix — retrain with a less extreme sampling ratio")
        print("(pos:neg of 1:3 or 1:5 rather than 1:1) so the learned prior is")
        print("closer to the inference distribution.")
    else:
        print(f"\nApply with:  evaluate.py --threshold {best_raw}")
        print("Report the threshold and how it was selected (validation split,")
        print("never test) in the methods section.")


if __name__ == "__main__":
    main()
