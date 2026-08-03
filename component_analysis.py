"""
component_analysis.py — is the lesion found but discarded, or never found?

Context
-------
About a third of test cases score near-zero Dice. Adaptive thresholding ruled
out a calibration artifact: every case has peak probability above 0.998, so
nothing is being zeroed out by the cutoff. The model is confident and wrong.

But "wrong" has two very different explanations, and they call for different
responses:

  (a) The lesion IS among the predicted connected components, just not the
      largest one. Largest-component selection then throws away the correct
      answer. This is a post-processing bug and is cheap to fix — pick the
      component by peak or mean probability instead of by size.

  (b) No predicted component overlaps the lesion at all. The model genuinely
      looked elsewhere. That is a real detection failure and no amount of
      post-processing recovers it.

This script separates them. For each failing case it enumerates every
connected component of the thresholded prediction and reports:

  best_dice        Dice of the BEST available component (the ceiling that
                   perfect component selection would reach)
  rank_by_size     where that component ranks by voxel count (1 = largest)
  rank_by_peak     where it ranks by peak probability
  rank_by_mean     where it ranks by mean probability

If best_dice is high but rank_by_size is poor, selection is the problem. If
best_dice is near zero, it is not.

    python component_analysis.py --checkpoint runs/full/best.pt \\
        --manifests ~/data/preprocessed/manifests --split val --n 60
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



def dice(pred, gt):
    denom = pred.sum() + gt.sum()
    return float(2 * np.logical_and(pred, gt).sum() / denom) if denom else float("nan")


@torch.no_grad()
def predict(model, img, device, patch, overlap=0.5):
    from monai.inferers import sliding_window_inference
    x = img.unsqueeze(0).to(device)
    logits = sliding_window_inference(
        x, roi_size=patch, sw_batch_size=2,
        predictor=lambda t: model(t)["logits"], overlap=overlap, mode="gaussian")
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def analyse(prob, gt, thr):
    from scipy import ndimage

    mask = prob > thr
    if mask.sum() == 0:
        return None

    lab, n = ndimage.label(mask)
    if n == 0:
        return None

    comps = []
    for i in range(1, n + 1):
        c = lab == i
        comps.append(dict(
            idx=i, size=int(c.sum()),
            peak=float(prob[c].max()), mean=float(prob[c].mean()),
            dice=dice(c, gt),
        ))

    by_size = sorted(comps, key=lambda c: -c["size"])
    by_peak = sorted(comps, key=lambda c: -c["peak"])
    by_mean = sorted(comps, key=lambda c: -c["mean"])

    best = max(comps, key=lambda c: c["dice"] if not np.isnan(c["dice"]) else -1)

    return dict(
        n_components=n,
        dice_largest=by_size[0]["dice"],
        dice_by_peak=by_peak[0]["dice"],
        dice_by_mean=by_mean[0]["dice"],
        best_dice=best["dice"],
        rank_by_size=1 + [c["idx"] for c in by_size].index(best["idx"]),
        rank_by_peak=1 + [c["idx"] for c in by_peak].index(best["idx"]),
        rank_by_mean=1 + [c["idx"] for c in by_mean].index(best["idx"]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 32])
    ap.add_argument("--threshold", type=float, default=0.98)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _ = build_from_ckpt(ckpt, device, tuple(args.patch))

    records = json.loads(
        (Path(args.manifests) / f"{args.split}.json").read_text())["records"][:args.n]

    print(f"checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    print(f"split:      {args.split}, {len(records)} cases, threshold {args.threshold}\n")

    rows = []
    for i, r in enumerate(records):
        z = np.load(r["npz"])
        img = torch.from_numpy(z["image"]).float()
        gt = z["label"].astype(bool)
        prob = predict(model, img, device, tuple(args.patch))
        a = analyse(prob, gt, args.threshold)
        if a:
            a["case"] = r["case"]
            rows.append(a)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)

    fail = df[df.dice_largest < 0.1]

    print(f"\n{'='*70}")
    print(f"ALL CASES  (n={len(df)})")
    print("=" * 70)
    print(f"  components per case (median):  {df.n_components.median():.0f}")
    print(f"  Dice, largest component:       {df.dice_largest.mean():.4f}")
    print(f"  Dice, highest peak prob:       {df.dice_by_peak.mean():.4f}")
    print(f"  Dice, highest mean prob:       {df.dice_by_mean.mean():.4f}")
    print(f"  Dice, ORACLE best component:   {df.best_dice.mean():.4f}")

    print(f"\n{'='*70}")
    print(f"FAILING CASES  (Dice of largest component < 0.1, n={len(fail)})")
    print("=" * 70)

    if len(fail) == 0:
        print("  none")
        return

    print(f"  components per case (median):  {fail.n_components.median():.0f}")
    print(f"  best available Dice (oracle):  {fail.best_dice.mean():.4f}")
    print(f"  ... of which score > 0.3:      {(fail.best_dice > 0.3).sum()}"
          f" / {len(fail)}")
    print()
    print(f"  Dice by peak-probability rule: {fail.dice_by_peak.mean():.4f}")
    print(f"  Dice by mean-probability rule: {fail.dice_by_mean.mean():.4f}")
    print()
    print(f"  median rank of correct component by size: "
          f"{fail.rank_by_size.median():.0f}")
    print(f"  median rank by peak probability:          "
          f"{fail.rank_by_peak.median():.0f}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    recoverable = (fail.best_dice > 0.3).mean()
    peak_gain = fail.dice_by_peak.mean() - fail.dice_largest.mean()

    if recoverable > 0.4:
        print(f"{100*recoverable:.0f}% of failing cases contain a component that")
        print("would score above 0.3. The lesion IS being found and then")
        print("discarded by largest-component selection.")
        if peak_gain > 0.05:
            print(f"\nSelecting by peak probability instead recovers "
                  f"{peak_gain:+.4f} Dice\non these cases. Worth switching, "
                  f"and worth reporting as a\npost-processing finding.")
        else:
            print("\nNeither peak nor mean probability identifies the correct")
            print("component reliably, so a better selection rule is needed —")
            print("but the information is present.")
    else:
        print(f"Only {100*recoverable:.0f}% of failing cases contain any")
        print("component overlapping the lesion. The model is looking in the")
        print("wrong place entirely — this is a genuine detection failure and")
        print("post-processing will not fix it.")
        print("\nLikely causes, in order of plausibility:")
        print("  - 96x96x32 patches give too little context to distinguish a")
        print("    lesion from background parenchymal enhancement")
        print("  - single post-contrast phase discards the enhancement dynamics")
        print("    that make lesions separable from BPE")
        print("  - BatchNorm at batch size 2 (nnU-Net uses InstanceNorm here)")
        print("  - 200 epochs against nnU-Net's 1000")

    df.to_csv("component_analysis.csv", index=False)
    print("\nper-case detail written to component_analysis.csv")


if __name__ == "__main__":
    main()
