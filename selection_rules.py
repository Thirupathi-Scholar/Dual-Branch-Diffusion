"""
selection_rules.py — choose the right connected component.

The problem
-----------
Component analysis on validation showed:

  largest-component selection   0.5965
  highest-peak-probability      0.5603
  highest-mean-probability      0.5013
  ORACLE (best available)       0.7044

Neither simple rule works. Largest fails when a big low-confidence blob of
background parenchymal enhancement outsizes the lesion. Peak fails when a tiny
high-confidence speck outranks it. The correct component ranks 2nd by size and
1st by peak among FAILING cases, but that ordering reverses on cases that
already succeed — so a single-criterion rule trades one failure mode for the
other.

Total probability mass — the sum of predicted probability over a component —
balances both: it is the model's own estimate of how many tumor voxels the
component contains. A large confident region beats both a small confident
speck and a large uncertain blob.

This script evaluates several rules on VALIDATION and reports which comes
closest to the oracle. The winner is then applied unchanged to test.

    python selection_rules.py --checkpoint runs/full/best.pt \\
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


def evaluate_rules(prob, gt, thr, min_size=20, open_radius=1):
    """Return {rule_name: dice} for one case."""
    from scipy import ndimage

    mask = prob > thr
    if mask.sum() == 0:
        return {}

    # morphological opening removes single-voxel speckle before labelling
    opened = ndimage.binary_opening(
        mask, ndimage.generate_binary_structure(3, 1), iterations=open_radius)
    if opened.sum() == 0:
        opened = mask

    out = {}

    for tag, m in [("", mask), ("_open", opened)]:
        lab, n = ndimage.label(m)
        if n == 0:
            continue

        comps = []
        for i in range(1, n + 1):
            c = lab == i
            size = int(c.sum())
            pv = prob[c]
            comps.append(dict(mask=c, size=size, peak=float(pv.max()),
                              mean=float(pv.mean()), mass=float(pv.sum())))

        big = [c for c in comps if c["size"] >= min_size] or comps

        rules = {
            f"largest{tag}":        max(comps, key=lambda c: c["size"]),
            f"peak{tag}":           max(comps, key=lambda c: c["peak"]),
            f"mass{tag}":           max(comps, key=lambda c: c["mass"]),
            f"size_x_mean{tag}":    max(comps, key=lambda c: c["size"] * c["mean"]),
            f"minsize_peak{tag}":   max(big,   key=lambda c: c["peak"]),
            f"minsize_mass{tag}":   max(big,   key=lambda c: c["mass"]),
            f"minsize_largest{tag}": max(big,  key=lambda c: c["size"]),
        }
        for name, c in rules.items():
            out[name] = dice(c["mask"], gt)

        if tag == "":
            out["oracle"] = max(dice(c["mask"], gt) for c in comps)
            out["n_components"] = n

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 32])
    ap.add_argument("--threshold", type=float, default=0.98)
    ap.add_argument("--min-size", type=int, default=20)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, _ = build_from_ckpt(ckpt, device, tuple(args.patch))

    records = json.loads(
        (Path(args.manifests) / f"{args.split}.json").read_text())["records"][:args.n]

    print(f"checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    print(f"split:      {args.split}, {len(records)} cases")
    print(f"threshold:  {args.threshold}   min component size: {args.min_size}\n")

    rows = []
    for i, r in enumerate(records):
        z = np.load(r["npz"])
        img = torch.from_numpy(z["image"]).float()
        gt = z["label"].astype(bool)
        prob = predict(model, img, device, tuple(args.patch))
        res = evaluate_rules(prob, gt, args.threshold, args.min_size)
        if res:
            res["case"] = r["case"]
            rows.append(res)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)

    rule_cols = [c for c in df.columns
                 if c not in ("case", "oracle", "n_components")]

    stats = []
    for c in rule_cols + ["oracle"]:
        v = df[c].dropna()
        stats.append(dict(rule=c, dice=v.mean(), median=v.median(),
                          zeros=int((v < 0.01).sum())))
    st = pd.DataFrame(stats).sort_values("dice", ascending=False)

    print(f"\n{'='*66}")
    print(f"SELECTION RULES  (n={len(df)}, median "
          f"{df.n_components.median():.0f} components per case)")
    print("=" * 66)
    print(f"{'rule':<22}{'mean Dice':>11}{'median':>10}{'zeros':>8}")
    print("-" * 66)
    for _, r in st.iterrows():
        mark = "  <- oracle ceiling" if r["rule"] == "oracle" else ""
        print(f"{r['rule']:<22}{r['dice']:>11.4f}{r['median']:>10.4f}"
              f"{int(r['zeros']):>8}{mark}")

    best = st[st.rule != "oracle"].iloc[0]
    baseline = st[st.rule == "largest"].iloc[0]
    oracle = st[st.rule == "oracle"].iloc[0]

    print(f"\n{'='*66}")
    print(f"best rule:  {best['rule']}   Dice {best['dice']:.4f}")
    print(f"current:    largest      Dice {baseline['dice']:.4f}")
    print(f"gain:       {best['dice'] - baseline['dice']:+.4f}")
    print(f"oracle:     {oracle['dice']:.4f}   "
          f"(closes {100*(best['dice']-baseline['dice'])/max(1e-9, oracle['dice']-baseline['dice']):.0f}% of the gap)")
    print("=" * 66)

    df.to_csv("selection_rules.csv", index=False)
    print("\nper-case detail written to selection_rules.csv")
    print("\nApply the winning rule uniformly to every configuration, state")
    print("that it was selected on validation, and re-run the test evaluation.")


if __name__ == "__main__":
    main()
