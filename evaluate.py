"""
evaluate.py — test-set evaluation, statistics, and publication figures.

Runs inference for every trained configuration on the HELD-OUT TEST SPLIT,
computes per-case metrics, and produces everything the manuscript needs:

    results/
      per_case_metrics.csv        every case x every config
      table2_comparison.csv       method comparison with 95% CIs
      table3_ablation.csv         ablation with paired significance tests
      table_stratified_size.csv   performance by lesion size
      significance_tests.csv      full paired Wilcoxon output, Holm-corrected
      fig_training_curves.png
      fig_roc.png                 from REAL predictions
      fig_ablation.png
      summary.md

Usage:
    python evaluate.py --data /path/to/MAMA-MIA --runs runs --out results

Every number written here traces to a checkpoint and a CSV. That is the point.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
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

from verify_metrics import dice_iou, hd95_msd, volume_error_pct, summarize


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def largest_component(mask):
    """
    Keep only the largest connected component.

    Standard post-processing for single-lesion segmentation and used by
    nnU-Net. Without it, scattered false-positive islands dominate the
    Hausdorff distance even when the main lesion is found correctly —
    observed HD95 of ~146 mm on volumes only ~350 mm across.
    """
    from scipy import ndimage
    if mask.sum() == 0:
        return mask
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


@torch.no_grad()
def predict_case(model, image, device, patch=(96, 96, 32), overlap=0.5):
    """Sliding-window inference at full volume resolution."""
    from monai.inferers import sliding_window_inference
    x = image.unsqueeze(0).to(device)
    logits = sliding_window_inference(
        x, roi_size=patch, sw_batch_size=2,
        predictor=lambda t: model(t)["logits"], overlap=overlap, mode="gaussian",
    )
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def iter_test_cases(splits, n_channels=None):
    """
    Yield (case_id, image, label, spacing) for the test split.

    Manifests written by preprocess.py point at .npz volumes that are already
    resampled, reoriented, foreground-cropped and normalized, so evaluation
    reads them directly. Raw NIfTI manifests fall back to the MONAI pipeline.
    """
    import numpy as np

    for r in splits["test"]:
        if "npz" in r:
            z = np.load(r["npz"])
            from train import fix_channels
            img = torch.from_numpy(
                fix_channels(z["image"], n_channels)).float()
            lab = z["label"].astype(bool)
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
            spacing = tuple(meta.get("spacing", (1.5, 1.5, 2.0)))
            yield r["case"], img, lab, spacing
        else:
            from monai import transforms as T
            tf = T.Compose([
                T.LoadImaged(keys=["image", "label"]),
                T.EnsureChannelFirstd(keys=["image", "label"]),
                T.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
                           mode=("bilinear", "nearest")),
                T.Orientationd(keys=["image", "label"], axcodes="RAS"),
                T.ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000,
                                       b_min=0.0, b_max=1.0, clip=True),
                T.ToTensord(keys=["image", "label"]),
            ])
            d = tf(dict(image=r["image"], label=r["label"]))
            yield (r["case"], d["image"].float(),
                   d["label"][0].numpy() > 0.5, (1.5, 1.5, 2.0))


def evaluate_config(cfg_name, ckpt_path, splits, device,
                    patch, spacing=(1.5, 1.5, 2.0), threshold=0.5,
                    collect_probs=False, limit=None, postprocess=True):
    """Returns (per-case dataframe, flat prob array, flat label array)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, _ = build_from_ckpt(ckpt, device, patch)

    rows, probs, labels = [], [], []
    n_total = len(splits["test"])

    n_channels = 4 if splits.get("_phase") == "all" else 1
    for i, (case_id, img, gt, spacing) in enumerate(
            iter_test_cases(splits, n_channels)):
        if limit and i >= limit:
            break
        prob = predict_case(model, img, device, patch)
        pred = prob > threshold
        if postprocess:
            pred = largest_component(pred)

        d, iou = dice_iou(pred, gt)
        h, msd = hd95_msd(pred, gt, spacing)
        rows.append(dict(
            case=case_id, config=cfg_name, dice=d, iou=iou,
            hd95_mm=h, msd_mm=msd, vol_err_pct=volume_error_pct(pred, gt),
            gt_voxels=int(gt.sum()),
        ))

        if collect_probs:
            # subsample to keep the ROC tractable across 200+ volumes
            flat_p, flat_g = prob.ravel(), gt.ravel().astype(np.uint8)
            keep = np.random.default_rng(i).choice(
                flat_p.size, size=min(20000, flat_p.size), replace=False)
            probs.append(flat_p[keep])
            labels.append(flat_g[keep])

        if (i + 1) % 10 == 0:
            import numpy as _np
            running = _np.nanmean([r["dice"] for r in rows])
            print(f"    {cfg_name}: {i+1}/{limit or n_total} cases   "
                  f"running Dice {running:.4f}", flush=True)

    df = pd.DataFrame(rows)
    p = np.concatenate(probs) if probs else None
    g = np.concatenate(labels) if labels else None
    return df, p, g


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def paired_tests(df, reference="full", metric="dice", alpha=0.05):
    """
    Paired Wilcoxon signed-rank of every configuration against the reference,
    with Holm-Bonferroni correction across the family of comparisons.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    ref = df[df.config == reference].set_index("case")[metric]
    out = []
    for cfg in sorted(df.config.unique()):
        if cfg == reference:
            continue
        other = df[df.config == cfg].set_index("case")[metric]
        common = ref.index.intersection(other.index)
        a, b = ref.loc[common].values, other.loc[common].values
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 5:
            continue
        try:
            stat, p = wilcoxon(a, b)
        except ValueError:
            stat, p = float("nan"), 1.0
        out.append(dict(config=cfg, n=len(a), mean_ref=a.mean(),
                        mean_cfg=b.mean(), delta=b.mean() - a.mean(),
                        statistic=stat, p_raw=p))

    if not out:
        return pd.DataFrame()
    res = pd.DataFrame(out)
    rej, p_adj, _, _ = multipletests(res.p_raw.values, alpha=alpha, method="holm")
    res["p_holm"] = p_adj
    res["significant"] = rej
    return res


def size_strata(df, small_q=0.33, large_q=0.67):
    """Split by ground-truth lesion volume. Thresholds are data-driven and reported."""
    d = df[df.config == "full"].copy()
    lo, hi = d.gt_voxels.quantile(small_q), d.gt_voxels.quantile(large_q)

    def label(v):
        return "Small" if v <= lo else ("Large" if v >= hi else "Medium")

    d["stratum"] = d.gt_voxels.map(label)
    rows = []
    for s in ["Small", "Medium", "Large"]:
        sub = d[d.stratum == s]
        row = dict(stratum=s, n=len(sub))
        for m in ["dice", "iou", "hd95_mm", "msd_mm", "vol_err_pct"]:
            st = summarize(sub[m].tolist())
            row[f"{m}_mean"] = st["mean"]
            row[f"{m}_sd"] = st["sd"]
            row[f"{m}_ci"] = f"[{st['ci_lo']:.3f}, {st['ci_hi']:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows), dict(small_max_voxels=float(lo),
                                    large_min_voxels=float(hi))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_training_curves(runs_dir, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = Path(runs_dir) / "full" / "log.csv"
    if not log.exists():
        print("  [skip] training curves — no log found")
        return
    df = pd.read_csv(log)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(df.epoch, df.train_dice, label="Training")
    ax[0].plot(df.epoch, df.val_dice, label="Validation")
    ax[0].set(xlabel="Epoch", ylabel="Dice", title="Dice")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(df.epoch, df.train_total, label="Training")
    ax[1].plot(df.epoch, df.val_total, label="Validation")
    ax[1].set(xlabel="Epoch", ylabel="Loss", title="Total loss")
    ax[1].legend(); ax[1].grid(alpha=.3)

    for c, lbl in [("train_seg", "L_seg"), ("train_diff", "L_diff"),
                   ("train_conf", "L_conf")]:
        if c in df:
            ax[2].plot(df.epoch, df[c], label=lbl)
    ax[2].set(xlabel="Epoch", ylabel="Loss", title="Loss components")
    ax[2].legend(); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_roc(probs, labels, out_path):
    """ROC from real held-out predictions. Note the title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    if probs is None:
        print("  [skip] ROC — no probabilities collected")
        return None
    fpr, tpr, _ = roc_curve(labels, probs)
    a = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"DualDiffSeg (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Voxel-wise ROC, held-out test set")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    return float(a)


def fig_ablation(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [c for c in M.ABLATIONS if c in df.config.unique()]
    means = [df[df.config == c].dice.mean() for c in order]
    errs = [df[df.config == c].dice.std() for c in order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#c44" if c == "full" else "#789" for c in order]
    ax.bar(order, means, yerr=errs, capsize=4, color=colors)
    ax.set(ylabel="Test Dice", title="Ablation study")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", required=True,
                    help="directory written by prepare_data.py")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 32])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-postprocess", action="store_true",
                    help="disable largest-connected-component filtering")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N test cases (quick check)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import train as TR
    splits = TR.load_manifests(args.manifests)
    print(f"test set: {len(splits['test'])} cases (official MAMA-MIA test split)\n")

    frames, probs, labels = [], None, None
    for cfg in M.ABLATIONS:
        ckpt = Path(args.runs) / cfg / "best.pt"
        if not ckpt.exists():
            print(f"[skip] {cfg} — no checkpoint")
            continue
        print(f"evaluating {cfg} ...")
        df, p, g = evaluate_config(cfg, ckpt, splits, device,
                                   tuple(args.patch), threshold=args.threshold,
                                   collect_probs=(cfg == "full"),
                                   limit=args.limit,
                                   postprocess=not args.no_postprocess)
        frames.append(df)
        if cfg == "full":
            probs, labels = p, g

    if not frames:
        raise SystemExit("no checkpoints found")

    allm = pd.concat(frames, ignore_index=True)
    allm.to_csv(out / "per_case_metrics.csv", index=False)

    # ---- comparison table ---------------------------------------------------
    rows = []
    for cfg in sorted(allm.config.unique()):
        sub = allm[allm.config == cfg]
        row = dict(config=cfg, n=len(sub))
        for m in ["dice", "iou", "hd95_mm", "msd_mm"]:
            st = summarize(sub[m].tolist())
            row[f"{m}"] = f"{st['mean']:.4f} ± {st['sd']:.4f}"
            row[f"{m}_ci95"] = f"[{st['ci_lo']:.4f}, {st['ci_hi']:.4f}]"
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "table2_comparison.csv", index=False)

    # ---- significance -------------------------------------------------------
    sig = paired_tests(allm)
    if not sig.empty:
        sig.to_csv(out / "significance_tests.csv", index=False)

    abl = comp.copy()
    abl.to_csv(out / "table3_ablation.csv", index=False)

    # ---- stratified ---------------------------------------------------------
    strat, thresholds = size_strata(allm)
    strat.to_csv(out / "table_stratified_size.csv", index=False)

    # ---- figures ------------------------------------------------------------
    print("\ngenerating figures ...")
    fig_training_curves(args.runs, out / "fig_training_curves.png")
    auc_val = fig_roc(probs, labels, out / "fig_roc.png")
    fig_ablation(allm, out / "fig_ablation.png")

    # ---- summary ------------------------------------------------------------
    full = allm[allm.config == "full"]
    st = summarize(full.dice.tolist())
    hd = summarize(full.hd95_mm.tolist())

    lines = [
        "# DualDiffSeg — test set results",
        "",
        f"Held-out test set: **{len(splits['test'])} cases**, patient-level split.",
        "All numbers below are computed by `evaluate.py` from saved checkpoints.",
        "",
        "## Headline",
        "",
        f"- Dice: **{st['mean']:.4f} ± {st['sd']:.4f}** "
        f"(95% CI [{st['ci_lo']:.4f}, {st['ci_hi']:.4f}])",
        f"- HD95: **{hd['mean']:.2f} ± {hd['sd']:.2f} mm** "
        f"(95% CI [{hd['ci_lo']:.2f}, {hd['ci_hi']:.2f}])",
    ]
    if auc_val is not None:
        lines.append(f"- Voxel-wise AUC: **{auc_val:.4f}** (real predictions, test set)")

    lines += [
        "",
        "## Reference point",
        "",
        "Published nnU-Net on MAMA-MIA (dataset authors, 5-fold, full-image):",
        "Dice 0.7620, IoU 0.6539, HD95 37.41 mm, MSD 11.08 mm.",
        "",
        f"Margin over that reference: **{st['mean'] - 0.7620:+.4f} Dice**.",
        "",
        "## Size strata thresholds",
        "",
        f"- Small: <= {thresholds['small_max_voxels']:.0f} foreground voxels",
        f"- Large: >= {thresholds['large_min_voxels']:.0f} foreground voxels",
        "",
        "## Files",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `per_case_metrics.csv` | every case x every configuration |",
        "| `table2_comparison.csv` | means, SDs, bootstrap 95% CIs |",
        "| `table3_ablation.csv` | ablation results |",
        "| `significance_tests.csv` | paired Wilcoxon, Holm-corrected |",
        "| `table_stratified_size.csv` | performance by lesion size |",
        "",
        "Report these numbers as measured. Every one is traceable to a checkpoint.",
    ]
    (out / "summary.md").write_text("\n".join(lines))

    print("\n" + "=" * 66)
    print(f"Dice  {st['mean']:.4f} ± {st['sd']:.4f}   "
          f"95% CI [{st['ci_lo']:.4f}, {st['ci_hi']:.4f}]")
    print(f"HD95  {hd['mean']:.2f} ± {hd['sd']:.2f} mm")
    print(f"vs published nnU-Net 0.7620:  {st['mean'] - 0.7620:+.4f} Dice")
    print("=" * 66)
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
