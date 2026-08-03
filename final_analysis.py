#!/usr/bin/env python3
"""
final_analysis.py — one inference pass, three outstanding results.

Run this from ~/ on the instance (same directory as train.py / evaluate.py /
verify_metrics.py). It runs sliding-window inference once over the 306 test
cases and from that single pass produces:

  1. Oracle connected-component selection at n=306      (scales up section 5.8)
  2. Selection performance across thresholds            (section 5.9)
  3. HD95 / MSD recomputed at TRUE per-case spacing     (section 4.2 / 3.6)

It also saves the probability maps, so nothing here ever has to be recomputed
on a GPU again — every further analysis can run locally off the .npz files.

Usage:
    python final_analysis.py \
        --manifests ~/data/manifests \
        --runs ~/runs \
        --audit ~/orientation_audit.json \
        --out ~/results_final \
        --probs ~/probs

    # quick smoke test on 10 cases before committing to the full run:
    python final_analysis.py ... --limit 10

Disk: probability maps are float16 + compressed, roughly 2-4 GB for 306 cases.
Pass --no-save-probs to skip, but then a rerun costs another GPU pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as TR
from evaluate import build_from_ckpt, iter_test_cases, predict_case
from verify_metrics import dice_iou, hd95_msd, volume_error_pct

TARGET_SPACING = (1.5, 1.5, 2.0)


# ---------------------------------------------------------------------------
# true spacing recovery
# ---------------------------------------------------------------------------
def true_spacing_map(audit_path, target=TARGET_SPACING):
    """
    preprocess.py computed resampling factors as zooms_in / target, where
    zooms_in came from header.get_zooms() in the ORIGINAL axis order, but
    applied them to an array already transposed to RAS by apply_orientation.

    For a case whose reorientation permutes axes, the factor on axis i was
    therefore derived from the wrong source zoom. The achieved spacing on
    axis i of the stored array is:

        achieved[i] = zooms_correct[i] / factor[i]
                    = target[i] * zooms_correct[i] / zooms_used[i]

    For untransposed cases zooms_correct == zooms_used and this returns the
    target, as it should.
    """
    audit = json.loads(Path(audit_path).expanduser().read_text())
    out = {}
    for case, v in audit.items():
        if "zooms_used" not in v:
            continue
        zu, zc = v["zooms_used"], v["zooms_correct"]
        try:
            out[case] = tuple(
                float(target[i] * zc[i] / zu[i]) if zu[i] else float(target[i])
                for i in range(3)
            )
        except (IndexError, ZeroDivisionError):
            out[case] = tuple(float(x) for x in target)
    return out


# ---------------------------------------------------------------------------
# component analysis
# ---------------------------------------------------------------------------
def component_table(prob, gt, threshold, min_voxels=10):
    """
    Label the thresholded mask and score every component against ground truth.
    Returns a list of dicts, one per surviving component.
    """
    mask = prob > threshold
    if not mask.any():
        return []
    lab, n = ndimage.label(mask)
    if n == 0:
        return []

    gt_sum = int(gt.sum())
    rows = []
    # bincount over labels is far faster than looping with boolean masks
    flat_lab = lab.ravel()
    sizes = np.bincount(flat_lab, minlength=n + 1)
    inter = np.bincount(flat_lab[gt.ravel()], minlength=n + 1)
    peak = ndimage.maximum(prob, lab, index=np.arange(1, n + 1))
    mean_p = ndimage.mean(prob, lab, index=np.arange(1, n + 1))

    for k in range(1, n + 1):
        v = int(sizes[k])
        if v < min_voxels:
            continue
        i = int(inter[k])
        d = 2.0 * i / (v + gt_sum) if (v + gt_sum) else 0.0
        rows.append(dict(comp=k, voxels=v, dice=d,
                         peak_prob=float(peak[k - 1]),
                         mean_prob=float(mean_p[k - 1])))
    return rows


CONF_CUTS = (0.9990, 0.9995, 0.9999)
DOM_CUTS = (0.30, 0.50, 0.70)


def selection_rules(rows):
    """
    Dice achieved by each selection rule, plus the oracle.

    Unconditional rules pick one component globally. Conditional rules keep
    the largest component by default and switch to the highest-peak-probability
    component only when the largest one looks like a failure.

    The motivation: on this data peak-probability selection recovers large
    amounts of Dice on FAILING cases while losing ground on successful ones,
    netting out to roughly zero. If failures can be detected from the
    prediction alone, the two effects can be separated.

    Three failure signals are tested, none of which use ground truth:
      disagree  - largest component is not the highest-peak-probability one
      lowconf   - largest component's peak probability is below a cutoff
      nodom     - largest component holds less than a fraction of total
                  predicted volume (no dominant candidate)
    """
    if not rows:
        base = dict(largest=0.0, oracle=0.0, peak_prob=0.0, mean_prob=0.0,
                    prob_mass=0.0, n_components=0, oracle_rank_by_volume=None,
                    largest_peak=0.0, largest_vol_frac=0.0, disagree=False)
        for c in CONF_CUTS:
            base[f"cond_lowconf_{c}"] = 0.0
        for d in DOM_CUTS:
            base[f"cond_nodom_{d}"] = 0.0
        base["cond_disagree"] = 0.0
        return base

    by_vol = sorted(rows, key=lambda r: -r["voxels"])
    lg = by_vol[0]
    pk = max(rows, key=lambda r: r["peak_prob"])
    best = max(rows, key=lambda r: r["dice"])
    total_vox = sum(r["voxels"] for r in rows)
    vol_frac = lg["voxels"] / total_vox if total_vox else 0.0
    disagree = lg["comp"] != pk["comp"]

    out = dict(
        largest=lg["dice"],
        oracle=best["dice"],
        peak_prob=pk["dice"],
        mean_prob=max(rows, key=lambda r: r["mean_prob"])["dice"],
        prob_mass=max(rows, key=lambda r: r["mean_prob"] * r["voxels"])["dice"],
        n_components=len(rows),
        oracle_rank_by_volume=by_vol.index(best) + 1,
        largest_peak=lg["peak_prob"],
        largest_vol_frac=vol_frac,
        disagree=disagree,
    )
    out["cond_disagree"] = pk["dice"] if disagree else lg["dice"]
    for c in CONF_CUTS:
        out[f"cond_lowconf_{c}"] = pk["dice"] if lg["peak_prob"] < c else lg["dice"]
    for d in DOM_CUTS:
        out[f"cond_nodom_{d}"] = pk["dice"] if vol_frac < d else lg["dice"]
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--audit", default=None,
                    help="orientation_audit.json — enables true-spacing metrics")
    ap.add_argument("--out", default="results_final")
    ap.add_argument("--probs", default="probs")
    ap.add_argument("--config", default="full")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.5, 0.8, 0.95])
    ap.add_argument("--metric-threshold", type=float, default=0.5,
                    help="threshold used for the corrected HD95/MSD table")
    ap.add_argument("--patch", type=int, nargs=3, default=[96, 96, 32])
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="which split to evaluate; val is used to SELECT "
                         "a selection rule, test to CONFIRM it")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-save-probs", action="store_true")
    a = ap.parse_args()

    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    pdir = Path(a.probs).expanduser()
    if not a.no_save_probs:
        pdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    spacing_by_case = {}
    if a.audit:
        spacing_by_case = true_spacing_map(a.audit)
        n_off = sum(1 for v in spacing_by_case.values()
                    if tuple(round(x, 4) for x in v) != TARGET_SPACING)
        print(f"true-spacing map: {len(spacing_by_case)} cases, "
              f"{n_off} with spacing != {TARGET_SPACING}")

    ck_path = Path(a.runs).expanduser() / a.config / "best.pt"
    if not ck_path.exists():
        sys.exit(f"checkpoint not found: {ck_path}")
    ckpt = torch.load(ck_path, map_location=device, weights_only=False)
    model, cfg = build_from_ckpt(ckpt, device, tuple(a.patch))
    n_ch = ckpt.get("in_ch", 1)
    patch = tuple(ckpt.get("patch", a.patch))
    print(f"loaded {cfg} from {ck_path}  (in_ch={n_ch}, patch={patch})")

    splits = TR.load_manifests(a.manifests)
    if a.split == "val":
        splits["test"] = splits["val"]          # iter_test_cases reads ["test"]
    total = len(splits["test"]) if a.limit is None else min(a.limit, len(splits["test"]))
    print(f"{a.split} cases: {total}\n")

    comp_rows, metric_rows = [], []
    t_start = time.time()

    for i, (case, img, gt, meta_spacing) in enumerate(
            iter_test_cases(splits, n_channels=n_ch)):
        if a.limit is not None and i >= a.limit:
            break

        prob = predict_case(model, img, device, patch)

        if not a.no_save_probs:
            np.savez_compressed(pdir / f"{case}.npz",
                                prob=prob.astype(np.float16),
                                gt=gt.astype(np.uint8))

        # ---- 1 & 2: components and selection rules across thresholds ----
        for th in a.thresholds:
            rows = component_table(prob, gt, th)
            sel = selection_rules(rows)
            sel.update(case=case, threshold=th)
            comp_rows.append(sel)

        # ---- 3: metrics at nominal vs true spacing ----
        pred = prob > a.metric_threshold
        lab_c, n_c = ndimage.label(pred)
        if n_c > 1:
            keep = 1 + int(np.argmax(np.bincount(lab_c.ravel())[1:]))
            pred_lcc = lab_c == keep
        else:
            pred_lcc = pred

        d, iou = dice_iou(pred_lcc, gt)
        sp_nom = tuple(meta_spacing)
        sp_true = spacing_by_case.get(case, sp_nom)
        h_nom, m_nom = hd95_msd(pred_lcc, gt, sp_nom)
        if tuple(round(x, 4) for x in sp_true) != tuple(round(x, 4) for x in sp_nom):
            h_true, m_true = hd95_msd(pred_lcc, gt, sp_true)
        else:
            h_true, m_true = h_nom, m_nom

        metric_rows.append(dict(
            case=case, dice=d, iou=iou,
            hd95_nominal=h_nom, msd_nominal=m_nom,
            hd95_true=h_true, msd_true=m_true,
            spacing_nominal=str(tuple(round(x, 4) for x in sp_nom)),
            spacing_true=str(tuple(round(x, 4) for x in sp_true)),
            spacing_corrected=tuple(round(x, 4) for x in sp_true) !=
                              tuple(round(x, 4) for x in sp_nom),
            vol_err_pct=volume_error_pct(pred_lcc, gt),
            gt_voxels=int(gt.sum()),
        ))

        if (i + 1) % 20 == 0 or i + 1 == total:
            el = time.time() - t_start
            print(f"  {i+1:>4}/{total}  {el/60:5.1f} min elapsed  "
                  f"~{el/(i+1)*(total-i-1)/60:5.1f} min left")

    cdf = pd.DataFrame(comp_rows)
    mdf = pd.DataFrame(metric_rows)
    cdf.to_csv(out / "component_selection_by_threshold.csv", index=False)
    mdf.to_csv(out / "metrics_spacing_corrected.csv", index=False)

    # ---------------- report ----------------
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("\n" + "=" * 68)
    P(f"ORACLE COMPONENT SELECTION  (config={cfg}, n={len(mdf)})")
    P("=" * 68)
    P(f"{'thresh':>7} {'ncomp':>7} {'largest':>9} {'peak_p':>9} "
      f"{'mean_p':>9} {'mass':>9} {'oracle':>9} {'gap':>8}")
    for th in a.thresholds:
        g = cdf[cdf.threshold == th]
        if g.empty:
            continue
        P(f"{th:>7.2f} {g.n_components.median():>7.0f} {g.largest.mean():>9.4f} "
          f"{g.peak_prob.mean():>9.4f} {g.mean_prob.mean():>9.4f} "
          f"{g.prob_mass.mean():>9.4f} {g.oracle.mean():>9.4f} "
          f"{g.oracle.mean()-g.largest.mean():>+8.4f}")

    rule_cols = ([c for c in cdf.columns if c.startswith("cond_")]
                 + ["peak_prob", "mean_prob", "prob_mass"])

    P("\nBest realisable rule at each threshold (vs largest-component):")
    winners = []
    for th in a.thresholds:
        g = cdf[cdf.threshold == th]
        if g.empty:
            continue
        cand = {r: g[r].mean() for r in rule_cols}
        best_r = max(cand, key=cand.get)
        delta = cand[best_r] - g.largest.mean()
        flag = "  <-- BEATS largest" if delta > 0.005 else ""
        winners.append((th, best_r, cand[best_r], delta))
        P(f"  {th:>5.2f}: {best_r:<22} {cand[best_r]:.4f} "
          f"({delta:+.4f} vs largest){flag}")

    P("\nCONDITIONAL RULES — full breakdown")
    P("(largest-component by default; switch to peak-probability on a "
      "failure signal)")
    for th in a.thresholds:
        g = cdf[cdf.threshold == th]
        if g.empty:
            continue
        P(f"\n  threshold {th:.2f}   largest {g.largest.mean():.4f}   "
          f"oracle {g.oracle.mean():.4f}")
        P(f"    {'rule':<24}{'Dice':>9}{'vs largest':>12}"
          f"{'% of oracle gap':>17}{'switched':>10}")
        gap = g.oracle.mean() - g.largest.mean()
        for r in rule_cols:
            d = g[r].mean() - g.largest.mean()
            if r == "cond_disagree":
                n_sw = int(g.disagree.sum())
            elif r.startswith("cond_lowconf_"):
                n_sw = int((g.largest_peak < float(r.split("_")[-1])).sum())
            elif r.startswith("cond_nodom_"):
                n_sw = int((g.largest_vol_frac < float(r.split("_")[-1])).sum())
            else:
                n_sw = len(g)
            pct = 100 * d / gap if gap > 1e-9 else float("nan")
            P(f"    {r:<24}{g[r].mean():>9.4f}{d:>+12.4f}"
              f"{pct:>16.1f}%{n_sw:>10}")

    # does the winning rule help failures without hurting successes?
    th_m = a.metric_threshold if a.metric_threshold in a.thresholds else a.thresholds[0]
    g = cdf[cdf.threshold == th_m]
    if not g.empty:
        best_r = max({r: g[r].mean() for r in rule_cols},
                     key=lambda r: g[r].mean())
        fail = g[g.largest < 0.01]
        succ = g[g.largest >= 0.30]
        P(f"\n  decomposition of '{best_r}' at threshold {th_m}:")
        P(f"    on largest-component FAILURES (<0.01, n={len(fail)}): "
          f"{fail.largest.mean():.4f} -> {fail[best_r].mean():.4f} "
          f"({fail[best_r].mean()-fail.largest.mean():+.4f})")
        P(f"    on largest-component SUCCESSES (>=0.30, n={len(succ)}): "
          f"{succ.largest.mean():.4f} -> {succ[best_r].mean():.4f} "
          f"({succ[best_r].mean()-succ.largest.mean():+.4f})")
        P("    A rule that gains on failures without losing on successes is "
          "the publishable one.")

    P("\n  NOTE: these rules are being fitted and evaluated on the same split. "
      "Any rule that wins here must be re-selected on validation and confirmed "
      "on test before it goes in the paper as a method.")

    P(f"\nOracle rank by volume (threshold {a.metric_threshold}): "
      f"median {cdf[cdf.threshold==a.metric_threshold].oracle_rank_by_volume.median():.0f}")

    P("\n" + "=" * 68)
    P("HD95 / MSD — NOMINAL vs TRUE SPACING")
    P("=" * 68)
    nc = int(mdf.spacing_corrected.sum())
    P(f"cases with corrected spacing: {nc}/{len(mdf)} ({100*nc/max(len(mdf),1):.1f}%)")
    P(f"{'metric':<10}{'nominal':>12}{'true':>12}{'delta':>10}")
    for a_, b_, nm in [("hd95_nominal", "hd95_true", "HD95 (mm)"),
                       ("msd_nominal", "msd_true", "MSD (mm)")]:
        P(f"{nm:<10}{mdf[a_].mean():>12.4f}{mdf[b_].mean():>12.4f}"
          f"{mdf[b_].mean()-mdf[a_].mean():>+10.4f}")
    if nc:
        sub = mdf[mdf.spacing_corrected]
        P(f"\naffected cases only (n={len(sub)}):")
        P(f"{'HD95':<10}{sub.hd95_nominal.mean():>12.4f}"
          f"{sub.hd95_true.mean():>12.4f}{sub.hd95_true.mean()-sub.hd95_nominal.mean():>+10.4f}")
        P(f"{'MSD':<10}{sub.msd_nominal.mean():>12.4f}"
          f"{sub.msd_true.mean():>12.4f}{sub.msd_true.mean()-sub.msd_nominal.mean():>+10.4f}")

    P(f"\nDice (largest-component, threshold {a.metric_threshold}): "
      f"{mdf.dice.mean():.4f}   failures <0.01: "
      f"{int((mdf.dice<0.01).sum())}/{len(mdf)}")

    (out / "final_analysis_report.txt").write_text("\n".join(lines))
    P(f"\nwritten to {out}/")
    P("  component_selection_by_threshold.csv")
    P("  metrics_spacing_corrected.csv")
    P("  final_analysis_report.txt")
    if not a.no_save_probs:
        P(f"probability maps: {pdir}/  ({len(mdf)} files)")
        P("Keep these — every further analysis can run off them without a GPU.")
    P(f"\ntotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()

