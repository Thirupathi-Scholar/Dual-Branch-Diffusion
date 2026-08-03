#!/usr/bin/env python3
"""
rescore.py — recompute everything at the paper's operating point, CPU only.

The overnight run saved probability maps to ~/probs (test) and ~/probs_val
(validation). This script reads those and redoes the analysis at whatever
thresholds you want, with NO GPU and NO model. It exists because the overnight
run defaulted to threshold 0.5 for the metric table, while the paper's tuned
operating point is 0.98.

Run it on the instance, or download ~/probs and run it on a laptop.

    python rescore.py --probs ~/probs --audit ~/orientation_audit.json \
                      --out ~/results_rescored

    # validation, for honest rule selection:
    python rescore.py --probs ~/probs_val --audit ~/orientation_audit.json \
                      --out ~/results_rescored_val

Key difference from the overnight run: thresholds include 0.98 and 0.99, and
the failure/success decomposition is reported at the real base rate rather
than whatever the sample happened to contain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

TARGET_SPACING = (1.5, 1.5, 2.0)
CONF_CUTS = (0.9990, 0.9995, 0.9999)
DOM_CUTS = (0.30, 0.50, 0.70)


def true_spacing_map(audit_path, target=TARGET_SPACING):
    audit = json.loads(Path(audit_path).expanduser().read_text())
    out = {}
    for case, v in audit.items():
        if "zooms_used" not in v:
            continue
        zu, zc = v["zooms_used"], v["zooms_correct"]
        try:
            out[case] = tuple(float(target[i] * zc[i] / zu[i]) if zu[i]
                              else float(target[i]) for i in range(3))
        except (IndexError, ZeroDivisionError):
            out[case] = tuple(float(x) for x in target)
    return out


def surface_distances(pred, gt, spacing):
    """HD95 and MSD in mm, symmetric, via distance transforms."""
    if not pred.any() or not gt.any():
        return float("nan"), float("nan")
    pe = pred ^ ndimage.binary_erosion(pred)
    ge = gt ^ ndimage.binary_erosion(gt)
    if not pe.any() or not ge.any():
        return float("nan"), float("nan")
    dt_g = ndimage.distance_transform_edt(~ge, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~pe, sampling=spacing)
    d_pg, d_gp = dt_g[pe], dt_p[ge]
    all_d = np.concatenate([d_pg, d_gp])
    return float(np.percentile(all_d, 95)), float(all_d.mean())


def components(prob, gt, threshold, min_voxels=10):
    mask = prob > threshold
    if not mask.any():
        return []
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    gt_sum = int(gt.sum())
    flat = lab.ravel()
    sizes = np.bincount(flat, minlength=n + 1)
    inter = np.bincount(flat[gt.ravel().astype(bool)], minlength=n + 1)
    idx = np.arange(1, n + 1)
    peak = ndimage.maximum(prob, lab, index=idx)
    meanp = ndimage.mean(prob, lab, index=idx)
    rows = []
    for k in idx:
        v = int(sizes[k])
        if v < min_voxels:
            continue
        d = 2.0 * int(inter[k]) / (v + gt_sum) if (v + gt_sum) else 0.0
        rows.append(dict(comp=int(k), voxels=v, dice=d,
                         peak_prob=float(peak[k - 1]),
                         mean_prob=float(meanp[k - 1])))
    return rows


def rules(rows):
    keys = (["largest", "oracle", "peak_prob", "mean_prob", "prob_mass",
             "cond_disagree"]
            + [f"cond_lowconf_{c}" for c in CONF_CUTS]
            + [f"cond_nodom_{d}" for d in DOM_CUTS])
    if not rows:
        out = {k: 0.0 for k in keys}
        out.update(n_components=0, largest_peak=0.0,
                   largest_vol_frac=0.0, disagree=False, oracle_rank=None)
        return out
    by_vol = sorted(rows, key=lambda r: -r["voxels"])
    lg, pk = by_vol[0], max(rows, key=lambda r: r["peak_prob"])
    best = max(rows, key=lambda r: r["dice"])
    tot = sum(r["voxels"] for r in rows)
    frac = lg["voxels"] / tot if tot else 0.0
    dis = lg["comp"] != pk["comp"]
    out = dict(largest=lg["dice"], oracle=best["dice"], peak_prob=pk["dice"],
               mean_prob=max(rows, key=lambda r: r["mean_prob"])["dice"],
               prob_mass=max(rows, key=lambda r: r["mean_prob"]*r["voxels"])["dice"],
               n_components=len(rows), largest_peak=lg["peak_prob"],
               largest_vol_frac=frac, disagree=dis,
               oracle_rank=by_vol.index(best) + 1,
               cond_disagree=pk["dice"] if dis else lg["dice"])
    for c in CONF_CUTS:
        out[f"cond_lowconf_{c}"] = pk["dice"] if lg["peak_prob"] < c else lg["dice"]
    for d in DOM_CUTS:
        out[f"cond_nodom_{d}"] = pk["dice"] if frac < d else lg["dice"]
    return out


def largest_cc(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(np.argmax(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--out", default="results_rescored")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 0.8, 0.95, 0.98, 0.99])
    ap.add_argument("--operating-point", type=float, default=0.98,
                    help="the paper's tuned threshold; metrics table uses this")
    a = ap.parse_args()

    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    pdir = Path(a.probs).expanduser()
    files = sorted(pdir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {pdir}")
    print(f"{len(files)} probability maps in {pdir}")

    sp_map = true_spacing_map(a.audit) if a.audit else {}

    crows, mrows = [], []
    for i, f in enumerate(files):
        case = f.stem
        z = np.load(f)
        prob = z["prob"].astype(np.float32)
        gt = z["gt"].astype(bool)

        for th in a.thresholds:
            r = rules(components(prob, gt, th))
            r.update(case=case, threshold=th)
            crows.append(r)

        pred = largest_cc(prob > a.operating_point)
        inter = int((pred & gt).sum()); ps, gs = int(pred.sum()), int(gt.sum())
        dice = 2*inter/(ps+gs) if (ps+gs) else 0.0
        iou = inter/(ps+gs-inter) if (ps+gs-inter) else 0.0
        sp_nom = TARGET_SPACING
        sp_true = sp_map.get(case, sp_nom)
        h_n, m_n = surface_distances(pred, gt, sp_nom)
        corrected = tuple(round(x,4) for x in sp_true) != tuple(round(x,4) for x in sp_nom)
        h_t, m_t = surface_distances(pred, gt, sp_true) if corrected else (h_n, m_n)
        mrows.append(dict(case=case, dice=dice, iou=iou,
                          hd95_nominal=h_n, msd_nominal=m_n,
                          hd95_true=h_t, msd_true=m_t,
                          spacing_corrected=corrected,
                          vol_err_pct=100*(ps-gs)/gs if gs else np.nan,
                          gt_voxels=gs))
        if (i+1) % 50 == 0 or i+1 == len(files):
            print(f"  {i+1}/{len(files)}")

    cdf, mdf = pd.DataFrame(crows), pd.DataFrame(mrows)
    cdf.to_csv(out/"component_selection_by_threshold.csv", index=False)
    mdf.to_csv(out/"metrics_spacing_corrected.csv", index=False)

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    op = a.operating_point
    P("\n" + "="*72)
    P(f"OPERATING POINT {op}  (n={len(mdf)})")
    P("="*72)
    P(f"Dice {mdf.dice.mean():.4f}   IoU {mdf.iou.mean():.4f}   "
      f"failures <0.01: {int((mdf.dice<0.01).sum())} "
      f"({100*(mdf.dice<0.01).mean():.1f}%)")
    nc = int(mdf.spacing_corrected.sum())
    P(f"\nspacing-corrected cases: {nc}/{len(mdf)} ({100*nc/len(mdf):.1f}%)")
    P(f"{'metric':<12}{'nominal':>12}{'true':>12}{'delta':>10}")
    P(f"{'HD95 (mm)':<12}{mdf.hd95_nominal.mean():>12.4f}"
      f"{mdf.hd95_true.mean():>12.4f}{mdf.hd95_true.mean()-mdf.hd95_nominal.mean():>+10.4f}")
    P(f"{'MSD (mm)':<12}{mdf.msd_nominal.mean():>12.4f}"
      f"{mdf.msd_true.mean():>12.4f}{mdf.msd_true.mean()-mdf.msd_nominal.mean():>+10.4f}")
    if nc:
        s = mdf[mdf.spacing_corrected]
        P(f"  affected only (n={len(s)}): HD95 {s.hd95_nominal.mean():.3f} -> "
          f"{s.hd95_true.mean():.3f}   MSD {s.msd_nominal.mean():.3f} -> "
          f"{s.msd_true.mean():.3f}")

    rule_cols = [c for c in cdf.columns if c.startswith("cond_")] + \
                ["peak_prob", "mean_prob", "prob_mass"]
    P("\n" + "="*72)
    P("SELECTION RULES")
    P("="*72)
    for th in a.thresholds:
        g = cdf[cdf.threshold == th]
        if g.empty: continue
        gap = g.oracle.mean() - g.largest.mean()
        P(f"\nthreshold {th:.2f}   largest {g.largest.mean():.4f}   "
          f"oracle {g.oracle.mean():.4f}   gap {gap:+.4f}   "
          f"median components {g.n_components.median():.0f}")
        for r in sorted(rule_cols, key=lambda c: -g[c].mean()):
            d = g[r].mean() - g.largest.mean()
            flag = "  <-- beats largest" if d > 0.005 else ""
            P(f"  {r:<24}{g[r].mean():>9.4f}{d:>+10.4f}"
              f"{100*d/gap if gap>1e-9 else float('nan'):>8.1f}% of gap{flag}")

    P("\n" + "="*72)
    P(f"FAILURE / SUCCESS DECOMPOSITION at threshold {op}")
    P("="*72)
    g = cdf[cdf.threshold == op]
    if not g.empty:
        fail, succ = g[g.largest < 0.01], g[g.largest >= 0.30]
        base = len(fail)/len(g)
        P(f"base failure rate in this split: {100*base:.1f}%  "
          f"(n_fail={len(fail)}, n_succ={len(succ)})")
        P(f"\n{'rule':<24}{'on failures':>14}{'on successes':>15}{'net':>10}")
        for r in rule_cols:
            df_ = fail[r].mean() - fail.largest.mean() if len(fail) else 0.0
            ds_ = succ[r].mean() - succ.largest.mean() if len(succ) else 0.0
            net = g[r].mean() - g.largest.mean()
            P(f"{r:<24}{df_:>+14.4f}{ds_:>+15.4f}{net:>+10.4f}")
        P("\nA rule is publishable only if it gains on failures WITHOUT")
        P("materially losing on successes. A large net gain driven by a")
        P("failure-enriched sample will not replicate at the true base rate.")

    (out/"rescore_report.txt").write_text("\n".join(lines))
    P(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()

