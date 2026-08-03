import argparse, json, glob, csv, os
from collections import Counter
from pathlib import Path
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
RAS = axcodes2ornt(("R", "A", "S"))

def audit(p):
    n = nib.load(str(p), mmap=True)
    o = ornt_transform(io_orientation(n.affine), RAS)
    order = tuple(int(x) for x in o[:, 0])
    z = tuple(round(float(v), 4) for v in n.header.get_zooms()[:3])
    return order != (0, 1, 2), order, z, tuple(z[i] for i in order)

def find_failures(thresh=0.01):
    """Sniff ~/results for a per-case CSV with case + dice columns."""
    best = (0, set())
    for f in glob.glob(os.path.expanduser("~/results/**/*.csv"), recursive=True):
        try:
            rows = list(csv.DictReader(open(f)))
        except Exception:
            continue
        if not rows: continue
        cols = {c.lower(): c for c in rows[0]}
        ck = next((cols[c] for c in cols if "case" in c or c in ("id", "subject")), None)
        dk = next((cols[c] for c in cols if c == "dice" or c.startswith("dice")), None)
        if not (ck and dk): continue
        fails = set()
        for r in rows:
            try:
                if float(r[dk]) < thresh: fails.add(r[ck].strip())
            except (ValueError, TypeError): pass
        if len(rows) > best[0]:
            best = (len(rows), fails)
            print(f"  using {f}  ({len(rows)} rows, {len(fails)} below {thresh})")
    return best[1]

ap = argparse.ArgumentParser()
ap.add_argument("--manifests", default=str(Path.home() / "data" / "manifests"))
ap.add_argument("--out", default=str(Path.home() / "orientation_audit.json"))
a = ap.parse_args()

print("locating failing cases:")
failing = find_failures()
print(f"failing set: {len(failing)}\n")

res, counts = {}, Counter()
for split in ("train", "val", "test"):
    f = Path(a.manifests) / f"{split}.json"
    if not f.exists():
        print(f"  (no {f})"); continue
    recs = json.loads(f.read_text())["records"]
    np_ = 0
    for r in recs:
        img = r["image"][0] if isinstance(r["image"], list) else r["image"]
        try:
            perm, order, zu, zc = audit(img)
        except Exception as e:
            res[r["case"]] = dict(split=split, error=str(e)[:120]); continue
        counts[order] += 1; np_ += perm
        res[r["case"]] = dict(split=split, permuted=bool(perm), axis_order=list(order),
                              zooms_used=list(zu), zooms_correct=list(zc))
    print(f"{split:<6} {len(recs):>5} cases  {np_:>4} transposed ({100*np_/max(len(recs),1):.1f}%)")

print("\naxis orders seen:")
for o, c in counts.most_common():
    print(f"  {o}: {c:>5}" + ("   <- correct" if o == (0,1,2) else "   <- TRANSPOSED"))

test = {c: v for c, v in res.items() if v.get("split") == "test" and "permuted" in v}
if failing and test:
    fp = sum(1 for c in failing if test.get(c, {}).get("permuted"))
    ft = sum(1 for c in failing if c in test)
    ok = [c for c in test if c not in failing]
    op = sum(1 for c in ok if test[c]["permuted"])
    print(f"\n--- cross-reference (test split) ---")
    print(f"failing     : {fp}/{ft} transposed ({100*fp/max(ft,1):.1f}%)")
    print(f"non-failing : {op}/{len(ok)} transposed ({100*op/max(len(ok),1):.1f}%)")
    try:
        from scipy.stats import fisher_exact
        odds, p = fisher_exact([[fp, ft-fp], [op, len(ok)-op]])
        print(f"Fisher exact: OR {odds:.2f}, p = {p:.3g}")
    except ImportError:
        print("(pip install scipy for the test)")
else:
    print("\n(no cross-reference — failing list or test manifest missing)")

Path(a.out).write_text(json.dumps(res, indent=1))
print(f"\nwritten: {a.out}")
