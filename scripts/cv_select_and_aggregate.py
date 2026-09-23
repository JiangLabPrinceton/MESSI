#!/usr/bin/env python
"""Wave-3 T4: goal-metric checkpoint selection + 5-fold CV aggregation.

Consumes a directory of ``benchmark_iedb_test.py`` JSONs named
``bench_cvf{k}_{val|test}_{arm}_{e<N>|best}.json`` (head metrics; NMP reused from the
shared cache). For each (arm, fold) it selects the epoch by the **goal metric on the
fold's VAL set** — argmax IoU50 AP, tie-broken by IoU70 then residue Pearson — subject
to a *loose* exact-AP collapse guard (exclude epochs whose val exact-AP fell below
``--exact-floor-frac`` × the run's max val exact-AP). It then reports that epoch's
metrics on the held-out **TEST** fold, and aggregates mean ± std across the 5 folds.

Selection leads on the region/density goal; exact-AP is a visible sanity guardrail,
never a co-primary (per project decision 2026-06-20). NMP columns (if present) are the
reference line. Prints a markdown table and writes a JSON summary.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path


def _read_macro(path: Path):
    """Stream out only the small ``.macro`` block via jq.

    The benchmark JSONs carry per-protein score_distributions and can be ~400MB
    each; loading them whole OOMs. jq extracts the tiny macro object in seconds.
    """
    try:
        out = subprocess.run(["jq", "-c", ".macro", str(path)],
                             capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return json.loads(out.stdout)
    except Exception:
        return None

FN = re.compile(r"^bench_cvf(\d+)_(val|test)_([a-z0-9]+)_(e\d+|best)\.json$")
GOAL = ("iou50", "iou70", "pears", "exap")


def _dig(d, *ks):
    for k in ks:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if isinstance(d, (int, float)) else None


def _head(macro):
    return {
        "iou50": _dig(macro, "iou_ladder", "head", "iou_0p50", "pp_ap"),
        "iou70": _dig(macro, "iou_ladder", "head", "iou_0p70", "pp_ap"),
        "pears": _dig(macro, "residue", "head", "pp_pearson"),
        "spear": _dig(macro, "residue", "head", "pp_spearman"),
        # residue landscape AUC/AP — the downstream-critical metric (GATE-1
        # no-regression). The flow consumes the per-residue hotspot field.
        "rauc":  _dig(macro, "residue", "head", "pp_auc"),
        "rap":   _dig(macro, "residue", "head", "pp_ap"),
        "exap":  _dig(macro, "iou_ladder", "head", "exact", "pp_ap"),
    }


def _nmp(macro):
    return {
        "iou50": _dig(macro, "iou_ladder", "nmp", "iou_0p50", "pp_ap"),
        "pears": _dig(macro, "residue", "nmp", "pp_pearson"),
        "rauc":  _dig(macro, "residue", "nmp", "pp_auc"),
        "rap":   _dig(macro, "residue", "nmp", "pp_ap"),
        "exap":  _dig(macro, "iou_ladder", "nmp", "exact", "pp_ap"),
    }


def load(eval_dir: Path):
    # data[(arm, fold)][split][tag] = head-metric dict ; nmp[(arm,fold)][split][tag]
    data = defaultdict(lambda: defaultdict(dict))
    nmp = defaultdict(lambda: defaultdict(dict))
    for f in eval_dir.glob("bench_cvf*_*.json"):
        m = FN.match(f.name)
        if not m:
            continue
        k, split, arm, tag = m.groups()
        macro = _read_macro(f)
        if macro is None:
            print(f"[warn] could not read macro from {f.name}")
            continue
        data[(arm, int(k))][split][tag] = _head(macro)
        nmp[(arm, int(k))][split][tag] = _nmp(macro)
    return data, nmp


def select_epoch(val_by_tag: dict, exact_floor_frac: float) -> str | None:
    epochs = [t for t in val_by_tag if t != "best"]
    if not epochs:
        return None
    exaps = [val_by_tag[t]["exap"] for t in epochs if val_by_tag[t]["exap"] is not None]
    floor = exact_floor_frac * max(exaps) if exaps else 0.0
    cand = [t for t in epochs if (val_by_tag[t]["exap"] or 0.0) >= floor]
    cand = cand or epochs
    return max(cand, key=lambda t: (val_by_tag[t]["iou50"] or -9,
                                    val_by_tag[t]["iou70"] or -9,
                                    val_by_tag[t]["pears"] or -9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--arms", default="beta4,beta4iourank")
    ap.add_argument("--exact-floor-frac", type=float, default=0.5,
                    help="loose exact-AP collapse guard for selection (default 0.5)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    data, nmp = load(eval_dir)
    arms = args.arms.split(",")

    # region metrics (paper credibility) | landscape metrics (downstream-critical)
    HEAD_METRICS = ("iou50", "iou70", "exap", "pears", "spear", "rauc", "rap")
    NMP_METRICS = ("iou50", "exap", "pears", "rauc", "rap")

    summary = {"arms": {}, "nmp_reference": {}}
    print(f"{'arm':16} fold sel_ep | IoU50 IoU70  ExAP | Pears Spear  rAUC   rAP")
    per_arm_test = {a: defaultdict(list) for a in arms}
    nmp_test = defaultdict(list)
    for arm in arms:
        folds = sorted({fk[1] for fk in data if fk[0] == arm})
        for k in folds:
            d = data[(arm, k)]
            ep = select_epoch(d.get("val", {}), args.exact_floor_frac)
            te = d.get("test", {}).get(ep) if ep else None
            if not te:
                print(f"{arm:16} {k}    {str(ep):6} | (missing test eval)")
                continue
            for m in HEAD_METRICS:
                if te.get(m) is not None:
                    per_arm_test[arm][m].append(te[m])
            nt = nmp[(arm, k)].get("test", {}).get(ep, {})
            for m in NMP_METRICS:
                if nt.get(m) is not None:
                    nmp_test[m].append(nt[m])
            g = lambda m: (te.get(m) if te.get(m) is not None else 0.0)
            print(f"{arm:16} {k}    {ep:6} | {g('iou50'):.3f} {g('iou70'):.3f} "
                  f"{g('exap'):.3f} | {g('pears'):.3f} {g('spear'):.3f} "
                  f"{g('rauc'):.3f} {g('rap'):.3f}")

    def ms(v):
        return (round(st.mean(v), 4), round(st.pstdev(v) if len(v) > 1 else 0.0, 4), len(v)) if v else (None, None, 0)

    print("\n=== 5-fold CV (mean±std, held-out test, goal-selected) — region | landscape ===")
    print(f"{'arm':18} {'IoU50':>13} {'IoU70':>13} {'ExAP':>13} | {'Pearson':>13} {'ResAUC':>13} {'ResAP':>13}")
    for arm in arms:
        t = per_arm_test[arm]
        row = {m: ms(t[m]) for m in HEAD_METRICS}
        summary["arms"][arm] = row
        def c(m): mu, sd, n = row[m]; return f"{mu:.3f}±{sd:.3f}" if mu is not None else "—"
        print(f"{arm:18} {c('iou50'):>13} {c('iou70'):>13} {c('exap'):>13} | "
              f"{c('pears'):>13} {c('rauc'):>13} {c('rap'):>13}")
    summary["nmp_reference"] = {m: ms(nmp_test[m]) for m in NMP_METRICS}
    nm = summary["nmp_reference"]
    def nmcell(m):
        mu = nm[m][0]
        return f"{mu:.3f}" if mu is not None else "—"
    print(f"{'NMP (ref)':18} {nmcell('iou50'):>13} {'':>13} {nmcell('exap'):>13} | "
          f"{nmcell('pears'):>13} {nmcell('rauc'):>13} {nmcell('rap'):>13}")

    # A-vs-B delta on region + landscape axes (only when >=2 arms given)
    if len(arms) >= 2 and all(a in summary["arms"] for a in arms[:2]):
        A, B = summary["arms"][arms[0]], summary["arms"][arms[1]]
        d = {m: round(B[m][0] - A[m][0], 4) for m in HEAD_METRICS
             if A[m][0] is not None and B[m][0] is not None}
        summary["delta_B_minus_A"] = d
        print(f"\nΔ({arms[1]}−{arms[0]}): " + " ".join(f"{m} {v:+.3f}" for m, v in d.items()))

    out = args.output or str(eval_dir / "cv_summary.json")
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"\n[cv] wrote {out}")


if __name__ == "__main__":
    main()
