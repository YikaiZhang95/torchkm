#!/usr/bin/env python
"""Final Q1 tables: the TorchKM rows of a run of the submitted code merged with
the baseline rows (cuML, Falkon, KeOps, EigenPro) of the original run.

The two runs must share the protocol (folds, grid, seed, precision); the
merge checks it, and checks that every TorchKM cell used the same RBF
bandwidth (sigest, seeded per repeat) as the baselines of its dataset and
repeat. TorchKM rows (`torchkm`, `torchkm_trunc`) come only from the new run.

Writes <out>.json (the merged records) and <out>.md: three compact tables
(test accuracy, time, GPU memory), then q1_full_kernel's detailed table and
TorchKM time profile.

Run:
  python benchmarks/q1_final_tables.py --baselines revision_results/q1_real_f32.json \\
      --torchkm revision_results/q1_real_torchkm_final.json --out revision_results/q1_real_final
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_full_kernel import write_markdown  # noqa: E402

METHODS = ["torchkm", "torchkm_trunc", "cuml", "falkon", "keops", "eigenpro"]
LABELS = {
    "torchkm": "TorchKM",
    "torchkm_trunc": "TorchKM truncated",
    "cuml": "cuML SVC",
    "falkon": "Falkon",
    "keops": "KeOps",
    "eigenpro": "EigenPro",
}
PROTOCOL = ("folds", "grid_size", "lam_max", "lam_min", "seed", "dtype", "repeats")


def mean_se(vals):
    v = np.asarray(vals, dtype=float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), se


def compact(doc, datasets):
    """Accuracy, time and memory tables, methods as columns."""
    cells = {}
    for r in doc["records"]:
        if r["status"] in ("ok", "capped"):
            cells.setdefault((r["dataset"], r["method"]), []).append(r)
    methods = [m for m in METHODS if any((d, m) in cells for d in datasets)]
    head = "| dataset | n_train | " + " | ".join(LABELS[m] for m in methods) + " |"
    rule = "|---|---:|" + "---:|" * len(methods)

    def table(fmt):
        lines = [head, rule]
        for d in datasets:
            n = next((rs[0]["n_train"] for (dd, _), rs in cells.items() if dd == d), None)
            row = [d, "-" if n is None else f"{n:,}"]
            for m in methods:
                rs = cells.get((d, m))
                row.append("-" if not rs else fmt(rs))
            lines.append("| " + " | ".join(row) + " |")
        return lines

    def mark(rs):
        capped = any(r["status"] == "capped" for r in rs)
        uncert = any((r.get("converged_frac") is not None and r["converged_frac"] < 1.0)
                     for r in rs)
        return ("*" if capped else "") + ("‡" if uncert else "")

    def acc(rs):
        m, se = mean_se([r["accuracy"] for r in rs])
        return f"{m:.4f} ± {se:.4f}{mark(rs)}"

    def time_s(rs):
        m, se = mean_se([r["time_s"] for r in rs])
        return f"{m:,.1f}{mark(rs)}"

    def mem(rs):
        v = [r.get("gpu_bytes") for r in rs if r.get("gpu_bytes")]
        return "-" if not v else f"{np.mean(v) / 1e9:.2f}"

    out = ["### Test accuracy (mean ± SE over repeats)", ""] + table(acc)
    out += ["", "### Time (s): CV sweep + final fit + test predictions", ""] + table(time_s)
    out += ["", "### GPU memory (GB, NVML peak of the process)", ""] + table(mem)
    out += [
        "",
        "\\* capped at the 2-hour limit per method x dataset x repeat (selects among "
        "the values completed). ‡ some lambda or fold fits ended above the certified "
        "gap. Methods: TorchKM = TorchKMSVC (full spectrum, cvksvm); TorchKM truncated "
        "= TorchKMSVC(spectrum=\"truncated\", spectrum_block=10); cuML SVC, Falkon "
        "(M = n), KeOps (kernel ridge, CG), EigenPro 2.",
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--torchkm", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()

    base = json.load(open(args.baselines))
    new = json.load(open(args.torchkm))
    for k in PROTOCOL:
        if base["args"].get(k) != new["args"].get(k):
            sys.exit(f"protocol differs in {k}: {base['args'].get(k)} vs {new['args'].get(k)}")
    if not np.allclose(base["grid_lambda"], new["grid_lambda"], rtol=1e-12):
        sys.exit("lambda grids differ")
    bw = {(r["dataset"], r["repeat"]): r.get("bandwidth_sigest") for r in base["records"]}
    for r in new["records"]:
        b = bw.get((r["dataset"], r["repeat"]))
        if b is not None and not np.isclose(b, r.get("bandwidth_sigest"), rtol=1e-9):
            sys.exit(f"bandwidth differs for {r['dataset']} repeat {r['repeat']}")

    datasets = list(new["args"]["datasets"])
    keep = [r for r in base["records"] if r["method"] not in ("torchkm", "torchkm_trunc")]
    records = keep + [r for r in new["records"] if r["dataset"] in datasets]
    order = {m: i for i, m in enumerate(METHODS)}
    records.sort(key=lambda r: (datasets.index(r["dataset"]) if r["dataset"] in datasets
                                else len(datasets), order.get(r["method"], 99), r["repeat"]))
    doc = dict(new, records=records)
    benv, nenv = base["environment"], new["environment"]
    doc["provenance"] = dict(
        torchkm_rows=dict(file=args.torchkm, torchkm=nenv.get("torchkm"),
                          commit=nenv.get("torchkm_commit"), date=nenv.get("timestamp_utc")),
        baseline_rows=dict(file=args.baselines, commit=benv.get("torchkm_commit"),
                           date=benv.get("timestamp_utc"), packages=benv.get("packages")),
    )
    with open(args.out + ".json", "w") as fh:
        json.dump(doc, fh, indent=1, default=float)

    detail_path = args.out + "_detail.md"
    write_markdown(doc, detail_path)
    detail = open(detail_path).read()
    os.remove(detail_path)
    gpu = (nenv.get("gpu") or {}).get("name")
    pk = benv.get("packages") or {}
    lines = [
        f"# Q1 final: {os.path.basename(args.out)}",
        "",
        f"{gpu}; torch {nenv.get('torch')}; {new['args']['dtype']} everywhere; "
        f"{new['args']['folds']}-fold CV on shared stratified folds; "
        f"{new['args']['grid_size']} lambda values in [{new['args']['lam_min']:g}, "
        f"{new['args']['lam_max']:g}]; {new['args']['repeats']} repeats (seed "
        f"{new['args']['seed']} + repeat); one sigest bandwidth per repeat, shared by "
        "every method.",
        "",
        f"- TorchKM rows: torchkm {nenv.get('torchkm')}, commit "
        f"{str(nenv.get('torchkm_commit'))[:7]}, run {str(nenv.get('timestamp_utc'))[:10]}.",
        f"- Baseline rows: the original run of {str(benv.get('timestamp_utc'))[:10]} "
        f"(cuML {pk.get('cuml')}, Falkon {pk.get('falkon')}); same protocol, bandwidths "
        "and folds (checked).",
        "",
    ]
    lines += compact(doc, datasets)
    lines += ["", "## Details", ""] + detail.split("\n")[2:]
    open(args.out + ".md", "w").write("\n".join(lines))
    print(f"wrote {args.out}.md and {args.out}.json ({len(records)} records)")


if __name__ == "__main__":
    main()
