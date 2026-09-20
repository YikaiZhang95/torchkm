"""Render benchmark JSON results (and R CSV rows) as Markdown or LaTeX tables.

Reads the ``records`` of one or more ``bench_*.py`` output files, groups them
by dataset and method (library plus its budget parameters: landmarks, rank,
centres, tau), and prints mean and standard error over repeats of the metrics
present in the records: accuracy, balanced accuracy, AUC, pinball loss,
coverage, end-to-end time, and peak memory (PyTorch allocator peak when
available, otherwise the NVML process peak).

R baselines write CSV rows (``dataset,library,repeat,tau,time_s,accuracy,auc,
pinball_loss,coverage,...``) that ``--r-csv`` merges as records.

Examples
--------
::

    python benchmarks/make_tables.py benchmarks/results/exact.json
    python benchmarks/make_tables.py benchmarks/results/kqr.json --r-csv benchmarks/results/kqr_r.csv --latex
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np

from _common import mean_se

METRICS = [
    ("accuracy", "Acc", "{:.4f}"),
    ("balanced_accuracy", "Bal. acc", "{:.4f}"),
    ("auc", "AUC", "{:.4f}"),
    ("pinball_loss", "Pinball", "{:.4f}"),
    ("coverage", "Coverage", "{:.3f}"),
    ("rmse_to_true_quantile", "RMSE(q)", "{:.4f}"),
    ("objective", "Objective", "{:.5f}"),
    ("relative_gap", "Rel. gap", "{:.1e}"),
    ("time_s", "Time (s)", "{:.1f}"),
    ("peak_gb", "Peak mem (GB)", "{:.2f}"),
]


def method_key(rec: Dict[str, Any]) -> str:
    name = rec.get("library") or rec.get("solver") or rec.get("estimator") or "?"
    params = rec.get("params") or {}
    bits = []
    if "num_landmarks" in params:
        bits.append(f"m={params['num_landmarks']}")
    if "nys_k" in params:
        bits.append(f"k={params['nys_k']}")
    if "centers" in params:
        bits.append(f"M={params['centers']}")
    if rec.get("tau") is not None:
        bits.append(f"tau={rec['tau']}")
    if rec.get("lam") is not None:
        bits.append(f"lambda={rec['lam']:g}")
    if rec.get("mode") == "exact" and name.startswith("torchkm") and not bits:
        bits.append("exact")
    return name + (" (" + ", ".join(bits) + ")" if bits else "")


def dataset_key(rec: Dict[str, Any]) -> str:
    if rec.get("dataset"):
        return str(rec["dataset"])
    if rec.get("n") is not None and rec.get("p") is not None:
        return f"n={rec['n']}, p={rec['p']}"
    return "?"


def peak_gb(rec: Dict[str, Any]):
    mem = rec.get("memory") or {}
    v = (
        rec.get("torch_peak_bytes")
        or mem.get("nvml_process_peak")
        or mem.get("torch_max_allocated")
    )
    return None if v is None else float(v) / 1e9


def load_records(paths: List[str], r_csvs: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path) as fh:
            doc = json.load(fh)
        records.extend(doc.get("records", []))
    for path in r_csvs:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                rec: Dict[str, Any] = {}
                for k, v in row.items():
                    if v in ("", "NA", None):
                        continue
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v
                rec.setdefault("status", "ok")
                records.append(rec)
    return records


def build_table(records: List[Dict[str, Any]]):
    groups: "OrderedDict[tuple, List[Dict[str, Any]]]" = OrderedDict()
    for rec in records:
        if rec.get("status") not in ("ok", "capped"):
            continue
        rec = dict(rec)
        rec["peak_gb"] = peak_gb(rec)
        groups.setdefault((dataset_key(rec), method_key(rec)), []).append(rec)
    present = [
        m
        for m in METRICS
        if any(r.get(m[0]) is not None for rs in groups.values() for r in rs)
    ]
    rows = []
    for (ds, method), recs in groups.items():
        row = {"dataset": ds, "method": method, "n": len(recs)}
        capped = sum(1 for r in recs if r.get("status") == "capped")
        if capped:
            row["method"] += f" [capped {capped}/{len(recs)}]"
        for key, _, fmt in present:
            m, se = mean_se([r.get(key) for r in recs])
            row[key] = (m, se, fmt)
        rows.append(row)
    return present, rows


def cell(value, latex: bool) -> str:
    m, se, fmt = value
    if not np.isfinite(m):
        return "-"
    s = fmt.format(m)
    if np.isfinite(se) and se > 0:
        s += f" ({fmt.format(se)})" if not latex else f" \\scriptsize({fmt.format(se)})"
    return s


def render_markdown(present, rows) -> str:
    head = ["Dataset", "Method", "Runs"] + [label for _, label, _ in present]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in rows:
        cells = [row["dataset"], row["method"], str(row["n"])]
        cells += [cell(row[key], False) for key, _, _ in present]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render_latex(present, rows) -> str:
    cols = "ll" + "r" * (len(present) + 1)
    head = ["Dataset", "Method", "Runs"] + [label for _, label, _ in present]
    out = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(head) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [row["dataset"], row["method"].replace("_", "\\_"), str(row["n"])]
        cells += [cell(row[key], True) for key, _, _ in present]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("json", nargs="*", help="bench_*.py result files")
    ap.add_argument(
        "--r-csv", nargs="*", default=[], help="CSV rows written by the R scripts"
    )
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    if not args.json and not args.r_csv:
        ap.error("pass at least one JSON or --r-csv file")
    records = load_records(args.json, args.r_csv)
    present, rows = build_table(records)
    if not rows:
        print("no successful records")
        return
    print(render_latex(present, rows) if args.latex else render_markdown(present, rows))


if __name__ == "__main__":
    sys.exit(main())
