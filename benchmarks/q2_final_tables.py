#!/usr/bin/env python
"""Final kernel DWD table of Q2: the TorchKM rows of a run of the submitted
code merged with the rows of the DWD package (KernGDWD, CPU) from the
original run.

The two runs must share the protocol (folds, grid, seed, repeats); the merge
checks it, and that every TorchKM cell used the same RBF bandwidth (sigest,
seeded per repeat) as the baseline of its dataset and repeat. TorchKM rows
(`torchkm_dwd`, `torchkm_dwd_trunc`) come only from the new run; KQR records
are left out.

Writes <out>.json (the merged records) and <out>.md (a compact accuracy / time /
memory table, then q2_kqr_dwd's detailed DWD table).

Run:
  python benchmarks/q2_final_tables.py --baselines revision_results/q2.json \\
      --torchkm revision_results/q2_dwd_torchkm_final.json --out revision_results/q2_dwd_final
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q2_kqr_dwd import write_markdown  # noqa: E402

METHODS = ["torchkm_dwd", "torchkm_dwd_trunc", "dwd_pkg"]
LABELS = {
    "torchkm_dwd": "TorchKM DWD (GPU)",
    "torchkm_dwd_trunc": "TorchKM DWD truncated (GPU)",
    "dwd_pkg": "DWD package KernGDWD (CPU)",
}
PROTOCOL = ("folds", "grid_size", "lam_max", "lam_min", "seed", "repeats")


def mean_se(vals):
    v = np.asarray(vals, dtype=float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--torchkm", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()

    base, new = json.load(open(args.baselines)), json.load(open(args.torchkm))
    for k in PROTOCOL:
        if base["args"].get(k) != new["args"].get(k):
            sys.exit(f"protocol differs in {k}: {base['args'].get(k)} vs {new['args'].get(k)}")
    if not np.allclose(base["grid_lambda"], new["grid_lambda"], rtol=1e-12):
        sys.exit("lambda grids differ")
    bw = {(r["dataset"], r["repeat"]): r.get("bandwidth_sigest")
          for r in base["records"] if r["method"] == "dwd_pkg"}
    for r in new["records"]:
        b = bw.get((r["dataset"], r["repeat"]))
        if b is not None and not np.isclose(b, r.get("bandwidth_sigest"), rtol=1e-9):
            sys.exit(f"bandwidth differs for {r['dataset']} repeat {r['repeat']}")

    datasets = [d for d in new["args"]["datasets"]]
    records = [r for r in base["records"] if r["method"] == "dwd_pkg"]
    records += [r for r in new["records"] if r["method"] in ("torchkm_dwd", "torchkm_dwd_trunc")]
    order = {m: i for i, m in enumerate(METHODS)}
    records.sort(key=lambda r: (datasets.index(r["dataset"]), order[r["method"]], r["repeat"]))
    doc = dict(new, records=records)
    benv, nenv = base["environment"], new["environment"]
    doc["provenance"] = dict(
        torchkm_rows=dict(file=args.torchkm, commit=nenv.get("torchkm_commit"),
                          date=nenv.get("timestamp_utc")),
        baseline_rows=dict(file=args.baselines, date=benv.get("timestamp_utc")),
    )
    with open(args.out + ".json", "w") as fh:
        json.dump(doc, fh, indent=1, default=float)

    cells = {}
    for r in records:
        if r["status"] in ("ok", "capped"):
            cells.setdefault((r["dataset"], r["method"]), []).append(r)
    head = "| dataset | n_train | method | test accuracy | AUC | time (s) | memory | note |"
    lines = [
        f"# Q2 final: kernel DWD ({os.path.basename(args.out)})",
        "",
        f"{(nenv.get('gpu') or {}).get('name')}; torch {nenv.get('torch')}; "
        f"{new['args']['folds']}-fold CV on shared folds; {new['args']['grid_size']} lambda "
        f"values in [{new['args']['lam_min']:g}, {new['args']['lam_max']:g}]; "
        f"{new['args']['repeats']} repeats (seed {new['args']['seed']} + repeat); one sigest "
        "bandwidth per repeat, shared by every method; float64.",
        "",
        f"- TorchKM rows: torchkm {nenv.get('torchkm')}, commit "
        f"{str(nenv.get('torchkm_commit'))[:7]}, run {str(nenv.get('timestamp_utc'))[:10]}.",
        f"- DWD package rows: the original run of {str(benv.get('timestamp_utc'))[:10]}; same "
        "protocol, bandwidths and folds (checked).",
        "",
        head,
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for d in datasets:
        for m in METHODS:
            rs = cells.get((d, m))
            if not rs:
                continue
            acc, ase = mean_se([r["accuracy"] for r in rs])
            auc, _ = mean_se([r["auc"] for r in rs])
            t, _ = mean_se([r["time_s"] for r in rs])
            mem = np.mean([r["memory"]["bytes"] for r in rs if r.get("memory")]) / 1e9
            where = rs[0]["memory"]["where"] if rs[0].get("memory") else ""
            note = []
            capped = [r for r in rs if r["status"] == "capped"]
            if capped:
                note.append(f"{len(capped)} of {len(rs)} capped at the 2-hour limit")
            cf = [r.get("converged_frac") for r in rs if r.get("converged_frac") is not None]
            if m == "torchkm_dwd_trunc" and cf:
                note.append("all lambdas certified" if min(cf) == 1.0
                            else f"certified {min(cf):.0%}-{max(cf):.0%} of lambdas")
            lines.append(
                f"| {d} | {rs[0]['n_train']:,} | {LABELS[m]} | {acc:.4f} ± {ase:.4f} | "
                f"{auc:.4f} | {t:,.1f} | {mem:.2f} GB {where} | {'; '.join(note)} |"
            )
    detail_path = args.out + "_detail.md"
    write_markdown(doc, detail_path)
    detail = open(detail_path).read()
    os.remove(detail_path)
    dwd = detail[detail.index("## Kernel DWD"):] if "## Kernel DWD" in detail else detail
    lines += ["", "## Details", "", dwd]
    open(args.out + ".md", "w").write("\n".join(lines))
    print(f"wrote {args.out}.md and {args.out}.json ({len(records)} records)")


if __name__ == "__main__":
    main()
