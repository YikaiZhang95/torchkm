#!/usr/bin/env python
"""The DWD objective of the DWD package's final fits in an earlier Q2 run.

Runs before q2_kqr_dwd.py recorded it: each repeat's final KernGDWD fit is
refitted exactly as run_dwd_pkg did (the selected lambda, q = 1, the repeat's
sigest bandwidth, np.random.seed(seed) before fit), checked against the
recorded test accuracy, and its objective, mean V(y f) + lambda a'Ka, is
written to --out (one record per dataset and repeat).

Run:
  python benchmarks/q2_dwd_pkg_objective.py --data-dir ~/libsvm_data \\
      --results revision_results/q2.json --out revision_results/q2_dwd_pkg_objective.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import classification_metrics, load_dataset  # noqa: E402
from q2_kqr_dwd import dwd_objectives  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from dwd.gen_kern_dwd import KernGDWD
    from sklearn.metrics.pairwise import rbf_kernel

    doc = json.load(open(args.results))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out, data = [], {}
    for r in doc["records"]:
        if r["method"] != "dwd_pkg" or r["status"] not in ("ok", "capped"):
            continue
        ds = r["dataset"]
        if ds not in data:  # DWD sets: one split for every repeat (as q2_kqr_dwd.load)
            data[ds] = load_dataset(ds, args.data_dir, seed=doc["args"]["seed"])
        d, sig, lam = data[ds], float(r["bandwidth_sigest"]), float(r["selected"])
        K = rbf_kernel(d["Xtr"], gamma=2.0 * sig)
        final = KernGDWD(lambd=lam, q=1.0, kernel="precomputed")
        np.random.seed(r["seed"])
        final.fit(K, d["ytr"])
        del K
        scores = np.concatenate([
            final.decision_function(rbf_kernel(d["Xtr"], d["Xte"][i:i + 4096], gamma=2.0 * sig))
            for i in range(0, len(d["Xte"]), 4096)
        ])
        acc = classification_metrics(d["yte"], scores)["accuracy"]
        obj = dwd_objectives(d["Xtr"], d["ytr"], sig,
                             np.asarray(final.dual_coef_).reshape(-1, 1),
                             np.asarray(final.intercept_).reshape(-1), [lam], dev)[0]
        rec = dict(dataset=ds, repeat=r["repeat"], seed=r["seed"], selected=lam,
                   objective=obj, accuracy_refit=acc, accuracy_recorded=r["accuracy"])
        print(f"{ds} repeat {r['repeat']}: lambda {lam:.3g}, objective {obj:.6f}, "
              f"test accuracy {acc:.4f} (recorded {r['accuracy']:.4f})", flush=True)
        out.append(rec)
        with open(args.out, "w") as fh:
            json.dump(dict(source=args.results, records=out), fh, indent=1)


if __name__ == "__main__":
    main()
