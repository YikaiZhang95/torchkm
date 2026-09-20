"""Paper figures from the campaign's JSON results.

Reads the files ``run_campaign.sh`` writes into a results directory and draws:

* ``fig1_envelope``: time and peak memory versus n, exact mode up to the first
  out-of-memory error and the Nyström path beyond it, with the ceilings of
  16, 24, 48 and 80 GB cards marked (``envelope.json``).
* ``fig2_scaling``: end-to-end time versus n on the Adult scaling study, one
  line per library (``scaling_torchkm.json`` + ``scaling_baselines.json``).
* ``fig3_covtype_budget``: accuracy versus number of landmarks on covtype,
  one line per Nyström rank, Falkon at the same centre counts, and the
  exact-mode 30k reference as a horizontal line (``covtype_rank.json`` +
  ``exact_torchkm.json``).
* ``fig4_solver_quality``: relative objective gap to the best solver versus
  lambda for the default, tight and scaled stopping rules
  (``solver_quality_*.json``).

Each figure is written as PDF and PNG. Missing inputs are reported and
skipped. Colours are a fixed, colourblind-safe assignment per library, so a
library keeps its hue across every figure; every series is also labelled
directly or in a legend, so identity never rests on colour alone.

Example::

    python benchmarks/make_figures.py --results benchmarks/results/<run> --out figures/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

# Fixed categorical assignment (validated palette; one hue per library, never cycled).
COLORS = {
    "torchkm": "#2a78d6",  # blue
    "torchkm_nystrom": "#1baf7a",  # aqua
    "cuml_svc": "#eb6834",  # orange
    "thundersvm": "#eda100",  # yellow
    "falkon": "#4a3aa7",  # violet
    "sklearn_svc": "#e87ba4",  # magenta
    "logreg": "#52514e",  # neutral ink for the linear baselines
    "linearsvc": "#8a8985",
}
LABELS = {
    "torchkm": "TorchKM (exact)",
    "torchkm_nystrom": "TorchKM (Nyström)",
    "cuml_svc": "cuML SVC",
    "thundersvm": "ThunderSVM",
    "falkon": "Falkon",
    "sklearn_svc": "scikit-learn SVC",
    "logreg": "logistic regression",
    "linearsvc": "LinearSVC",
}
CARD_GB = [16, 24, 48, 80]


def load(results: str, name: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(results, name)
    if not os.path.exists(path):
        print(f"[skip] {name} not found")
        return None
    with open(path) as fh:
        return json.load(fh).get("records", [])


def ok(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in records if r.get("status") in ("ok", "capped")]


def mean_se(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    return float(a.mean()), (
        float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    )


def peak_bytes(rec: Dict[str, Any]) -> Optional[float]:
    """GPU peak when the run had a GPU, otherwise host RSS (CPU runs)."""
    mem = rec.get("memory") or {}
    v = (
        rec.get("torch_peak_bytes")
        or mem.get("nvml_process_peak")
        or mem.get("host_rss_peak")
    )
    return None if v is None else float(v)


def log_axes(ax, x: bool = True, y: bool = True) -> None:
    """Log scales only where the plotted data allow it (positive values)."""
    xs = np.concatenate(
        [np.asarray(line.get_xdata(), dtype=float) for line in ax.get_lines()]
        or [np.array([])]
    )
    ys = np.concatenate(
        [np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()]
        or [np.array([])]
    )
    if x and xs.size and np.nanmin(xs) > 0:
        ax.set_xscale("log")
    if y and ys.size and np.nanmax(ys) > 0:
        ax.set_yscale("log")


def style(plt):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def save(fig, out: str, name: str) -> None:
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    print(f"wrote {name}.pdf/.png")


def fig_envelope(results: str, out: str, plt) -> None:
    recs = load(results, "envelope.json")
    if not recs:
        return
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for mode, key, label in (
        ("exact", "torchkm", "exact"),
        ("nystrom", "torchkm_nystrom", "Nyström"),
    ):
        by_n = defaultdict(list)
        for r in ok(recs):
            if r.get("estimator", "svm") == "svm" and r.get("mode") == mode:
                by_n[r["n"]].append(r)
        ns = sorted(by_n)
        if not ns:
            continue
        t = [mean_se([r["time_s"] for r in by_n[n]])[0] for n in ns]
        m = [mean_se([peak_bytes(r) for r in by_n[n]])[0] for n in ns]
        ax_t.plot(ns, t, marker="o", color=COLORS[key], label=label)
        ax_m.plot(ns, np.asarray(m) / 1e9, marker="o", color=COLORS[key], label=label)
    oom = sorted(
        {r["n"] for r in recs if r.get("mode") == "exact" and r.get("status") == "oom"}
    )
    if oom:
        for ax in (ax_t, ax_m):
            ax.axvline(oom[0], color="#e34948", linewidth=1, linestyle="--")
        ax_m.annotate(
            f"first OOM\nn={oom[0]:,}",
            (oom[0], ax_m.get_ylim()[1] * 0.6),
            color="#e34948",
            fontsize=8,
        )
    for gb in CARD_GB:
        ax_m.axhline(gb, color="#c3c2b7", linewidth=0.8, linestyle=":")
        ax_m.annotate(
            f"{gb} GB",
            (ax_m.get_xlim()[0], gb),
            fontsize=7,
            color="#52514e",
            va="bottom",
        )
    ax_t.set(xlabel="training size n", ylabel="end-to-end time (s)")
    ax_m.set(xlabel="training size n", ylabel="peak memory (GB)")
    log_axes(ax_t)
    log_axes(ax_m)
    ax_t.legend(loc="upper left")
    fig.suptitle("TorchKM operating envelope: full train-and-tune pipeline", fontsize=9)
    save(fig, out, "fig1_envelope")


def fig_scaling(results: str, out: str, plt) -> None:
    recs = []
    for name in ("scaling_torchkm.json", "scaling_baselines.json"):
        r = load(results, name)
        if r:
            recs += r
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    by_lib = defaultdict(lambda: defaultdict(list))
    for r in ok(recs):
        by_lib[r["library"]][r["n_train"]].append(r["time_s"])
    for lib in LABELS:  # fixed order and colour per library
        if lib not in by_lib:
            continue
        ns = sorted(by_lib[lib])
        ts = [mean_se(by_lib[lib][n])[0] for n in ns]
        ax.plot(ns, ts, marker="o", color=COLORS[lib], label=LABELS[lib])
        ax.annotate(
            LABELS[lib],
            (ns[-1], ts[-1]),
            fontsize=7,
            color=COLORS[lib],
            xytext=(3, 0),
            textcoords="offset points",
        )
    ax.set(xlabel="training size n (Adult a1a to a9a)", ylabel="end-to-end time (s)")
    log_axes(ax)
    ax.legend(fontsize=7, loc="upper left")
    save(fig, out, "fig2_scaling")


def fig_covtype(results: str, out: str, plt) -> None:
    recs = load(results, "covtype_rank.json")
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    by_rank = defaultdict(lambda: defaultdict(list))
    falkon = defaultdict(list)
    for r in ok(recs):
        p = r.get("params") or {}
        if r["library"] == "torchkm_nystrom":
            by_rank[p.get("nys_k")][p.get("num_landmarks")].append(r["accuracy"])
        elif r["library"] == "falkon":
            falkon[p.get("centers")].append(r["accuracy"])
    shades = [
        "#9ec1ec",
        "#5f9be0",
        "#2a78d6",
        "#17498a",
    ]  # one hue, light -> dark by rank
    for shade, k in zip(shades, sorted(by_rank)):
        ms = sorted(by_rank[k])
        ax.plot(
            ms,
            [mean_se(by_rank[k][m])[0] for m in ms],
            marker="o",
            color=shade,
            label=f"TorchKM rank {k}",
        )
    if falkon:
        ms = sorted(falkon)
        ax.plot(
            ms,
            [mean_se(falkon[m])[0] for m in ms],
            marker="s",
            color=COLORS["falkon"],
            label="Falkon",
        )
    exact = load(results, "exact_torchkm.json") or []
    ref = [
        r["accuracy"]
        for r in ok(exact)
        if r.get("dataset") == "covtype_30k" and r["library"] == "torchkm"
    ]
    if ref:
        ax.axhline(np.mean(ref), color=COLORS["torchkm"], linestyle="--", linewidth=1)
        ax.annotate(
            "exact mode, 30k subsample",
            (ax.get_xlim()[0], np.mean(ref)),
            fontsize=7,
            color=COLORS["torchkm"],
            va="bottom",
        )
    ax.set(xscale="log", xlabel="landmarks / centres", ylabel="test accuracy (covtype)")
    ax.legend(fontsize=7)
    save(fig, out, "fig3_covtype_budget")


def fig_solver_quality(results: str, out: str, plt) -> None:
    runs = [
        ("solver_quality_default.json", "default KKTeps=1e-3", "#e34948"),
        ("solver_quality_kkt1e-6.json", "KKTeps=1e-6", "#2a78d6"),
        ("solver_quality_scaled.json", "scale-aware rule", "#1baf7a"),
        ("solver_quality_tol1e-8.json", "KKTeps=1e-6, tol=1e-8", "#4a3aa7"),
    ]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    drawn = False
    for name, label, color in runs:
        recs = load(results, name)
        if not recs:
            continue
        by_lam = defaultdict(list)
        for r in ok(recs):
            if r.get("solver") == "torchkm" and r.get("relative_gap") is not None:
                by_lam[r["lam"]].append(max(r["relative_gap"], 1e-9))
        lams = sorted(by_lam)
        if not lams:
            continue
        ax.plot(
            lams,
            [np.median(by_lam[l]) for l in lams],
            marker="o",
            color=color,
            label=label,
        )
        drawn = True
    if not drawn:
        return
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="lambda",
        ylabel="relative objective gap to best solver",
    )
    ax.legend(fontsize=7)
    save(fig, out, "fig4_solver_quality")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results", required=True, help="directory with the campaign JSON files"
    )
    ap.add_argument(
        "--out", default=None, help="output directory (default: <results>/figures)"
    )
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "figures")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    fig_envelope(args.results, out, plt)
    fig_scaling(args.results, out, plt)
    fig_covtype(args.results, out, plt)
    fig_solver_quality(args.results, out, plt)


if __name__ == "__main__":
    sys.exit(main())
