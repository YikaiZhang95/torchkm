#!/usr/bin/env bash
# The full benchmark campaign for the JMLR revision, in dependency order, with
# the exact settings from EXPERIMENT_DESIGN.md. Run on the GPU workstation:
#
#   DATA=~/libsvm THUNDERSVM=~/thundersvm/python bash benchmarks/run_campaign.sh
#
# Environment variables:
#   DATA        directory of LIBSVM files (required)
#   THUNDERSVM  path to thundersvm/python (optional; the column is skipped otherwise)
#   DEVICE      cuda (default) or cpu
#   OUT         results directory (default benchmarks/results/<UTC timestamp>)
#   KKT         TorchKM stopping rule for every run (default "--kkt-eps 1e-6";
#               use "--kkt-scaled" for the scale-aware rule)
#   ONLY        space-separated experiment ids to run (default: all), e.g. "E1 E5"
#
# Every step writes JSON under $OUT and a log next to it; a failed step does
# not stop the campaign (its log says why). Re-running with the same OUT skips
# steps whose JSON already exists.

set -uo pipefail

DATA="${DATA:?set DATA to the LIBSVM directory}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)}"
KKT="${KKT:---kkt-eps 1e-6}"
# Converged-solution setting for the solver-quality comparison (E1e, E10d):
# tight inner step tolerance as well as the tight KKT rule.
TIGHT="${TIGHT:---kkt-eps 1e-6 --tol 1e-8}"
ONLY="${ONLY:-}"
THUNDER=""
if [[ -n "${THUNDERSVM:-}" ]]; then THUNDER="--thundersvm-path ${THUNDERSVM}"; fi
B=benchmarks
mkdir -p "$OUT"

run() {  # run <id> <out.json> <command...>
    local id="$1" json="$2"; shift 2
    if [[ -n "$ONLY" && " $ONLY " != *" $id "* ]]; then return; fi
    if [[ -s "$OUT/$json" ]]; then echo "[$id] $json exists, skipping"; return; fi
    echo "[$id] $(date -u +%H:%M:%S) $*"
    "$@" --device "$DEVICE" --out "$OUT/$json" > "$OUT/${json%.json}.log" 2>&1 \
        || echo "[$id] FAILED, see $OUT/${json%.json}.log"
}

{   # runner snapshot for the archive README
    echo "# Campaign $(basename "$OUT")"; echo
    echo "commit: $(git rev-parse HEAD)"; echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "device: $DEVICE"; echo "kkt: $KKT"; echo
    echo '```'; nvidia-smi 2>/dev/null || echo "no nvidia-smi"; echo '```'
    echo '```'; python -m pip freeze; echo '```'
} > "$OUT/README.md"

# ---------------------------------------------------------------------------
# E8 memory envelope (Q4). Paper protocol (50 C values, 10 folds); stops at OOM.
run E8 envelope.json python $B/bench_memory_envelope.py $KKT --repeats 3 \
    --sizes 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 \
    --nystrom-sizes 10000 50000 100000 250000 500000 1000000 --p 100
run E8b envelope_all_estimators.json python $B/bench_memory_envelope.py $KKT \
    --estimators svm dwd logit kqr --sizes 5000 10000 20000 --skip-nystrom \
    --grid-size 10 --folds 5

# ---------------------------------------------------------------------------
# E1 exact-range suite (Q1, Q2b, Q2e, item 5): kernel-natural problems, n <= 30k.
run E1a exact_torchkm.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite exact --libraries torchkm linear $KKT --repeats 10
run E1b exact_gpu_smo.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite exact --libraries cuml_svc thundersvm --repeats 3 --time-cap 7200 \
    --float32-baselines $THUNDER
run E1c exact_cuml_grid10.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite exact --libraries cuml_svc --repeats 3 --grid-size 10 --float32-baselines
run E1d exact_sklearn.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite exact --libraries sklearn_svc --repeats 1 --time-cap 7200
# E1e: TorchKM at the converged-solution setting; same folds and seeds as E1a,
# so the table can show accuracy (expected identical) and the time cost.
run E1e exact_torchkm_tight.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite exact --libraries torchkm $TIGHT --repeats 10

# ---------------------------------------------------------------------------
# E2 Adult scaling study (Q2a, Q4): a1a..a9a, exact while it fits, Nystrom always.
run E2a scaling_torchkm.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite scaling --libraries torchkm torchkm_nystrom linear $KKT --repeats 10
run E2b scaling_baselines.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite scaling --libraries sklearn_svc cuml_svc thundersvm --repeats 3 \
    --time-cap 7200 --float32-baselines $THUNDER

# ---------------------------------------------------------------------------
# E3 imbalanced sets (Q2c): w8a, ijcnn1 full; AUC is the headline.
run E3a imbalanced_torchkm.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite imbalanced --libraries torchkm_nystrom linear $KKT --repeats 10 \
    --landmarks 2000 5000
run E3b imbalanced_falkon.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite imbalanced --libraries falkon --repeats 3 --falkon-centers 2000 10000 \
    --time-cap 7200

# ---------------------------------------------------------------------------
# E4 scale tests (Q2d): covtype 581k, MNIST8m 4v6 1.27M, Nystrom path.
run E4a scale_torchkm.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite scale --libraries torchkm_nystrom $KKT --repeats 5 --landmarks 2000 5000
run E4b scale_falkon.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite scale --libraries falkon --repeats 3 --falkon-centers 2000 10000 20000 \
    --time-cap 14400
run E4c scale_linear.json python $B/bench_gpu_libraries.py --data-dir "$DATA" \
    --suite scale --libraries linear --repeats 3 --time-cap 7200

# ---------------------------------------------------------------------------
# E5 covtype budget curve (Q2f, item 3).
run E5 covtype_rank.json python $B/bench_covtype_rank.py --data-dir "$DATA" $KKT \
    --landmarks 2000 5000 10000 --ranks 30 300 1000 full --with-falkon --repeats 3

# ---------------------------------------------------------------------------
# E6 kernel quantile regression (Q3): exports splits for the R side.
run E6 kqr.json python $B/bench_kqr.py --data-dir "$DATA" $KKT \
    --datasets synthetic cadata abalone cpusmall space_ga --taus 0.1 0.5 0.9 \
    --repeats 5 --synthetic-n 10000 --export-splits "$OUT/kqr_splits"
run E6b kqr_large.json python $B/bench_kqr.py --data-dir "$DATA" $KKT \
    --datasets YearPredictionMSD --methods torchkm_kqr_nystrom linear_qr \
    --taus 0.1 0.5 0.9 --repeats 3 --time-cap 7200

# ---------------------------------------------------------------------------
# E7 kernel DWD (Q3): exports splits for kerndwd.
run E7 dwd.json python $B/bench_dwd.py --data-dir "$DATA" $KKT \
    --datasets gisette ijcnn1_30k mnist_3v8 --repeats 5 --export-splits "$OUT/dwd_splits"

# ---------------------------------------------------------------------------
# E9 Table 2 with accuracy, AUC and memory (Q5); E10 solver quality at fixed lambda.
run E9 table2.json python $B/table2_simulation.py --repeats 20 --matched-kernel \
    --max-iter 100000 $THUNDER
SQ="--sizes 10000,10 10000,100 10000,1000 20000,100 --lambdas 1e-1 1e-2 1e-3 1e-4 1e-5 --repeats 3"
run E10a solver_quality_default.json python $B/bench_solver_quality.py $SQ $THUNDER
run E10b solver_quality_kkt1e-6.json python $B/bench_solver_quality.py $SQ \
    --kkt-eps 1e-6 --solvers torchkm
run E10c solver_quality_scaled.json python $B/bench_solver_quality.py $SQ \
    --kkt-scaled --solvers torchkm
run E10d solver_quality_tol1e-8.json python $B/bench_solver_quality.py $SQ \
    --kkt-eps 1e-6 --tol 1e-8 --solvers torchkm

# ---------------------------------------------------------------------------
# R baselines (Q3). Skipped when Rscript is missing; check the package APIs first.
if command -v Rscript > /dev/null; then
    [[ -d "$OUT/kqr_splits" && ! -s "$OUT/kqr_r.csv" ]] && \
        Rscript $B/r/bench_kqr.R "$OUT/kqr_splits" "$OUT/kqr_r.csv" > "$OUT/kqr_r.log" 2>&1
    [[ -d "$OUT/dwd_splits" && ! -s "$OUT/dwd_r.csv" ]] && \
        Rscript $B/r/bench_dwd.R "$OUT/dwd_splits" "$OUT/dwd_r.csv" > "$OUT/dwd_r.log" 2>&1
    Rscript -e 'print(sessionInfo())' > "$OUT/sessionInfo.txt" 2>&1
else
    echo "Rscript not found: run benchmarks/r/bench_kqr.R and bench_dwd.R on a machine with R"
fi

# ---------------------------------------------------------------------------
# Tables and figures from whatever finished.
for f in exact_torchkm exact_gpu_smo exact_cuml_grid10 exact_sklearn exact_torchkm_tight \
         scaling_torchkm scaling_baselines imbalanced_torchkm imbalanced_falkon scale_torchkm \
         scale_falkon scale_linear covtype_rank kqr kqr_large dwd table2 solver_quality_default \
         solver_quality_kkt1e-6 solver_quality_scaled solver_quality_tol1e-8 envelope; do
    [[ -s "$OUT/$f.json" ]] && python $B/make_tables.py "$OUT/$f.json" > "$OUT/$f.md"
done
[[ -s "$OUT/kqr_r.csv" ]] && python $B/make_tables.py "$OUT/kqr.json" --r-csv "$OUT/kqr_r.csv" > "$OUT/kqr_with_r.md"
[[ -s "$OUT/dwd_r.csv" ]] && python $B/make_tables.py "$OUT/dwd.json" --r-csv "$OUT/dwd_r.csv" > "$OUT/dwd_with_r.md"
python $B/make_figures.py --results "$OUT" --out "$OUT/figures" || true
echo "campaign finished: $OUT"
