#!/usr/bin/env bash
# The simplest way to run the GPU experiments, at sizes like the paper's
# (exact mode capped at 20k training rows, the 10k synthetic cells, a few
# repeats). Each step writes a JSON and a Markdown table into RESULTS; a step
# whose JSON exists is skipped, so re-running resumes. The full protocol with
# every baseline is benchmarks/run_campaign.sh.
#
#   bash benchmarks/run_simple.sh /some/writable/dir /home/yzhang705/libsvm_data
#
# Variables: DEVICE (cuda), REPEATS (3), MAX_TRAIN (20000), EXTRA (e.g. --smoke).

set -uo pipefail
RESULTS="${1:?usage: run_simple.sh RESULTS_DIR DATA_DIR}"
DATA="${2:?usage: run_simple.sh RESULTS_DIR DATA_DIR}"
DEVICE="${DEVICE:-cuda}"
REPEATS="${REPEATS:-3}"
MAX_TRAIN="${MAX_TRAIN:-20000}"
EXTRA="${EXTRA:-}"
B=benchmarks
mkdir -p "$RESULTS"

step() {  # step <name> <script> <args...>
    local name="$1" script="$2"; shift 2
    if [[ -s "$RESULTS/$name.json" ]]; then echo "== $name: done already"; return; fi
    echo "== $name  $(date +%H:%M)"
    python "$B/$script" "$@" $EXTRA --device "$DEVICE" --out "$RESULTS/$name.json" \
        > "$RESULTS/$name.log" 2>&1 || echo "   FAILED (see $RESULTS/$name.log)"
    [[ -s "$RESULTS/$name.json" ]] && python "$B/make_tables.py" "$RESULTS/$name.json" > "$RESULTS/$name.md"
}

# memory envelope: exact mode 5k..25k (stops at the first OOM), Nystrom to 250k
step envelope       bench_memory_envelope.py --kkt-eps 1e-6 --repeats 1 \
                    --sizes 5000 10000 15000 20000 25000 --nystrom-sizes 50000 100000 250000

# solver quality at fixed lambda, the paper's 10k cells: default vs converged tolerances
step solver_quality bench_solver_quality.py --sizes 10000,10 10000,100 \
                    --lambdas 1e-1 1e-2 1e-3 1e-4 --repeats 2
step solver_quality_tight bench_solver_quality.py --sizes 10000,10 10000,100 \
                    --lambdas 1e-1 1e-2 1e-3 1e-4 --repeats 2 --kkt-eps 1e-6 --tol 1e-8 --solvers torchkm

# exact suite (ijcnn1, MNIST 3v8 / 4v9, covtype, w7a), training sets capped at MAX_TRAIN
step exact          bench_gpu_libraries.py --data-dir "$DATA" --suite exact --libraries torchkm linear \
                    --kkt-eps 1e-6 --repeats "$REPEATS" --max-train "$MAX_TRAIN"
step exact_tight    bench_gpu_libraries.py --data-dir "$DATA" --suite exact --libraries torchkm \
                    --kkt-eps 1e-6 --tol 1e-8 --repeats "$REPEATS" --max-train "$MAX_TRAIN"

# Adult and Web at the paper's sizes (a7a, a8a, a9a, w7a, w8a): exact while it fits, Nystrom always
step paper_sets     bench_gpu_libraries.py --data-dir "$DATA" --datasets a7a a8a a9a w7a w8a \
                    --libraries torchkm torchkm_nystrom linear --kkt-eps 1e-6 --repeats "$REPEATS"

# covtype budget curve on the Nystrom path
step covtype_rank   bench_covtype_rank.py --data-dir "$DATA" --kkt-eps 1e-6 --repeats 2 \
                    --landmarks 2000 5000 --ranks 30 300 1000

# large sets on the Nystrom path (ijcnn1 full, covtype, MNIST8m 4v6)
step nystrom_large  bench_gpu_libraries.py --data-dir "$DATA" --datasets ijcnn1 covtype mnist8m_4v6 \
                    --libraries torchkm_nystrom --kkt-eps 1e-6 --repeats 2 --landmarks 2000

# Table 2's 10k cells with accuracy, AUC and peak memory
step table2         table2_simulation.py --repeats 5 --matched-kernel --skip-sklearn \
                    --sizes 10000,10 10000,100 10000,1000

python "$B/make_figures.py" --results "$RESULTS" --out "$RESULTS/figures" 2>/dev/null || true
echo "finished. Markdown tables: $RESULTS/*.md"
