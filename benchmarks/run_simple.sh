#!/usr/bin/env bash
# The simplest way to run the GPU experiments: one after another, each
# writing a JSON and a Markdown table into RESULTS. A step whose JSON already
# exists is skipped, so re-running resumes.
#
#   bash benchmarks/run_simple.sh /some/writable/dir /home/yzhang705/libsvm_data
#
# Steps (settings from EXPERIMENT_DESIGN.md): memory envelope, solver quality
# at fixed lambda, exact suite (TorchKM + linear baselines, default and
# converged tolerances), covtype budget curve, imbalanced suite, scale suite,
# Table 2. Only files you already have are needed. DEVICE=cpu for a test.

set -uo pipefail
RESULTS="${1:?usage: run_simple.sh RESULTS_DIR DATA_DIR}"
DATA="${2:?usage: run_simple.sh RESULTS_DIR DATA_DIR}"
DEVICE="${DEVICE:-cuda}"
EXTRA="${EXTRA:-}"          # e.g. EXTRA=--smoke for a quick check
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

step envelope       bench_memory_envelope.py --kkt-eps 1e-6 --repeats 3
step solver_quality bench_solver_quality.py --sizes 10000,10 10000,100 10000,1000 20000,100 \
                    --lambdas 1e-1 1e-2 1e-3 1e-4 1e-5 --repeats 3
step solver_quality_tight bench_solver_quality.py --sizes 10000,10 10000,100 10000,1000 20000,100 \
                    --lambdas 1e-1 1e-2 1e-3 1e-4 1e-5 --repeats 3 --kkt-eps 1e-6 --tol 1e-8 --solvers torchkm
step exact          bench_gpu_libraries.py --data-dir "$DATA" --suite exact --libraries torchkm linear \
                    --kkt-eps 1e-6 --repeats 10
step exact_tight    bench_gpu_libraries.py --data-dir "$DATA" --suite exact --libraries torchkm \
                    --kkt-eps 1e-6 --tol 1e-8 --repeats 10
step covtype_rank   bench_covtype_rank.py --data-dir "$DATA" --kkt-eps 1e-6 --repeats 3 \
                    --landmarks 2000 5000 10000 --ranks 30 300 1000 full
step imbalanced     bench_gpu_libraries.py --data-dir "$DATA" --suite imbalanced --libraries torchkm_nystrom linear \
                    --kkt-eps 1e-6 --repeats 10 --landmarks 2000 5000
step scale          bench_gpu_libraries.py --data-dir "$DATA" --suite scale --libraries torchkm_nystrom \
                    --kkt-eps 1e-6 --repeats 5 --landmarks 2000 5000
step table2         table2_simulation.py --repeats 20 --matched-kernel --skip-sklearn

python "$B/make_figures.py" --results "$RESULTS" --out "$RESULTS/figures" 2>/dev/null || true
echo "finished. Markdown tables: $RESULTS/*.md"
