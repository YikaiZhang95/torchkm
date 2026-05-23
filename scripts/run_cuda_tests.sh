#!/usr/bin/env bash
# Run the TorchKM test suite under CUDA on the local machine,
# collect logs into benchmarks/cuda-runs/<UTC-timestamp>/, and
# optionally commit + push the bundle so it becomes part of the
# repository's GPU validation record.
#
# Usage:
#   scripts/run_cuda_tests.sh              # collect bundle, don't commit
#   scripts/run_cuda_tests.sh --commit     # also git commit the bundle
#   scripts/run_cuda_tests.sh --push       # commit and push
#
# Prerequisites:
#   - NVIDIA driver + GPU visible to `nvidia-smi`
#   - Python virtualenv with `pip install -e ".[dev,examples,viz]"`
#   - Run from a clean checkout (the script reads git rev-parse HEAD)

set -e
set -o pipefail

ACTION="${1:-collect}"   # collect | --commit | --push

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ---- Preflight ----------------------------------------------------------
echo "==> Preflight checks"
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?" >&2
    exit 1
fi
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch is not installed in this environment." >&2
    echo "Hint:  python -m pip install -e \".[dev,examples,viz]\"" >&2
    exit 1
fi
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: PyTorch reports CUDA is not available." >&2
    echo "Check that the wheel was built with CUDA support:" >&2
    python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda)" >&2
    exit 1
fi

# ---- Run setup ----------------------------------------------------------
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="benchmarks/cuda-runs/${TIMESTAMP}"
SHORT_SHA="$(git rev-parse --short HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

mkdir -p "${RUN_DIR}"
echo "==> Run directory: ${RUN_DIR}"
echo "==> Commit:        $(git rev-parse HEAD)  (${SHORT_SHA})"
echo "==> Branch / ref:  ${BRANCH}"
echo

# ---- Capture environment snapshot --------------------------------------
echo "==> Capturing runner-snapshot.txt"
{
    echo "Commit: $(git rev-parse HEAD)"
    echo "Short:  ${SHORT_SHA}"
    echo "Ref:    ${BRANCH}"
    echo "Date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Host:   $(hostname)"
    echo
    echo "## uname -a"
    uname -a
    echo
    echo "## nvidia-smi"
    nvidia-smi
    echo
    echo "## torch / CUDA"
    python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('device count:', torch.cuda.device_count()); [print('device', i, ':', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
    echo
    echo "## pip freeze"
    python -m pip freeze
} > "${RUN_DIR}/runner-snapshot.txt"

# ---- Full suite with coverage -------------------------------------------
echo
echo "==> Running full pytest with coverage"
FULL_RESULT=PASS
python -m pytest -q \
    --cov=torchkm \
    --cov-report=term-missing:skip-covered \
    --cov-report=xml:"${RUN_DIR}/coverage-cuda.xml" \
    --junitxml="${RUN_DIR}/junit-cuda.xml" \
    2>&1 | tee "${RUN_DIR}/cuda-test-output.log" || FULL_RESULT=FAIL

# ---- CUDA-marked tests --------------------------------------------------
echo
echo "==> Running pytest -m cuda -v"
CUDA_MARK_RESULT=PASS
python -m pytest -m cuda -v 2>&1 | tee "${RUN_DIR}/cuda-marked-tests.log" \
    || CUDA_MARK_RESULT=FAIL

# Detect whether any CUDA-marked test actually ran (rather than being
# skipped because the runner had no GPU). A skip means we accidentally
# ran on a CPU machine and the bundle should be marked FAIL.
if grep -qE "no tests ran|0 selected" "${RUN_DIR}/cuda-marked-tests.log" 2>/dev/null; then
    echo "WARNING: no CUDA-marked tests were collected" >&2
    CUDA_MARK_RESULT=FAIL
elif grep -qE "skipped, 0 passed" "${RUN_DIR}/cuda-marked-tests.log" 2>/dev/null; then
    echo "WARNING: CUDA-marked tests were skipped (not executed)" >&2
    CUDA_MARK_RESULT=FAIL
fi

# ---- Verdict ------------------------------------------------------------
if [[ "${FULL_RESULT}" == "PASS" && "${CUDA_MARK_RESULT}" == "PASS" ]]; then
    OVERALL=PASS
else
    OVERALL=FAIL
fi
echo "${OVERALL}" > "${RUN_DIR}/STATUS"

# ---- Update the index ---------------------------------------------------
INDEX_FILE="benchmarks/cuda-runs/INDEX.md"
GPU_LABEL="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 | tr -d ',')"
if [[ ! -f "${INDEX_FILE}" ]]; then
    {
        echo "# CUDA validation index"
        echo
        echo "Each row records one execution of \`scripts/run_cuda_tests.sh\`."
        echo "See \`docs/developer/cuda_testing.md\` for the protocol details."
        echo
        echo "| Timestamp (UTC) | Status | Commit | Branch | GPU | Bundle |"
        echo "| --- | --- | --- | --- | --- | --- |"
    } > "${INDEX_FILE}"
fi

# Append the run as the most-recent row at the top of the table body
TMP_INDEX="$(mktemp)"
HEADER_LINES="$(grep -n '^| ---' "${INDEX_FILE}" | head -n1 | cut -d: -f1)"
{
    head -n "${HEADER_LINES}" "${INDEX_FILE}"
    echo "| ${TIMESTAMP} | ${OVERALL} | \`${SHORT_SHA}\` | ${BRANCH} | ${GPU_LABEL} | [\`${TIMESTAMP}/\`](./${TIMESTAMP}/) |"
    tail -n +$((HEADER_LINES + 1)) "${INDEX_FILE}"
} > "${TMP_INDEX}"
mv "${TMP_INDEX}" "${INDEX_FILE}"

# ---- Summary ------------------------------------------------------------
echo
echo "============================================================"
echo "  CUDA test run complete"
echo "  Bundle: ${RUN_DIR}"
echo "  Status: ${OVERALL}"
echo "  Commit: ${SHORT_SHA} on ${BRANCH}"
echo "  GPU:    ${GPU_LABEL}"
echo "============================================================"

# ---- Optional commit / push --------------------------------------------
case "${ACTION}" in
    --commit|--push)
        echo
        echo "==> Committing bundle"
        git add "${RUN_DIR}" "${INDEX_FILE}"
        git commit -m "CUDA run ${TIMESTAMP}: ${OVERALL} at ${SHORT_SHA} on ${GPU_LABEL}"
        if [[ "${ACTION}" == "--push" ]]; then
            echo "==> Pushing"
            git push
        else
            echo "(skipping git push — pass --push to also push)"
        fi
        ;;
    collect)
        echo
        echo "(bundle was not committed — re-run with --commit or --push)"
        ;;
esac

# Exit non-zero if the run failed so CI / wrappers see the verdict
[[ "${OVERALL}" == "PASS" ]] || exit 1
