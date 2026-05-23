# CUDA testing protocol

The hosted GitHub Actions runners do not provide a GPU, so the test
suite in `tests.yml` runs CPU-only. TorchKM's CUDA code paths are
exercised in two complementary ways:

1. **Self-hosted CUDA workflow** — `.github/workflows/cuda-tests.yml`.
   Dispatches manually or on a weekly schedule onto a runner labelled
   `[self-hosted, linux, cuda]`. The job stays unscheduled until such a
   runner is registered, so it costs nothing in the meantime.
2. **Documented periodic protocol** — this page. Run the protocol on a
   CUDA workstation, then commit the resulting log bundle under
   `benchmarks/cuda-runs/`. The protocol is the source of record while
   no self-hosted runner is online.

Either path produces the same artefact bundle, so reviewers can audit
either one without having to know which was used.

## When to run

- **Before tagging a release.** Every `vX.Y.Z` tag should be accompanied
  by a fresh CUDA log bundle on the release commit.
- **On dependency bumps.** PyTorch / CUDA bumps in `setup.cfg`.
- **After non-trivial solver changes.** Anything touching
  `torchkm/cvk*.py`, `torchkm/cvknys*.py`, or `torchkm/functions.py`.

## Required environment

| Component | Minimum tested | Notes |
| --- | --- | --- |
| OS | Ubuntu 22.04 LTS | Other Linux distros are fine if CUDA drivers are properly installed. |
| Python | 3.10 | Match one of the versions in `tests.yml`. |
| NVIDIA driver | 535+ | Driver must support the CUDA build PyTorch was compiled against. |
| GPU | Any with ≥ 8 GB VRAM | The full suite peaks around 4 GB. |
| PyTorch | The version pinned by `setup.cfg`'s `install_requires` | `pip install -e ".[dev,examples,viz]"` will resolve a CUDA wheel automatically. |

`nvidia-smi` must return a working device. `python -c "import torch;
print(torch.cuda.is_available())"` must print `True`.

## Manual protocol

Run from a fresh clone of the commit you want to verify:

```bash
# 1) Set up an isolated environment
python -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,examples,viz]"

# 2) Sanity check
python -c "import torch; assert torch.cuda.is_available(); \
    print('cuda:', torch.version.cuda, 'device:', torch.cuda.get_device_name(0))"

# 3) Capture run metadata
mkdir -p benchmarks/cuda-runs/$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=$(ls -dt benchmarks/cuda-runs/*/ | head -n1)
{
    echo "Commit: $(git rev-parse HEAD)"
    echo "Ref: $(git rev-parse --abbrev-ref HEAD)"
    echo "Date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "## nvidia-smi"
    nvidia-smi
    echo
    echo "## pip freeze"
    python -m pip freeze
} > "${RUN_DIR}runner-snapshot.txt"

# 4) Full suite with coverage (writes XML + log)
python -m pytest -q \
    --cov=torchkm \
    --cov-report=term-missing:skip-covered \
    --cov-report=xml:"${RUN_DIR}coverage-cuda.xml" \
    --junitxml="${RUN_DIR}junit-cuda.xml" \
    2>&1 | tee "${RUN_DIR}cuda-test-output.log"

# 5) Verbose CUDA-marked tests
python -m pytest -m cuda -v 2>&1 | tee "${RUN_DIR}cuda-marked-tests.log"
```

The protocol should leave the following files under `${RUN_DIR}`:

```
runner-snapshot.txt        # nvidia-smi + pip freeze + commit + ref + date
cuda-test-output.log       # stdout from the full pytest run
cuda-marked-tests.log      # stdout from `pytest -m cuda -v`
coverage-cuda.xml          # cobertura-format coverage report
junit-cuda.xml             # JUnit-format test report
```

## What "passing" means

A CUDA run is considered passing when **all** of the following hold:

- `cuda-test-output.log` ends with `N passed, 0 failed` (skips are
  allowed — environment-specific skips are fine, but neither
  `test_torchkmsvc_cuda_smoke` nor `test_device_cuda` should be among
  the skipped tests).
- `cuda-marked-tests.log` shows each `@pytest.mark.cuda` test executing
  on the GPU (i.e. the skipif-on-cpu marker did not trigger).
- `coverage-cuda.xml` reports overall coverage `≥ 90 %` — the same
  threshold the CPU CI enforces. CUDA-only branches typically push
  this slightly higher than the CPU-only run.

If a run fails, file a GitHub issue referencing the run directory and
link the artefact bundle. Do not delete failing runs from
`benchmarks/cuda-runs/` — they are part of the historical record.

## Committing the artefacts

`benchmarks/cuda-runs/<UTC-timestamp>/` is checked in to the repo so
that reviewers can audit which commits have been validated on GPU and
when. Each directory should also contain a one-line `STATUS` file with
the verdict (`PASS` or `FAIL`) so the catalogue can be scanned
quickly:

```bash
echo "PASS" > "${RUN_DIR}STATUS"
git add benchmarks/cuda-runs/$(basename "${RUN_DIR}")
git commit -m "CUDA run $(basename ${RUN_DIR}): PASS at $(git rev-parse --short HEAD)"
```

Logs are typically a few tens of kilobytes each, so the repository
size impact is negligible. If the bundle ever grows beyond a couple of
MB, move it to git LFS or to a release attachment and replace the
in-tree directory with a stub linking to the artefact.

## Running the self-hosted workflow

Once a runner with labels `self-hosted, linux, cuda` is registered:

1. Trigger via the **Run workflow** button on the Actions tab and pick
   a ref, or wait for the scheduled Monday 07:00 UTC run.
2. The job uploads the same five files described above as a build
   artefact (`cuda-test-artifacts-<run-id>`, retained 90 days).
3. After the run completes, download the artefact and commit it under
   `benchmarks/cuda-runs/` so the in-repo audit trail remains the
   canonical record.

The workflow file itself (`.github/workflows/cuda-tests.yml`) is the
source of truth for which commands are run — keep this document and
the workflow file in sync if either is updated.
