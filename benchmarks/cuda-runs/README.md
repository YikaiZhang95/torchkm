# CUDA test runs

Each subdirectory here is the artefact bundle from a single execution
of the protocol described in
[`docs/developer/cuda_testing.md`](../../docs/developer/cuda_testing.md).

Directory layout:

```
benchmarks/cuda-runs/
└── <UTC-timestamp>/
    ├── STATUS                  # one-line verdict: PASS or FAIL
    ├── runner-snapshot.txt     # nvidia-smi + pip freeze + commit + date
    ├── cuda-test-output.log    # full pytest stdout
    ├── cuda-marked-tests.log   # `pytest -m cuda -v` stdout
    ├── coverage-cuda.xml       # cobertura coverage report
    └── junit-cuda.xml          # JUnit test report
```

Adding a run:

1. Follow the manual protocol in `docs/developer/cuda_testing.md`.
2. Place the bundle under `<UTC-timestamp>/`.
3. Write `PASS` or `FAIL` to `STATUS`.
4. Commit with a message like `CUDA run YYYYMMDDTHHMMSSZ: PASS at <short SHA>`.

If a run was triggered via the `.github/workflows/cuda-tests.yml`
self-hosted workflow, download the published artefact and commit it
here so the in-tree audit trail remains canonical.
