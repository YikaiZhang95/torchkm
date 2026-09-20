# Archived benchmark results

Each subdirectory is one paper-scale run of the scripts in `benchmarks/`:

```
benchmarks/results/
└── <UTC-timestamp>/
    ├── README.md            commit, GPU, commands, anything unusual
    ├── envelope.json        bench_memory_envelope.py
    ├── scaling.json         bench_gpu_libraries.py --suite scaling
    ├── exact.json           bench_gpu_libraries.py --suite exact
    ├── imbalanced.json      bench_gpu_libraries.py --suite imbalanced
    ├── scale.json           bench_gpu_libraries.py --suite scale
    ├── covtype_rank.json    bench_covtype_rank.py
    ├── kqr.json, kqr_r.csv  bench_kqr.py and r/bench_kqr.R
    ├── dwd.json, dwd_r.csv  bench_dwd.py and r/bench_dwd.R
    ├── solver_quality.json  bench_solver_quality.py
    └── sessionInfo.txt      R session for the CSV rows
```

Every JSON file carries its own environment snapshot. Generate the tables
with `python benchmarks/make_tables.py <files>`; the paper and the
documentation quote those tables. Do not commit the exported CSV splits
(`*_splits/`), which are large and reproducible from the seeds.
