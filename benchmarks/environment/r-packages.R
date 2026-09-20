#!/usr/bin/env Rscript
# R packages for the KQR and DWD baselines.
pkgs <- c("kernlab", "fastkqr", "kerndwd")
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
print(sessionInfo())
