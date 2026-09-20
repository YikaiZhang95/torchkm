#!/usr/bin/env Rscript
# Kernel quantile regression baselines in R on the splits exported by
# benchmarks/bench_kqr.py (--export-splits DIR).
#
#   Rscript benchmarks/r/bench_kqr.R SPLIT_DIR OUT_CSV [--taus 0.1,0.5,0.9] \
#           [--methods fastkqr,kernlab] [--grid 50] [--cmax 1e3] [--cmin 1e-3]
#
# For every "<dataset>_rep<r>_train.csv" in SPLIT_DIR (features x0..xp, target
# y, fold id in 1..K) and its "_test.csv", the script fits
#
#   * fastkqr  (Tang, Gu and Wang, 2026): the reference CPU implementation of
#              the algorithm TorchKM's KQR path is derived from; tuned by
#              K-fold cross-validation over the same lambda grid and the same
#              folds as the Python side;
#   * kernlab::kqr (Karatzoglou et al., 2004): the interior-point QP solver;
#              tuned by the same fold loop (slow: expect minutes per fit at
#              n > 5,000, cap the grid with --grid if needed);
#
# and appends one CSV row per (dataset, repeat, tau, method) with the pinball
# loss and empirical coverage on the test split (same definitions as
# _common.quantile_metrics) and the wall-clock time of the whole tune-and-fit
# pipeline. The Python side merges the rows with
#   python benchmarks/make_tables.py kqr.json --r-csv OUT_CSV
#
# The RBF bandwidth is the kernlab convention sigma_k with
# k(x, x') = exp(-sigma_k ||x - x'||^2); to match TorchKM's kernel
# exp(-2 * sigest * d^2) set sigma_k = 2 * sigest. The Python export does not
# carry sigest, so sigma_k is estimated here with kernlab::sigest on the
# training features, which is what fastkqr's own examples do.
#
# Package APIs (check `?fastkqr::cv.kqr` and `?kernlab::kqr` for the installed
# versions; argument names are those of fastkqr 1.0 and kernlab 0.9):
#   fastkqr::kqr(x, y, lambda, tau, sigma)          fastkqr::cv.kqr(x, y, lambda, tau, sigma, nfolds, foldid)
#   kernlab::kqr(x, y, tau, C, kernel = "rbfdot", kpar = list(sigma = sigma_k))

suppressPackageStartupMessages({
  library(kernlab)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: bench_kqr.R SPLIT_DIR OUT_CSV [--taus 0.1,0.5,0.9] [--methods fastkqr,kernlab] [--grid 50]")
}
split_dir <- args[1]
out_csv <- args[2]
opt <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
taus <- as.numeric(strsplit(opt("--taus", "0.1,0.5,0.9"), ",")[[1]])
methods <- strsplit(opt("--methods", "fastkqr,kernlab"), ",")[[1]]
grid_size <- as.integer(opt("--grid", "50"))
c_max <- as.numeric(opt("--cmax", "1e3"))
c_min <- as.numeric(opt("--cmin", "1e-3"))

has_fastkqr <- requireNamespace("fastkqr", quietly = TRUE)
if ("fastkqr" %in% methods && !has_fastkqr) {
  message("fastkqr is not installed (install.packages('fastkqr')); skipping it")
  methods <- setdiff(methods, "fastkqr")
}

pinball <- function(y, q, tau) {
  u <- y - q
  mean(ifelse(u >= 0, tau * u, (tau - 1) * u))
}

write_row <- function(row) {
  write.table(as.data.frame(row, stringsAsFactors = FALSE), out_csv,
              sep = ",", row.names = FALSE, col.names = !file.exists(out_csv),
              append = file.exists(out_csv))
}

fit_kernlab_cv <- function(x, y, foldid, tau, Cs, sigma_k) {
  # Same fold loop as _common.cv_sweep with a pinball score.
  cv <- sapply(Cs, function(C) {
    mean(sapply(sort(unique(foldid)), function(k) {
      tr <- foldid != k
      fit <- kernlab::kqr(x[tr, , drop = FALSE], y[tr], tau = tau, C = C,
                          kernel = "rbfdot", kpar = list(sigma = sigma_k))
      pinball(y[!tr], as.numeric(predict(fit, x[!tr, , drop = FALSE])), tau)
    }))
  })
  C_best <- Cs[which.min(cv)]
  list(model = kernlab::kqr(x, y, tau = tau, C = C_best, kernel = "rbfdot",
                            kpar = list(sigma = sigma_k)),
       C_best = C_best)
}

train_files <- list.files(split_dir, pattern = "_train\\.csv$", full.names = TRUE)
if (length(train_files) == 0) stop("no *_train.csv files in ", split_dir)

for (train_file in train_files) {
  test_file <- sub("_train\\.csv$", "_test.csv", train_file)
  base <- sub("_train\\.csv$", "", basename(train_file))
  dataset <- sub("_rep[0-9]+$", "", base)
  repeat_id <- as.integer(sub(".*_rep([0-9]+)$", "\\1", base))
  tr <- read.csv(train_file)
  te <- read.csv(test_file)
  xtr <- as.matrix(tr[, grep("^x", names(tr))])
  xte <- as.matrix(te[, grep("^x", names(te))])
  ytr <- tr$y
  yte <- te$y
  foldid <- tr$fold
  n <- nrow(xtr)
  # Target scaling mirrors bench_kqr.standardize_split: centre and scale by the
  # training sd for the solver, score in original units.
  y_mean <- mean(ytr)
  y_sd <- sd(ytr)
  if (!is.finite(y_sd) || y_sd == 0) y_sd <- 1
  ytr_s <- (ytr - y_mean) / y_sd

  set.seed(repeat_id)
  sigma_k <- as.numeric(kernlab::sigest(xtr, scaled = FALSE)[2])
  Cs <- 10^seq(log10(c_max), log10(c_min), length.out = grid_size)
  lambdas <- 1 / (2 * n * Cs)

  for (tau in taus) {
    if ("fastkqr" %in% methods) {
      t0 <- proc.time()[["elapsed"]]
      status <- "ok"
      q <- NULL
      res <- tryCatch({
        cvfit <- fastkqr::cv.kqr(xtr, ytr_s, lambda = lambdas, tau = tau,
                                 sigma = sigma_k, nfolds = max(foldid), foldid = foldid)
        lam_best <- cvfit$lambda.min
        fit <- fastkqr::kqr(xtr, ytr_s, lambda = lam_best, tau = tau, sigma = sigma_k)
        q <- as.numeric(predict(fit, xtr, xte, lambda = lam_best)) * y_sd + y_mean
        lam_best
      }, error = function(e) { status <<- paste("failed:", conditionMessage(e)); NA })
      dt <- proc.time()[["elapsed"]] - t0
      write_row(list(dataset = dataset, library = "fastkqr", repeat_id = repeat_id, tau = tau,
                     status = status, time_s = dt,
                     pinball_loss = if (is.null(q)) NA else pinball(yte, q, tau),
                     coverage = if (is.null(q)) NA else mean(yte <= q),
                     best_lambda = res, n_train = n, n_test = nrow(xte)))
      message(sprintf("%s rep%d tau=%.2f fastkqr %s %.1fs", dataset, repeat_id, tau, status, dt))
    }
    if ("kernlab" %in% methods) {
      t0 <- proc.time()[["elapsed"]]
      status <- "ok"
      q <- NULL
      res <- tryCatch({
        out <- fit_kernlab_cv(xtr, ytr_s, foldid, tau, Cs, sigma_k)
        q <- as.numeric(predict(out$model, xte)) * y_sd + y_mean
        out$C_best
      }, error = function(e) { status <<- paste("failed:", conditionMessage(e)); NA })
      dt <- proc.time()[["elapsed"]] - t0
      write_row(list(dataset = dataset, library = "kernlab_kqr", repeat_id = repeat_id, tau = tau,
                     status = status, time_s = dt,
                     pinball_loss = if (is.null(q)) NA else pinball(yte, q, tau),
                     coverage = if (is.null(q)) NA else mean(yte <= q),
                     best_C = res, n_train = n, n_test = nrow(xte)))
      message(sprintf("%s rep%d tau=%.2f kernlab %s %.1fs", dataset, repeat_id, tau, status, dt))
    }
  }
}
message("rows written to ", out_csv)
