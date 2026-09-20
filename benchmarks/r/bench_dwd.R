#!/usr/bin/env Rscript
# Kernel DWD baseline in R (kerndwd, Wang and Zou 2018) on the splits exported
# by benchmarks/bench_dwd.py (--export-splits DIR).
#
#   Rscript benchmarks/r/bench_dwd.R SPLIT_DIR OUT_CSV [--grid 50] [--cmax 1e3] [--cmin 1e-3] [--qval 1]
#
# For every "<dataset>_rep<r>_train.csv" (features x0..xp, label y in {-1, 1},
# fold id in 1..K) and its "_test.csv", the script tunes kerndwd over the same
# lambda grid with the same folds (cv.kerndwd with foldid), refits at the
# selected lambda, and appends one CSV row with accuracy, balanced accuracy,
# AUC (from the link scores) and the wall-clock time of the whole
# tune-and-fit pipeline. Merge with
#   python benchmarks/make_tables.py dwd.json --r-csv OUT_CSV
#
# The RBF kernel uses kernlab's rbfdot(sigma_k) with k = exp(-sigma_k d^2);
# sigma_k is estimated with kernlab::sigest on the training features.
#
# Package API (kerndwd 2.0): kerndwd(x, y, kern, lambda, qval, ...),
# cv.kerndwd(x, y, kern, lambda, qval, nfolds, foldid, ...),
# predict(object, kern, x, newx, type = c("class", "link")).

suppressPackageStartupMessages({
  library(kernlab)
  library(kerndwd)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: bench_dwd.R SPLIT_DIR OUT_CSV [--grid 50] [--cmax 1e3] [--cmin 1e-3] [--qval 1]")
split_dir <- args[1]
out_csv <- args[2]
opt <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}
grid_size <- as.integer(opt("--grid", "50"))
c_max <- as.numeric(opt("--cmax", "1e3"))
c_min <- as.numeric(opt("--cmin", "1e-3"))
qval <- as.numeric(opt("--qval", "1"))

auc <- function(y, score) {
  # Rank-based AUC, identical to sklearn.metrics.roc_auc_score for ties-free scores.
  pos <- score[y > 0]
  neg <- score[y <= 0]
  if (length(pos) == 0 || length(neg) == 0) return(NA)
  r <- rank(c(pos, neg))
  (sum(r[seq_along(pos)]) - length(pos) * (length(pos) + 1) / 2) / (length(pos) * length(neg))
}

write_row <- function(row) {
  write.table(as.data.frame(row, stringsAsFactors = FALSE), out_csv,
              sep = ",", row.names = FALSE, col.names = !file.exists(out_csv),
              append = file.exists(out_csv))
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

  set.seed(repeat_id)
  sigma_k <- as.numeric(kernlab::sigest(xtr, scaled = FALSE)[2])
  kern <- kernlab::rbfdot(sigma = sigma_k)
  Cs <- 10^seq(log10(c_max), log10(c_min), length.out = grid_size)
  lambdas <- 1 / (2 * n * Cs)

  t0 <- proc.time()[["elapsed"]]
  status <- "ok"
  score <- NULL
  lam_best <- tryCatch({
    cvfit <- kerndwd::cv.kerndwd(xtr, ytr, kern, lambda = lambdas, qval = qval,
                                 nfolds = max(foldid), foldid = foldid)
    lam_best <- cvfit$lambda.min
    fit <- kerndwd::kerndwd(xtr, ytr, kern, lambda = lam_best, qval = qval)
    score <- as.numeric(predict(fit, kern, xtr, xte, type = "link"))
    lam_best
  }, error = function(e) { status <<- paste("failed:", conditionMessage(e)); NA })
  dt <- proc.time()[["elapsed"]] - t0

  if (!is.null(score)) {
    pred <- ifelse(score > 0, 1, -1)
    acc <- mean(pred == yte)
    tpr <- mean(pred[yte > 0] > 0)
    tnr <- mean(pred[yte <= 0] <= 0)
    bal <- (tpr + tnr) / 2
    a <- auc(yte, score)
  } else {
    acc <- NA; bal <- NA; a <- NA
  }
  write_row(list(dataset = dataset, library = "kerndwd", repeat_id = repeat_id, status = status,
                 time_s = dt, accuracy = acc, balanced_accuracy = bal, auc = a,
                 best_lambda = lam_best, n_train = n, n_test = nrow(xte)))
  message(sprintf("%s rep%d kerndwd %s acc=%s %.1fs", dataset, repeat_id, status,
                  ifelse(is.na(acc), "NA", sprintf("%.4f", acc)), dt))
}
message("rows written to ", out_csv)
