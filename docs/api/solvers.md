# Low-Level Solvers API

This page documents low-level solvers in TorchKM. These are intended for advanced users who want direct access to the numerical routines.

## Kernel SVM

::: torchkm.cvksvm.cvksvm

## Kernel SVM, truncated spectrum

The solver of `TorchKMSVC(spectrum="truncated")` and `TorchKMSVC(low_rank=True)`.
`torchkm.experimental` keeps a copy of it for exploring changes.

::: torchkm.cvksvm.SpectralSVMPath

## Kernel DWD

::: torchkm.cvkdwd.cvkdwd

## Kernel Logistic Regression

::: torchkm.cvklogit.cvklogit

## Kernel Quantile Regression

::: torchkm.cvkqr.cvkqr

## Notes

The solver docs above are generated from the existing source docstrings and
signatures. Low-level solvers generally expect torch tensors, explicit fold
assignments or fold counts, tuning-parameter grids, and device-aware inputs. The
high-level estimators handle more input conversion and CPU fallback for common
workflows.
