# Low-Level Solvers API

This page documents low-level solvers in TorchKM. These are intended for advanced users who want direct access to the numerical routines.

## Kernel SVM

::: torchkm.cvksvm.cvksvm

## Kernel DWD

::: torchkm.cvkdwd.cvkdwd

## Kernel Logistic Regression

::: torchkm.cvklogit.cvklogit

## Kernel Quantile Regression

::: torchkm.cvkqr.cvkqr

## Nyström SVM

::: torchkm.cvknyssvm.cvknyssvm

## Nyström DWD

::: torchkm.cvknysdwd.cvknysdwd

## Nyström Logistic Regression

::: torchkm.cvknyslogit.cvknyslogit

## Nyström Quantile Regression

::: torchkm.cvknyqr.cvknyqr

## Notes

The solver docs above are generated from the existing source docstrings and
signatures. Low-level solvers generally expect torch tensors, explicit fold
assignments or fold counts, tuning-parameter grids, and device-aware inputs. The
high-level estimators handle more input conversion and CPU fallback for common
workflows.
