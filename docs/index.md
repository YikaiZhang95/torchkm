# TorchKM

TorchKM is a PyTorch-based library for kernel machines with fast training and model selection. It is designed for users who want the statistical behavior of classical kernel methods while taking advantage of GPU-friendly linear algebra.

The main idea behind TorchKM is simple: in many kernel-machine workflows, the bottleneck is not fitting one model, but fitting many models across cross-validation folds and tuning parameters. TorchKM integrates training and tuning so that repeated matrix computations can be reused.

## What TorchKM provides

TorchKM currently focuses on binary kernel classifiers, kernel quantile
regression, and related model-selection routines. It provides:

- kernel support vector classification;
- kernel distance-weighted discrimination;
- kernel logistic regression;
- kernel quantile regression;
- pathwise model selection over a grid of regularization values;
- exact cross-validation reuse for kernel machines;
- GPU acceleration through PyTorch/CUDA, with CPU fallback;
- Nyström approximation for larger data sets;
- a scikit-learn-style estimator interface for common workflows.

## Why use TorchKM?

TorchKM is useful when you want to tune nonlinear kernel classifiers without repeatedly refitting a separate model for every fold and every regularization value. It is especially helpful when the full training-and-tuning pipeline is the expensive part of the analysis.

## Quick links

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Model selection](user_guide/model_selection.md)
- [Kernel SVM](user_guide/svm.md)
- [Nyström approximation](user_guide/nystrom.md)
- [Probability calibration](user_guide/probability_calibration.md)
- [API reference](api/estimators.md)
- [Developer guide](developer/architecture.md)
- [Reproducing paper benchmarks](examples/reproduce_paper_benchmarks.md)
- [FAQ](faq.md)

## Citation

If you use TorchKM in academic work, please cite the software paper listed in the repository README.
