import numpy as np
import pytest

from torchkm.estimators import TorchKMDWD, TorchKMKQR, TorchKMLogit, TorchKMSVC


def _tiny_binary_features(n=24):
    rng = np.random.default_rng(123)
    x0 = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([x0, x0**2])
    X += 0.01 * rng.normal(size=X.shape)
    y = np.where(x0 >= 0, 1, -1)
    return X.astype(np.float64), y


def _tiny_regression_features(n=24):
    rng = np.random.default_rng(123)
    x0 = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([x0, x0**2])
    X += 0.01 * rng.normal(size=X.shape)
    y = np.sin(2.0 * x0) + 0.05 * rng.normal(size=n)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.mark.parametrize("estimator_cls", [TorchKMDWD, TorchKMLogit])
def test_low_rank_dwd_and_logit_public_api_cpu(estimator_cls):
    X, y = _tiny_binary_features()
    Cs = np.array([1.0, 0.2], dtype=np.float64)

    clf = estimator_cls(
        kernel="rbf",
        Cs=Cs,
        nC=len(Cs),
        cv=2,
        low_rank=True,
        num_landmarks=6,
        nys_k=3,
        device="cpu",
        max_iter=30,
        solver_gamma=1e-6,
        random_state=123,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:5])
    score = clf.decision_function(X[:5])

    assert pred.shape == (5,)
    assert score.shape == (5,)
    assert np.isfinite(score).all()
    assert clf.best_C_ > 0
    assert clf.low_rank_basis_dim_ > 0
    assert clf.low_rank_basis_dim_ <= 3
    assert clf.num_landmarks_ <= 6
    assert clf.nys_k_ <= 3


@pytest.mark.parametrize("estimator_cls", [TorchKMDWD, TorchKMLogit])
def test_exact_dwd_and_logit_public_api_cpu(estimator_cls):
    X, y = _tiny_binary_features(n=16)
    Cs = np.array([1.0, 0.2], dtype=np.float64)

    clf = estimator_cls(
        kernel="linear",
        Cs=Cs,
        nC=len(Cs),
        cv=2,
        device="cpu",
        max_iter=20,
        solver_gamma=1e-6,
        random_state=123,
        store_path=True,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:4])
    score = clf.decision_function(X[:4])

    assert pred.shape == (4,)
    assert score.shape == (4,)
    assert np.isfinite(score).all()
    assert clf.alpmat_path_ is not None
    assert clf.pred_path_ is not None


@pytest.mark.parametrize("kernel", ["linear", "poly"])
def test_svc_linear_and_poly_kernels_cpu(kernel):
    X, y = _tiny_binary_features()
    Cs = np.array([1.0, 0.2], dtype=np.float64)

    clf = TorchKMSVC(
        kernel=kernel,
        Cs=Cs,
        nC=len(Cs),
        cv=2,
        device="cpu",
        max_iter=30,
        solver_gamma=1e-6,
        random_state=123,
        poly_degree=2,
        poly_coef0=1.0,
        poly_gamma=0.5,
    )
    clf.fit(X, y)

    pred = clf.predict(X[:4])
    score = clf.decision_function(X[:4])

    assert pred.shape == (4,)
    assert score.shape == (4,)
    assert np.isfinite(score).all()


def test_svc_precomputed_kernel_path_cpu():
    X, y = _tiny_binary_features(n=16)
    K = X @ X.T + 0.1 * np.eye(X.shape[0])
    Cs = np.array([1.0, 0.2], dtype=np.float64)

    clf = TorchKMSVC(
        kernel="precomputed",
        Cs=Cs,
        nC=len(Cs),
        cv=2,
        device="cpu",
        max_iter=30,
        solver_gamma=1e-6,
        random_state=123,
    )
    clf.fit(K, y)

    score = clf.decision_function(K[:3, :])
    pred = clf.predict(K[:3, :])

    assert score.shape == (3,)
    assert pred.shape == (3,)
    assert np.isfinite(score).all()

    with pytest.raises(ValueError, match="precomputed"):
        clf.decision_function(K[:3, :3])


@pytest.mark.parametrize("kernel", ["linear", "poly"])
def test_kqr_linear_and_poly_kernels_cpu(kernel):
    X, y = _tiny_regression_features(n=20)
    Cs = np.array([1.0, 0.2], dtype=np.float64)

    reg = TorchKMKQR(
        kernel=kernel,
        Cs=Cs,
        nC=len(Cs),
        cv=2,
        tau=0.5,
        device="cpu",
        max_iter=20,
        solver_gamma=1e-6,
        random_state=123,
        poly_degree=2,
        poly_coef0=1.0,
        poly_gamma=0.5,
    )
    reg.fit(X, y)

    pred = reg.predict(X[:4])

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()


def test_kqr_precomputed_kernel_path_cpu():
    X, y = _tiny_regression_features(n=14)
    K = X @ X.T + 0.1 * np.eye(X.shape[0])
    Cs = np.array([1.0, 0.2], dtype=np.float64)

    reg = TorchKMKQR(
        kernel="precomputed",
        Cs=Cs,
        nC=len(Cs),
        cv=2,
        tau=0.5,
        device="cpu",
        max_iter=20,
        solver_gamma=1e-6,
        random_state=123,
        store_path=True,
    )
    reg.fit(K, y)

    pred = reg.predict(K[:3, :])

    assert pred.shape == (3,)
    assert np.isfinite(pred).all()
    assert reg.alpmat_path_ is not None
    assert reg.pred_path_ is not None

    with pytest.raises(ValueError, match="precomputed"):
        reg.predict(K[:3, :3])


def test_svc_predict_proba_requires_probability_true():
    X, y = _tiny_binary_features()
    clf = TorchKMSVC(
        kernel="linear",
        Cs=np.array([1.0, 0.2]),
        nC=2,
        cv=2,
        device="cpu",
        max_iter=20,
        random_state=123,
    )
    clf.fit(X, y)

    with pytest.raises(AttributeError, match="probability=False"):
        clf.predict_proba(X[:3])


def test_low_rank_validation_errors():
    X, y = _tiny_binary_features()
    Cs = np.array([1.0, 0.2])

    with pytest.raises(ValueError, match="precomputed"):
        TorchKMSVC(
            kernel="precomputed",
            low_rank=True,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X @ X.T, y)

    with pytest.raises(ValueError, match="rbf"):
        TorchKMSVC(
            kernel="linear",
            low_rank=True,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X, y)

    with pytest.raises(ValueError, match="rbf_sigma"):
        TorchKMSVC(
            kernel="rbf",
            low_rank=True,
            rbf_sigma=1.0,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X, y)

    with pytest.raises(ValueError, match="num_landmarks"):
        TorchKMSVC(
            kernel="rbf",
            low_rank=True,
            num_landmarks=0,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X, y)

    with pytest.raises(ValueError, match="nys_k"):
        TorchKMSVC(
            kernel="rbf",
            low_rank=True,
            nys_k=0,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X, y)


def test_low_rank_fit_time_options_are_applied():
    X, y = _tiny_binary_features(n=20)

    clf = TorchKMSVC(
        kernel="rbf",
        Cs=np.array([1.0, 0.2]),
        nC=2,
        cv=2,
        device="cpu",
        max_iter=20,
        solver_gamma=1e-6,
        random_state=123,
    )
    clf.fit(X, y, low_rank=True, num_landmarks=5, nys_k=3)

    assert clf.low_rank is True
    assert clf.num_landmarks == 5
    assert clf.nys_k == 3
    assert clf.num_landmarks_ <= 5
    assert clf.nys_k_ <= 3


def test_kqr_low_rank_validation_errors():
    X, y = _tiny_regression_features()
    Cs = np.array([1.0, 0.2])

    with pytest.raises(ValueError, match="precomputed"):
        TorchKMKQR(
            kernel="precomputed",
            low_rank=True,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X @ X.T, y)

    with pytest.raises(ValueError, match="rbf"):
        TorchKMKQR(
            kernel="linear",
            low_rank=True,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X, y)

    with pytest.raises(ValueError, match="num_landmarks"):
        TorchKMKQR(
            kernel="rbf",
            low_rank=True,
            num_landmarks=0,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X, y)

    with pytest.raises(ValueError, match="nys_k"):
        TorchKMKQR(
            kernel="rbf",
            low_rank=True,
            nys_k=0,
            Cs=Cs,
            nC=2,
            cv=2,
            device="cpu",
        ).fit(X, y)


def test_svc_platt_plot_training_data_if_matplotlib_available():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)

    X, y = _tiny_binary_features(n=24)
    clf = TorchKMSVC(
        kernel="linear",
        Cs=np.array([1.0, 0.2]),
        nC=2,
        cv=2,
        device="cpu",
        probability=True,
        max_iter=30,
        solver_gamma=1e-6,
        random_state=123,
    )
    clf.fit(X, y)

    ax, stats = clf.platt_plot(n_bins=4, annotate_counts=False)

    assert "ece" in stats
    assert "brier" in stats
    assert np.isfinite(stats["ece"])
    assert np.isfinite(stats["brier"])
    assert ax is not None
