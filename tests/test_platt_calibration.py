import numpy as np
import pytest
import torch

from torchkm.platt import PlattScalerTorch


def _platt_data():
    f = torch.tensor([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0], dtype=torch.double)
    y = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1], dtype=torch.double)
    return f, y


def test_platt_predict_proba_before_fit_raises():
    model = PlattScalerTorch(device="cpu")
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(torch.zeros(3, dtype=torch.double))


def test_platt_fit_predict_predict_proba_cpu():
    f, y = _platt_data()

    model = PlattScalerTorch(device="cpu", max_iter=50, tol=1e-10)
    model.fit(f, y)

    proba = model.predict_proba(f)
    pred = model.predict(f)

    assert proba.shape == (f.numel(), 2)
    assert torch.isfinite(proba).all()
    assert torch.allclose(proba.sum(dim=1), torch.ones(f.numel(), dtype=torch.double))
    assert pred.shape == (f.numel(),)
    assert set(pred.cpu().numpy().tolist()).issubset({-1.0, 1.0})


def test_platt_reliability_curve_and_ece():
    f, y = _platt_data()

    model = PlattScalerTorch(device="cpu", max_iter=50)
    model.fit(f, y)
    proba = model.predict_proba(f)[:, 1]

    bin_centers, mean_pred, frac_pos, counts = model.reliability_curve(
        y, proba, n_bins=4
    )
    ece = model.expected_calibration_error(mean_pred, frac_pos, counts)

    assert bin_centers.shape == (4,)
    assert mean_pred.shape == (4,)
    assert frac_pos.shape == (4,)
    assert counts.shape == (4,)
    assert counts.sum() == f.numel()
    assert np.isfinite(ece)
    assert 0.0 <= ece <= 1.0


def test_platt_reliability_curve_converts_integer_labels():
    model = PlattScalerTorch(device="cpu")
    y = np.array([-1, -1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])

    _, mean_pred, frac_pos, counts = model.reliability_curve(y, p, n_bins=2)
    ece = model.expected_calibration_error(mean_pred, frac_pos, counts)
    brier = model.brier_score(y, p)

    assert counts.sum() == y.size
    assert np.isfinite(ece)
    assert np.isfinite(brier)
    assert 0.0 <= brier <= 1.0


def test_platt_plot_calibration_smoke(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    model = PlattScalerTorch(device="cpu")
    shown = {"called": False}

    def _fake_show():
        shown["called"] = True

    monkeypatch.setattr(plt, "show", _fake_show)

    model.plot_calibration(
        bin_centers=np.array([0.25, 0.75]),
        mean_pred=np.array([0.2, 0.8]),
        frac_pos=np.array([0.25, 0.75]),
        counts=np.array([3, 5]),
        show_counts=True,
    )

    assert shown["called"] is True
    plt.close("all")


def test_platt_handles_nearly_singular_constant_scores():
    f = torch.zeros(8, dtype=torch.double)
    y = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1], dtype=torch.double)

    model = PlattScalerTorch(device="cpu", max_iter=3, reg=0.0)
    model.fit(f, y)
    proba = model.predict_proba(f)

    assert proba.shape == (8, 2)
    assert torch.isfinite(proba).all()
