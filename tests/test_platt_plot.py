# SPDX-License-Identifier: MIT
"""Behavioural tests for ``TorchKMSVC.platt_plot`` and the underlying
``PlattScalerTorch`` Newton-with-damping line search.

The basic happy path (``platt_plot()`` on stored training data) is
covered in ``test_estimator_paths.py``; the cases here focus on the
edge inputs — passing ``X`` without ``y``, requesting an unknown
binning strategy, a calibration curve that has to be drawn on a
user-supplied axis, and so on.
"""

import os
import unittest

import numpy as np
import torch


def _make_classifier(probability):
    from torchkm.estimators import TorchKMSVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    est = TorchKMSVC(
        kernel="rbf",
        nC=3,
        cv=3,
        device="cpu",
        random_state=0,
        max_iter=60,
        probability=probability,
    ).fit(X, y)
    return est, X, y


class TestPlattPlotContract(unittest.TestCase):
    """Behavioural contract for ``TorchKMSVC.platt_plot``."""

    def test_plot_requires_probability_true(self):
        # Without probability calibration there is nothing to plot.
        est, _, _ = _make_classifier(probability=False)
        with self.assertRaisesRegex(AttributeError, "probability=True"):
            est.platt_plot()

    def test_predict_proba_requires_probability_true(self):
        # The same contract applies to ``predict_proba``.
        est, X, _ = _make_classifier(probability=False)
        with self.assertRaisesRegex(AttributeError, "probability=False"):
            est.predict_proba(X)

    def test_plot_uses_stored_calibration_data_by_default(self):
        # When neither ``X`` nor ``y`` is provided, ``platt_plot`` should
        # fall back to the calibration scores stored during ``fit``.
        import matplotlib

        matplotlib.use("Agg", force=True)
        est, _, _ = _make_classifier(probability=True)
        ax, stats = est.platt_plot(n_bins=4, annotate_counts=True)
        try:
            self.assertIn("ece", stats)
            self.assertIn("brier", stats)
            self.assertTrue(0.0 <= stats["ece"] <= 1.0)
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")

    def test_plot_accepts_quantile_strategy_with_X_and_y(self):
        # Quantile binning should be selectable when ``X`` and ``y`` are
        # given.
        import matplotlib

        matplotlib.use("Agg", force=True)
        est, X, y = _make_classifier(probability=True)
        ax, stats = est.platt_plot(X=X, y=y, n_bins=3, strategy="quantile")
        try:
            self.assertIn("ece", stats)
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")

    def test_plot_rejects_X_without_y(self):
        # Supplying ``X`` without labels is ambiguous and should be flagged.
        est, X, _ = _make_classifier(probability=True)
        with self.assertRaisesRegex(ValueError, "y must also be provided"):
            est.platt_plot(X=X)

    def test_plot_rejects_unknown_binning_strategy(self):
        # The strategy argument is restricted to ``uniform`` / ``quantile``.
        est, _, _ = _make_classifier(probability=True)
        with self.assertRaisesRegex(ValueError, "uniform.*quantile"):
            est.platt_plot(strategy="bogus")

    def test_plot_reports_when_stored_data_is_missing(self):
        # If the cached calibration data has been cleared, the plotter
        # should refuse to silently fall back to garbage.
        est, _, _ = _make_classifier(probability=True)
        est.platt_scores_ = None
        with self.assertRaisesRegex(AttributeError, "Stored calibration data"):
            est.platt_plot()

    def test_plot_writes_to_user_axis_and_savepath(self):
        # When the caller supplies an axis and a ``savepath`` the plot
        # should be rendered onto that axis and the file written to disk.
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        est, _, _ = _make_classifier(probability=True)
        fig, user_ax = plt.subplots(figsize=(4, 4))
        savepath = "/tmp/torchkm_platt_test.png"
        try:
            ax, _ = est.platt_plot(
                ax=user_ax, savepath=savepath, annotate_counts=False, n_bins=3
            )
            self.assertIs(ax, user_ax)
            self.assertTrue(os.path.exists(savepath))
        finally:
            plt.close("all")
            if os.path.exists(savepath):
                os.remove(savepath)

    def test_plot_handles_one_dimensional_predict_proba(self):
        # Some calibration backends return a 1-D probability vector. The
        # plotter should reshape such input without complaint.
        import matplotlib

        matplotlib.use("Agg", force=True)
        est, X, y = _make_classifier(probability=True)

        original = est.predict_proba

        def fake_predict_proba(X_in):
            return original(X_in)[:, -1]

        est.predict_proba = fake_predict_proba
        try:
            ax, stats = est.platt_plot(X=X, y=y, n_bins=3)
            self.assertIn("ece", stats)
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
            est.predict_proba = original

    def test_plot_rejects_X_and_y_with_mismatched_lengths(self):
        # Predicted probabilities and labels must agree on length.
        est, X, y = _make_classifier(probability=True)
        with self.assertRaisesRegex(ValueError, "same length"):
            est.platt_plot(X=X, y=y[:5])


class TestPlattScalerLineSearch(unittest.TestCase):
    """Stress the Newton-with-damping line search inside ``PlattScalerTorch``.

    A scoring direction that disagrees with the class labels forces the
    initial Newton step to overshoot, so the damped backtracking branch
    has to fire to make progress.
    """

    def test_damped_step_recovers_from_overshooting_newton_direction(self):
        from torchkm.platt import PlattScalerTorch

        f = torch.tensor(
            [-5.0, -4.0, -3.0, -2.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double
        )
        # Anti-correlated labels: positive scores belong to the negative
        # class, so the first Newton step overshoots and must be damped.
        y = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.double)
        model = PlattScalerTorch(device="cpu", max_iter=30, tol=1e-12, reg=1.0)
        model.fit(f, y)
        proba = model.predict_proba(f)
        self.assertTrue(torch.isfinite(proba).all())


if __name__ == "__main__":
    unittest.main()
