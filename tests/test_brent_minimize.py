# SPDX-License-Identifier: MIT
"""The solvers' intercept search: Brent's method with its state in Python floats."""

import torch

from torchkm.functions import brent_minimize


def test_brent_minimize_finds_the_minimum_with_one_read_per_evaluation():
    calls = []

    def f(b):
        calls.append(b)
        return torch.tensor((b - 1.234) ** 2 + 0.5)  # a one-element tensor, as objfun

    x, fx = brent_minimize(f, -100.0, 100.0)
    assert isinstance(x, float) and isinstance(fx, float)
    assert abs(x - 1.234) < 1e-3 and abs(fx - 0.5) < 1e-6
    assert all(isinstance(b, float) for b in calls)
    assert len(calls) < 60


def test_brent_minimize_on_a_hinge_intercept_matches_a_grid():
    torch.manual_seed(1)
    f_x = torch.randn(200, dtype=torch.double)
    y = torch.where(torch.randn(200, dtype=torch.double) + f_x > 0, 1.0, -1.0)

    def hinge(b):
        return torch.clamp(1.0 - y * (f_x + b), min=0.0).mean()

    x, fx = brent_minimize(hinge, -100.0, 100.0)
    grid = torch.linspace(-5.0, 5.0, 20001, dtype=torch.double)
    best = min(float(hinge(float(b))) for b in grid)
    assert fx <= best + 1e-4
