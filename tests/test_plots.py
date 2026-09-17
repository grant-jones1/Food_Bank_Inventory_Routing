"""Tests for foodbank_ir.plots."""

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.figure
import pytest

from foodbank_ir.evaluate import evaluate_policy
from foodbank_ir.instance import generate_random_instance
from foodbank_ir.plots import plot_policy_comparison
from foodbank_ir.policies import PeriodicInspectionPolicy


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_plot_policy_comparison_returns_figure(tmp_path):
    """The comparison plot returns a Figure and writes a file when save_path is given."""
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    policy = PeriodicInspectionPolicy(instance, inspection_period=2)
    result = evaluate_policy(policy, instance, num_episodes=2, horizon=3)

    out = tmp_path / "comparison.png"
    fig = plot_policy_comparison([result], save_path=str(out))

    assert isinstance(fig, matplotlib.figure.Figure)
    assert out.exists()
