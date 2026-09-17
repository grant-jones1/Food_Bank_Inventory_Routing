"""Tests for foodbank_ir.evaluate."""

import pytest

from foodbank_ir.evaluate import evaluate_policy
from foodbank_ir.instance import generate_random_instance
from foodbank_ir.policies import PeriodicInspectionPolicy


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_evaluate_policy_aggregates_episodes():
    """Evaluation returns one EpisodeResult per episode and a finite mean/CI."""
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    policy = PeriodicInspectionPolicy(instance, inspection_period=2)
    result = evaluate_policy(policy, instance, num_episodes=4, horizon=5, base_seed=0)

    assert result.policy_name == "periodic_inspection"
    assert len(result.episodes) == 4
    assert result.mean_cost >= 0.0
    assert result.ci95[0] <= result.mean_cost <= result.ci95[1]
    assert set(result.mean_breakdown) == {"routing", "holding", "shortage", "inspection"}


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_evaluate_policy_is_reproducible_under_common_random_numbers():
    """Same base_seed gives identical episode costs across two runs."""
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    policy = PeriodicInspectionPolicy(instance, inspection_period=2)
    a = evaluate_policy(policy, instance, num_episodes=3, horizon=5, base_seed=7)
    b = evaluate_policy(policy, instance, num_episodes=3, horizon=5, base_seed=7)
    assert [e.total_cost for e in a.episodes] == [e.total_cost for e in b.episodes]
