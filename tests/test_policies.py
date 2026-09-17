"""Tests for foodbank_ir.policies."""

import numpy as np
import pytest

from foodbank_ir.instance import generate_random_instance
from foodbank_ir.policies import (
    PeriodicInspectionPolicy,
    Policy,
    ReorderPointPolicy,
    RolloutPolicy,
)
from foodbank_ir.simulator import Action, Simulator


def test_policy_base_is_abstract():
    """Policy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Policy(instance=None)  # type: ignore[abstract]


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_periodic_inspection_policy_returns_feasible_action():
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    sim = Simulator(instance, seed=0)
    belief = sim.reset()
    policy = PeriodicInspectionPolicy(instance, inspection_period=2)
    policy.reset()
    action = policy.act(belief)
    assert isinstance(action, Action)
    assert action.deliveries.shape == (5, 3)
    assert action.inspect.shape == (5,)


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_reorder_point_policy_visits_when_below_s():
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    sim = Simulator(instance, seed=0)
    belief = sim.reset()
    policy = ReorderPointPolicy(
        instance,
        reorder_point=np.full((5, 3), 1e9),  # everything is "below s"
        order_up_to=np.full((5, 3), 2e9),
        uncertainty_threshold=1e9,
    )
    policy.reset()
    assert set(policy.agencies_to_visit(belief)) == {0, 1, 2, 3, 4}


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_rollout_policy_wraps_base_policy():
    instance = generate_random_instance(num_agencies=3, num_items=2, seed=0)
    sim = Simulator(instance, seed=0)
    belief = sim.reset()
    base = PeriodicInspectionPolicy(instance, inspection_period=2)
    policy = RolloutPolicy(instance, base_policy=base, horizon=2, num_samples=2, seed=0)
    policy.reset()
    action = policy.act(belief)
    assert isinstance(action, Action)
