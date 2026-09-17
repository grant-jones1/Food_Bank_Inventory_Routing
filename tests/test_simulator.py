"""Tests for foodbank_ir.simulator."""

import numpy as np
import pytest

from foodbank_ir.instance import generate_random_instance
from foodbank_ir.simulator import Action, Simulator


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_reset_returns_initial_belief():
    """Reset yields a belief at period 0 with the right shapes."""
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    sim = Simulator(instance, seed=0)
    belief = sim.reset()
    assert belief.period == 0
    assert belief.mean.shape == (5, 3)
    assert belief.variance.shape == (5, 3)
    assert belief.periods_since_inspection.shape == (5,)


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_step_with_no_action_only_incurs_holding_and_shortage():
    """An empty action (no routes, deliveries, or inspections) has zero routing/inspection cost."""
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    sim = Simulator(instance, seed=0)
    sim.reset()
    action = Action(
        routes={},
        deliveries=np.zeros((5, 3)),
        inspect=np.zeros(5, dtype=bool),
    )
    result = sim.step(action)
    assert result.cost_breakdown["routing"] == pytest.approx(0.0)
    assert result.cost_breakdown["inspection"] == pytest.approx(0.0)
    assert result.belief.period == 1


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_inspection_collapses_belief_variance():
    """After inspecting an agency, its belief variance is zero and staleness resets."""
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    sim = Simulator(instance, seed=0, inspection_requires_visit=False)
    sim.reset()
    inspect = np.zeros(5, dtype=bool)
    inspect[2] = True
    result = sim.step(Action(routes={}, deliveries=np.zeros((5, 3)), inspect=inspect))
    assert np.allclose(result.belief.variance[2], 0.0)
    assert result.belief.periods_since_inspection[2] == 0
    assert not np.isnan(result.observation[2]).any()
    assert np.isnan(result.observation[0]).all()
