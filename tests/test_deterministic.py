"""Tests for foodbank_ir.deterministic (full-information MILP)."""

import numpy as np
import pytest

from foodbank_ir.deterministic import solve_full_information
from foodbank_ir.instance import generate_random_instance


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_full_information_milp_solves_tiny_instance():
    """A 3-agency, 2-item, 2-period MILP solves to optimality with consistent shapes."""
    instance = generate_random_instance(num_agencies=3, num_items=2, seed=0)
    horizon = 2
    initial_inventory = np.zeros((3, 2))
    demand = np.ones((horizon, 3, 2))

    solution = solve_full_information(instance, horizon, initial_inventory, demand)

    assert solution.solver_status.lower() in {"optimal", "ok"}
    assert solution.deliveries.shape == (horizon, 3, 2)
    assert solution.inventory.shape == (horizon, 3, 2)
    assert solution.shortages.shape == (horizon, 3, 2)
    assert solution.objective >= 0.0


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_zero_demand_yields_zero_deliveries():
    """With zero demand and zero initial inventory, no deliveries or routing are needed."""
    instance = generate_random_instance(num_agencies=3, num_items=2, seed=0)
    horizon = 2
    solution = solve_full_information(
        instance, horizon, np.zeros((3, 2)), np.zeros((horizon, 3, 2))
    )
    assert np.allclose(solution.deliveries, 0.0)
    assert solution.objective == pytest.approx(0.0)
