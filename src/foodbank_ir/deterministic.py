"""Full-information MILP baseline for multi-item inventory routing.

This module formulates and solves the *deterministic*, full-information
version of the problem over a finite horizon: given known initial inventories
and a fixed (e.g. expected or sampled) demand trajectory, jointly choose
vehicle routes and multi-item delivery quantities each period to minimise
routing + holding + shortage cost. There is no inspection decision here, since
all inventory is observed; the model serves as a lower bound / benchmark for
the POMDP policies.

Models are built with ``pyomo`` and solved with HiGHS (via ``highspy``) by
default. Core logic is intentionally unimplemented (``NotImplementedError``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from foodbank_ir.instance import FoodBankInstance


@dataclass
class DeterministicSolution:
    """Solution of the full-information MILP.

    Attributes:
        objective: Optimal (or best-found) objective value.
        routes: For each period ``t`` and vehicle ``v``, the ordered sequence of
            agency ids visited, starting and ending at the depot (depot omitted).
        deliveries: Delivery quantities as an array of shape
            ``(horizon, num_agencies, num_items)``.
        inventory: Resulting end-of-period inventory, same shape as
            ``deliveries``.
        shortages: Unmet demand per period/agency/item, same shape as
            ``deliveries``.
        solver_status: Termination condition reported by the solver.
        solve_time_sec: Wall-clock solve time in seconds.
        mip_gap: Final relative MIP gap, if reported by the solver.
        extras: Any additional diagnostics (e.g. node count, dual bound).
    """

    objective: float
    routes: dict[tuple[int, int], list[int]]
    deliveries: np.ndarray
    inventory: np.ndarray
    shortages: np.ndarray
    solver_status: str
    solve_time_sec: float
    mip_gap: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def build_model(
    instance: FoodBankInstance,
    horizon: int,
    initial_inventory: np.ndarray,
    demand: np.ndarray,
    *,
    subtour_elimination: str = "mtz",
    allow_split_deliveries: bool = False,
) -> Any:
    """Build the full-information inventory routing MILP as a pyomo model.

    Decision variables (per period ``t``):
        * ``x[t, v, i, j]`` binary arc-traversal variables for each vehicle,
        * ``y[t, v, i]`` binary visit indicators,
        * ``q[t, i, k]`` continuous delivery quantity of item ``k`` to agency
          ``i``,
        * ``I[t, i, k]`` continuous end-of-period inventory,
        * ``s[t, i, k]`` continuous shortage (unmet demand).

    Constraints:
        * inventory balance ``I[t] = I[t-1] + q[t] - demand[t] + s[t]``,
        * deliveries only to visited agencies, respecting vehicle and agency
          storage capacities,
        * vehicle flow conservation and depot departure/return,
        * subtour elimination (MTZ or lazy cuts, per ``subtour_elimination``).

    Objective: minimise discounted sum of routing cost + holding cost +
    shortage cost over the horizon.

    Args:
        instance: Problem instance (network, items, costs).
        horizon: Number of periods ``T``.
        initial_inventory: Known starting inventory, shape
            ``(num_agencies, num_items)``.
        demand: Known/assumed demand trajectory, shape
            ``(horizon, num_agencies, num_items)``.
        subtour_elimination: Subtour elimination scheme, one of ``"mtz"`` or
            ``"lazy"``.
        allow_split_deliveries: If ``True``, an agency may be served by more
            than one vehicle in the same period.

    Returns:
        A ``pyomo.environ.ConcreteModel`` ready to be passed to
        :func:`solve_model`.

    Raises:
        ValueError: If array shapes are inconsistent with ``instance`` or
            ``horizon``.
    """
    raise NotImplementedError


def solve_model(
    model: Any,
    *,
    solver: str = "appsi_highs",
    time_limit_sec: float | None = None,
    mip_gap: float | None = None,
    tee: bool = False,
) -> DeterministicSolution:
    """Solve a model built by :func:`build_model` and extract its solution.

    Args:
        model: The pyomo model to solve.
        solver: Pyomo solver name; defaults to HiGHS via the ``appsi``
            interface (requires ``highspy``).
        time_limit_sec: Optional wall-clock time limit.
        mip_gap: Optional relative MIP gap tolerance.
        tee: If ``True``, stream solver output to stdout.

    Returns:
        A :class:`DeterministicSolution` with routes, deliveries, and
        diagnostics.

    Raises:
        RuntimeError: If the solver reports an infeasible or error status.
    """
    raise NotImplementedError


def solve_full_information(
    instance: FoodBankInstance,
    horizon: int,
    initial_inventory: np.ndarray,
    demand: np.ndarray,
    **solver_kwargs: Any,
) -> DeterministicSolution:
    """Convenience wrapper: build and solve the full-information MILP.

    Equivalent to ``solve_model(build_model(...), **solver_kwargs)``.

    Args:
        instance: Problem instance.
        horizon: Number of periods.
        initial_inventory: Starting inventory, shape
            ``(num_agencies, num_items)``.
        demand: Demand trajectory, shape ``(horizon, num_agencies, num_items)``.
        **solver_kwargs: Forwarded to :func:`solve_model`.

    Returns:
        The :class:`DeterministicSolution`.
    """
    raise NotImplementedError


def expected_value_lower_bound(
    instance: FoodBankInstance,
    horizon: int,
    initial_inventory: np.ndarray,
    *,
    num_samples: int = 1,
    seed: int | None = None,
    **solver_kwargs: Any,
) -> float:
    """Compute a full-information (perfect-hindsight) lower bound on cost.

    Samples ``num_samples`` demand trajectories from the instance's demand
    distributions, solves the deterministic MILP for each, and averages the
    optimal objectives. This is the classical "wait-and-see" bound used to
    quantify the cost of uncertainty and the value of information.

    Args:
        instance: Problem instance.
        horizon: Number of periods.
        initial_inventory: Starting inventory, shape
            ``(num_agencies, num_items)``.
        num_samples: Number of demand scenarios to sample and solve.
        seed: Random seed for demand sampling.
        **solver_kwargs: Forwarded to :func:`solve_model`.

    Returns:
        The sample-average perfect-information cost.
    """
    raise NotImplementedError
