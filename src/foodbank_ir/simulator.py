"""Stochastic simulation environment with partial observability.

The environment models the food bank's periodic decision problem as a POMDP:

* The **true state** is the actual inventory of every item at every agency.
* The **belief state** is the decision maker's probability distribution over
  the true inventory, maintained separately per (agency, item). Inventory at an
  agency is only revealed when the food bank *pays to inspect* it; otherwise
  the belief evolves by pushing the demand distribution forward through the
  inventory balance equation.
* Each period the **action** specifies which agencies to visit (route), how
  much of each item to deliver, and which agencies to inspect.

Core logic (transitions, belief updates, cost accounting) is intentionally
unimplemented (``NotImplementedError``); see ``CLAUDE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from foodbank_ir.instance import FoodBankInstance


@dataclass
class Action:
    """One period's decision.

    Attributes:
        routes: For each vehicle index, the ordered list of agency ids to
            visit (depot omitted; the vehicle starts and ends at the depot).
        deliveries: Delivery quantities, shape ``(num_agencies, num_items)``.
            Non-zero entries must correspond to agencies present in
            ``routes``.
        inspect: Boolean mask of shape ``(num_agencies,)`` indicating which
            agencies' inventories to observe this period. Inspecting an
            agency incurs ``instance.inspection_cost``. Whether inspection
            requires a visit is an instance/environment option.
    """

    routes: dict[int, list[int]]
    deliveries: np.ndarray
    inspect: np.ndarray


@dataclass
class TrueState:
    """Hidden, fully-specified state of the environment.

    Attributes:
        inventory: Actual inventory levels, shape ``(num_agencies, num_items)``.
        period: Current period index (0-based).
    """

    inventory: np.ndarray
    period: int = 0


@dataclass
class BeliefState:
    """Decision maker's belief over the hidden inventory.

    The belief is factored across ``(agency, item)`` pairs. Each factor is
    represented by summary statistics and, where the demand family permits, a
    discrete or parametric distribution over inventory levels.

    Attributes:
        mean: Belief mean inventory, shape ``(num_agencies, num_items)``.
        variance: Belief variance, shape ``(num_agencies, num_items)``.
        periods_since_inspection: Periods elapsed since each agency was last
            inspected, shape ``(num_agencies,)``. ``0`` means observed this
            period; large values indicate stale information.
        last_observed_inventory: Inventory observed at the most recent
            inspection of each agency, shape ``(num_agencies, num_items)``.
        period: Current period index (0-based).
        support: Optional explicit discrete support/pmf per ``(agency, item)``
            when exact belief propagation is used (e.g. small Poisson demand).
    """

    mean: np.ndarray
    variance: np.ndarray
    periods_since_inspection: np.ndarray
    last_observed_inventory: np.ndarray
    period: int = 0
    support: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None


@dataclass
class StepResult:
    """Outcome of one call to :meth:`Simulator.step`.

    Attributes:
        cost: Total cost incurred this period.
        cost_breakdown: Cost decomposed into ``routing``, ``holding``,
            ``shortage``, and ``inspection`` components.
        observation: Inventory revealed by inspection this period, shape
            ``(num_agencies, num_items)`` with ``NaN`` for uninspected
            agencies.
        belief: Updated belief state after transition and observation.
        info: Free-form diagnostics (e.g. realised demand, if the environment
            is configured to expose it for analysis).
    """

    cost: float
    cost_breakdown: dict[str, float]
    observation: np.ndarray
    belief: BeliefState
    info: dict[str, Any] = field(default_factory=dict)


class Simulator:
    """Partially observable inventory routing environment.

    Typical usage::

        sim = Simulator(instance, seed=0)
        belief = sim.reset()
        for _ in range(T):
            action = policy.act(belief, sim.instance)
            result = sim.step(action)
            belief = result.belief

    Args:
        instance: The problem instance to simulate.
        seed: Random seed for demand sampling.
        inspection_requires_visit: If ``True``, an agency can only be
            inspected in a period in which it is also visited by a vehicle.
        initial_inventory: Optional known starting inventory, shape
            ``(num_agencies, num_items)``. If ``None``, drawn from a
            generator-defined prior.
        expose_true_state: If ``True``, :attr:`true_state` is accessible
            (useful for oracle policies and testing); policies should never
            rely on it.
    """

    def __init__(
        self,
        instance: FoodBankInstance,
        *,
        seed: int | None = None,
        inspection_requires_visit: bool = True,
        initial_inventory: np.ndarray | None = None,
        expose_true_state: bool = False,
    ) -> None:
        self.instance = instance
        self.seed = seed
        self.inspection_requires_visit = inspection_requires_visit
        self.initial_inventory = initial_inventory
        self.expose_true_state = expose_true_state
        self._rng: np.random.Generator | None = None
        self._true_state: TrueState | None = None
        self._belief: BeliefState | None = None

    @property
    def true_state(self) -> TrueState:
        """Return the hidden true state (only if ``expose_true_state``).

        Raises:
            PermissionError: If the environment was not constructed with
                ``expose_true_state=True``.
        """
        raise NotImplementedError

    @property
    def belief(self) -> BeliefState:
        """Return the current belief state."""
        raise NotImplementedError

    def reset(self, *, seed: int | None = None) -> BeliefState:
        """Reset the environment to period 0 and return the initial belief.

        The initial belief is fully informed (all agencies treated as
        inspected at period 0) if ``initial_inventory`` was provided;
        otherwise it reflects the generator's prior.

        Args:
            seed: Optional seed to override the constructor's seed.

        Returns:
            The initial :class:`BeliefState`.
        """
        raise NotImplementedError

    def step(self, action: Action) -> StepResult:
        """Advance the environment by one period.

        Order of events within a period:
            1. Vehicles execute ``action.routes``; routing cost accrues.
            2. ``action.deliveries`` are added to agency inventories.
            3. Demand is realised and consumed; holding/shortage costs accrue
               on the resulting inventory and unmet demand.
            4. Agencies flagged in ``action.inspect`` are observed at
               post-demand inventory; inspection cost accrues.
            5. The belief is propagated through steps 2-3 and conditioned on
               the observations from step 4.

        Args:
            action: The :class:`Action` to apply.

        Returns:
            A :class:`StepResult` with the period cost and updated belief.

        Raises:
            ValueError: If the action is infeasible (e.g. deliveries to
                unvisited agencies, vehicle capacity exceeded, inspection of
                an unvisited agency when ``inspection_requires_visit``).
        """
        raise NotImplementedError

    def inspect(self, agency_ids: list[int]) -> np.ndarray:
        """Observe the true inventory of the given agencies *without* stepping.

        This is a standalone helper mainly for policies that reason about the
        value of information mid-decision and for tests; the canonical way to
        inspect during simulation is via ``Action.inspect`` in :meth:`step`.

        Args:
            agency_ids: Agencies whose inventory to reveal.

        Returns:
            Array of shape ``(num_agencies, num_items)`` with revealed
            inventory for the listed agencies and ``NaN`` elsewhere.
        """
        raise NotImplementedError

    def sample_demand(self) -> np.ndarray:
        """Sample one period of demand for every ``(agency, item)``.

        Returns:
            Array of shape ``(num_agencies, num_items)`` drawn from
            ``instance.demand``.
        """
        raise NotImplementedError


def propagate_belief(
    belief: BeliefState,
    deliveries: np.ndarray,
    instance: FoodBankInstance,
) -> BeliefState:
    """Push a belief forward one period through the inventory balance.

    Applies deliveries and convolves the current belief with the per-period
    demand distribution (clipped at zero inventory), without conditioning on
    any observation. Used both by :meth:`Simulator.step` and by lookahead
    policies that need to simulate belief evolution.

    Args:
        belief: Prior belief.
        deliveries: Delivery quantities, shape ``(num_agencies, num_items)``.
        instance: Problem instance (provides demand distributions).

    Returns:
        The propagated (predictive) belief.
    """
    raise NotImplementedError


def update_belief(
    belief: BeliefState,
    observation: np.ndarray,
) -> BeliefState:
    """Condition a belief on an inspection observation.

    Inspected agencies' factors collapse to point masses at the observed
    inventory and their ``periods_since_inspection`` resets to zero; all
    other factors are unchanged.

    Args:
        belief: The predictive belief to update.
        observation: Observed inventory, shape ``(num_agencies, num_items)``
            with ``NaN`` for uninspected agencies.

    Returns:
        The posterior belief.
    """
    raise NotImplementedError
