"""Belief-based reorder-point policy.

A state-dependent heuristic in the spirit of ``(s, S)`` policies, adapted to
partial observability: an agency is *visited* when its belief-mean inventory
for any item falls below a reorder point ``s``, and delivered up to ``S``.
Separately, an agency is *inspected* when the belief uncertainty (variance or
periods since last inspection) exceeds a threshold, so that information is
acquired only when the belief has become too diffuse to plan against.
"""

from __future__ import annotations

import numpy as np

from foodbank_ir.instance import FoodBankInstance
from foodbank_ir.policies.base import Policy
from foodbank_ir.simulator import Action, BeliefState


class ReorderPointPolicy(Policy):
    """Visit when belief-mean inventory is low; inspect when belief is diffuse.

    Args:
        instance: Problem instance.
        reorder_point: Per ``(agency, item)`` reorder point ``s``, shape
            ``(num_agencies, num_items)``.
        order_up_to: Per ``(agency, item)`` order-up-to level ``S``, same
            shape; must satisfy ``S >= s`` elementwise.
        uncertainty_threshold: Inspect an agency when its maximum belief
            standard deviation across items exceeds this value.
        max_periods_without_inspection: Force inspection of any agency not
            observed for this many periods regardless of variance (``None``
            disables).
        inspect_on_visit: If ``True``, always inspect any agency that is
            visited for delivery (inspection is then nearly free at the
            margin if it requires a visit anyway).
        seed: Random seed for tie-breaking.
    """

    name = "reorder_point"

    def __init__(
        self,
        instance: FoodBankInstance,
        *,
        reorder_point: np.ndarray,
        order_up_to: np.ndarray,
        uncertainty_threshold: float,
        max_periods_without_inspection: int | None = None,
        inspect_on_visit: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(instance, seed=seed)
        self.reorder_point = reorder_point
        self.order_up_to = order_up_to
        self.uncertainty_threshold = uncertainty_threshold
        self.max_periods_without_inspection = max_periods_without_inspection
        self.inspect_on_visit = inspect_on_visit

    def act(self, belief: BeliefState) -> Action:
        """Apply the reorder-point and uncertainty-threshold rules.

        Args:
            belief: Current belief state.

        Returns:
            The period's :class:`~foodbank_ir.simulator.Action`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """No per-episode state; provided for interface completeness."""
        raise NotImplementedError

    def agencies_to_visit(self, belief: BeliefState) -> list[int]:
        """Return agencies whose belief-mean inventory triggers a delivery.

        Args:
            belief: Current belief state.

        Returns:
            Agency ids with some item below its reorder point.
        """
        raise NotImplementedError

    def agencies_to_inspect(self, belief: BeliefState, visiting: list[int]) -> list[int]:
        """Return agencies whose belief uncertainty triggers an inspection.

        Args:
            belief: Current belief state.
            visiting: Agencies already selected for a delivery visit this
                period (relevant when ``inspect_on_visit`` is set or when
                inspection requires a visit).

        Returns:
            Agency ids to inspect.
        """
        raise NotImplementedError
