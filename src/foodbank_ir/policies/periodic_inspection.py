"""Periodic inspection policy.

A simple, interpretable baseline: each agency is inspected on a fixed cycle
(every ``inspection_period`` periods, possibly staggered across agencies) and
deliveries are made to bring belief-mean inventory up to a target level
whenever an agency is visited. This isolates the effect of *scheduled* versus
*state-dependent* information acquisition.
"""

from __future__ import annotations

import numpy as np

from foodbank_ir.instance import FoodBankInstance
from foodbank_ir.policies.base import Policy
from foodbank_ir.simulator import Action, BeliefState


class PeriodicInspectionPolicy(Policy):
    """Inspect every agency on a fixed schedule; deliver up-to-target on visits.

    Args:
        instance: Problem instance.
        inspection_period: Number of periods between inspections of the same
            agency.
        target_level: Order-up-to inventory target per ``(agency, item)``,
            shape ``(num_agencies, num_items)``. If ``None``, a default is
            derived from the demand distributions (e.g. a multiple of mean
            demand over the inspection period).
        stagger: If ``True``, spread agencies' inspection times evenly across
            the cycle so that roughly ``num_agencies / inspection_period``
            agencies are inspected each period.
        seed: Random seed for any tie-breaking.
    """

    name = "periodic_inspection"

    def __init__(
        self,
        instance: FoodBankInstance,
        *,
        inspection_period: int = 3,
        target_level: np.ndarray | None = None,
        stagger: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(instance, seed=seed)
        self.inspection_period = inspection_period
        self.target_level = target_level
        self.stagger = stagger

    def act(self, belief: BeliefState) -> Action:
        """Visit and inspect agencies due this period; deliver up to target.

        Args:
            belief: Current belief state.

        Returns:
            The period's :class:`~foodbank_ir.simulator.Action`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the inspection schedule to period 0."""
        raise NotImplementedError

    def agencies_due(self, period: int) -> list[int]:
        """Return agency ids scheduled for inspection in ``period``.

        Args:
            period: The current period index.

        Returns:
            List of agency ids whose inspection falls in this period.
        """
        raise NotImplementedError
