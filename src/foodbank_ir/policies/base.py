"""Abstract base class for decision policies.

A policy maps the current belief state (never the hidden true state) to an
:class:`~foodbank_ir.simulator.Action` covering routing, delivery quantities,
and inspection decisions for one period. Policies may carry internal memory
(e.g. inspection schedules, precomputed routes) and must implement
:meth:`Policy.reset` to clear it between episodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from foodbank_ir.instance import FoodBankInstance
from foodbank_ir.simulator import Action, BeliefState


class Policy(ABC):
    """Base class for all policies.

    Subclasses must implement :meth:`act` and :meth:`reset`, and should set
    :attr:`name` to a short identifier used in result tables and plots.

    Args:
        instance: The problem instance the policy will act on. Policies may
            precompute instance-dependent structures here.
        seed: Random seed for any internal randomness (e.g. rollout
            sampling).
    """

    name: str = "base"

    def __init__(self, instance: FoodBankInstance, *, seed: int | None = None) -> None:
        self.instance = instance
        self.seed = seed

    @abstractmethod
    def act(self, belief: BeliefState) -> Action:
        """Choose this period's routing, delivery, and inspection decisions.

        Args:
            belief: The current belief over agency inventories.

        Returns:
            A feasible :class:`~foodbank_ir.simulator.Action`.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear any per-episode internal state before a new episode."""

    def observe(self, observation: Any, cost: float) -> None:
        """Optional hook called after each environment step.

        The default implementation is a no-op; policies that learn or track
        history online may override it.

        Args:
            observation: The observation array returned by the simulator.
            cost: The cost incurred in the step just completed.
        """
        return None

    def build_action(
        self,
        visits: list[int],
        deliveries: Any,
        inspect: list[int],
    ) -> Action:
        """Helper to assemble a feasible :class:`~foodbank_ir.simulator.Action`.

        Constructs vehicle routes over ``visits`` (e.g. via a routing
        subroutine such as OR-Tools' VRP solver) and packages them with the
        given deliveries and inspection set into an ``Action``.

        Args:
            visits: Agency ids to visit this period.
            deliveries: Delivery quantities, shape ``(num_agencies, num_items)``.
            inspect: Agency ids to inspect this period.

        Returns:
            A feasible ``Action``.
        """
        raise NotImplementedError
