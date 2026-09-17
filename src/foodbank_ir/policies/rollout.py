"""Rollout (one-step lookahead / approximate dynamic programming) policy.

At each period the policy enumerates a candidate set of actions (visit sets,
delivery quantities, and inspection sets), and for each candidate estimates
the discounted cost-to-go by simulating a *base policy* forward over a finite
horizon from the resulting belief, averaging over sampled demand and
inspection outcomes. The candidate with the lowest estimated total cost is
executed. This is the primary ADP method for quantifying the value of
inspection beyond what fixed-rule heuristics capture.
"""

from __future__ import annotations

from typing import Any

from foodbank_ir.instance import FoodBankInstance
from foodbank_ir.policies.base import Policy
from foodbank_ir.simulator import Action, BeliefState


class RolloutPolicy(Policy):
    """One-step lookahead with Monte Carlo rollout of a base policy.

    Args:
        instance: Problem instance.
        base_policy: The policy used to simulate cost-to-go from each
            candidate post-decision belief (typically a
            :class:`~foodbank_ir.policies.reorder_point.ReorderPointPolicy`).
        horizon: Number of periods to roll the base policy forward.
        num_samples: Number of Monte Carlo trajectories per candidate action.
        candidate_generator: Optional callable ``(belief, instance) -> list[Action]``
            producing the candidate action set; if ``None``, a default
            generator enumerates a bounded neighbourhood of the base policy's
            own action (toggling inspections, adding/removing visits).
        max_candidates: Upper bound on candidates evaluated per period, to
            control computational cost.
        seed: Random seed for rollout sampling.
    """

    name = "rollout"

    def __init__(
        self,
        instance: FoodBankInstance,
        *,
        base_policy: Policy,
        horizon: int = 5,
        num_samples: int = 20,
        candidate_generator: Any | None = None,
        max_candidates: int = 50,
        seed: int | None = None,
    ) -> None:
        super().__init__(instance, seed=seed)
        self.base_policy = base_policy
        self.horizon = horizon
        self.num_samples = num_samples
        self.candidate_generator = candidate_generator
        self.max_candidates = max_candidates

    def act(self, belief: BeliefState) -> Action:
        """Evaluate candidate actions by rollout and return the best.

        Args:
            belief: Current belief state.

        Returns:
            The candidate :class:`~foodbank_ir.simulator.Action` with the
            lowest estimated immediate + discounted cost-to-go.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the policy and its base policy for a new episode."""
        raise NotImplementedError

    def generate_candidates(self, belief: BeliefState) -> list[Action]:
        """Produce the candidate action set for the current belief.

        Args:
            belief: Current belief state.

        Returns:
            A list of at most ``max_candidates`` feasible actions.
        """
        raise NotImplementedError

    def estimate_cost_to_go(self, belief: BeliefState, action: Action) -> float:
        """Estimate expected discounted cost of ``action`` followed by the base policy.

        Simulates ``num_samples`` trajectories of length ``horizon`` from a
        copy of the environment initialised at ``belief``, with the true
        state sampled from the belief, and averages the discounted costs.

        Args:
            belief: Current belief state.
            action: The candidate first action.

        Returns:
            Sample-average discounted cost over the rollout horizon.
        """
        raise NotImplementedError
