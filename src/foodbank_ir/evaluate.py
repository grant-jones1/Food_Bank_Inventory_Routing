"""Policy evaluation by Monte Carlo simulation.

Runs a policy on a :class:`~foodbank_ir.simulator.Simulator` for a number of
independent episodes and aggregates per-episode discounted cost into summary
statistics (mean, standard error, confidence intervals, cost breakdown).
Common random numbers across policies are supported via per-episode seeds so
that policy comparisons are paired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from foodbank_ir.instance import FoodBankInstance
from foodbank_ir.policies.base import Policy


@dataclass
class EpisodeResult:
    """Cost trajectory of a single simulated episode.

    Attributes:
        total_cost: Discounted total cost over the episode.
        cost_breakdown: Discounted cost by component (``routing``,
            ``holding``, ``shortage``, ``inspection``).
        per_period_cost: Undiscounted cost in each period, shape ``(T,)``.
        num_inspections: Total number of agency-inspections performed.
        num_visits: Total number of agency-visits performed.
        seed: The seed used for this episode.
    """

    total_cost: float
    cost_breakdown: dict[str, float]
    per_period_cost: np.ndarray
    num_inspections: int
    num_visits: int
    seed: int | None = None


@dataclass
class EvaluationResult:
    """Aggregate statistics over many episodes of one policy on one instance.

    Attributes:
        policy_name: Identifier of the evaluated policy.
        episodes: All per-episode results.
        mean_cost: Sample mean of ``total_cost`` across episodes.
        std_error: Standard error of the mean.
        ci95: 95% confidence interval ``(lower, upper)`` for the mean.
        mean_breakdown: Mean discounted cost by component.
        metadata: Free-form info (instance id, horizon, wall-clock time, ...).
    """

    policy_name: str
    episodes: list[EpisodeResult]
    mean_cost: float
    std_error: float
    ci95: tuple[float, float]
    mean_breakdown: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return per-episode results as a tidy ``DataFrame`` (one row per episode)."""
        raise NotImplementedError


def run_episode(
    policy: Policy,
    instance: FoodBankInstance,
    horizon: int,
    *,
    seed: int | None = None,
    inspection_requires_visit: bool = True,
) -> EpisodeResult:
    """Simulate one episode of ``policy`` on ``instance``.

    Args:
        policy: The policy to run; ``policy.reset()`` is called first.
        instance: The problem instance.
        horizon: Number of periods to simulate (a truncation of the
            infinite-horizon problem; choose large enough relative to the
            discount factor that truncation error is negligible).
        seed: Seed for the simulator's demand sampling.
        inspection_requires_visit: Forwarded to the simulator.

    Returns:
        The :class:`EpisodeResult`.
    """
    raise NotImplementedError


def evaluate_policy(
    policy: Policy,
    instance: FoodBankInstance,
    *,
    num_episodes: int = 100,
    horizon: int = 50,
    base_seed: int = 0,
    inspection_requires_visit: bool = True,
    n_jobs: int = 1,
) -> EvaluationResult:
    """Evaluate ``policy`` over ``num_episodes`` independent episodes.

    Episode ``i`` uses seed ``base_seed + i`` so that evaluating several
    policies with the same ``base_seed`` yields common random numbers.

    Args:
        policy: The policy to evaluate.
        instance: The problem instance.
        num_episodes: Number of independent episodes.
        horizon: Periods per episode.
        base_seed: Seed offset for episode seeds.
        inspection_requires_visit: Forwarded to the simulator.
        n_jobs: Number of parallel worker processes (``1`` = serial).

    Returns:
        The aggregated :class:`EvaluationResult`.
    """
    raise NotImplementedError


def compare_policies(
    policies: list[Policy],
    instance: FoodBankInstance,
    **evaluate_kwargs: Any,
) -> pd.DataFrame:
    """Evaluate several policies with common random numbers and tabulate.

    Args:
        policies: Policies to compare.
        instance: The problem instance.
        **evaluate_kwargs: Forwarded to :func:`evaluate_policy` (all policies
            receive the same ``base_seed``).

    Returns:
        A ``DataFrame`` with one row per policy and columns for mean cost,
        standard error, confidence bounds, and mean cost by component.
    """
    raise NotImplementedError


def value_of_information(
    policy_with_inspection: Policy,
    policy_without_inspection: Policy,
    instance: FoodBankInstance,
    **evaluate_kwargs: Any,
) -> dict[str, float]:
    """Estimate the value of information for a given instance.

    Computes the paired difference in mean cost between a policy that may
    inspect and an otherwise-identical policy that never inspects, together
    with the paired standard error and the total inspection spend of the
    inspecting policy.

    Args:
        policy_with_inspection: Policy allowed to inspect.
        policy_without_inspection: The same policy with inspection disabled.
        instance: The problem instance.
        **evaluate_kwargs: Forwarded to :func:`evaluate_policy`.

    Returns:
        Dict with keys ``"voi"`` (cost reduction from inspection),
        ``"voi_std_error"``, ``"inspection_spend"``, and ``"net_voi"``.
    """
    raise NotImplementedError
