"""Comparison plots for the computational study.

All functions accept tidy ``pandas`` inputs produced by
:mod:`foodbank_ir.evaluate` and return the ``matplotlib`` ``Figure`` so
callers can save or further customise it. Nothing is displayed or written to
disk unless ``save_path`` is given.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import pandas as pd

from foodbank_ir.evaluate import EvaluationResult


def plot_policy_comparison(
    results: list[EvaluationResult],
    *,
    save_path: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Bar chart of mean discounted cost per policy with 95% CI error bars.

    Args:
        results: One :class:`~foodbank_ir.evaluate.EvaluationResult` per
            policy.
        save_path: If given, save the figure to this path.
        **kwargs: Forwarded to ``matplotlib`` for styling.

    Returns:
        The figure.
    """
    raise NotImplementedError


def plot_cost_breakdown(
    results: list[EvaluationResult],
    *,
    save_path: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Stacked bar chart of mean cost by component (routing/holding/shortage/inspection).

    Args:
        results: One result per policy.
        save_path: If given, save the figure to this path.
        **kwargs: Forwarded to ``matplotlib`` for styling.

    Returns:
        The figure.
    """
    raise NotImplementedError


def plot_value_of_information(
    sweep: pd.DataFrame,
    *,
    x: str = "inspection_cost",
    save_path: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Line plot of estimated value of information against a swept parameter.

    Args:
        sweep: DataFrame with one row per swept setting and columns ``x``,
            ``"voi"``, and ``"voi_std_error"`` (see
            :func:`foodbank_ir.evaluate.value_of_information`).
        x: Column to use on the horizontal axis (e.g. ``"inspection_cost"``
            or ``"demand_cv"``).
        save_path: If given, save the figure to this path.
        **kwargs: Forwarded to ``matplotlib`` for styling.

    Returns:
        The figure.
    """
    raise NotImplementedError


def plot_inspection_frequency(
    results: list[EvaluationResult],
    *,
    save_path: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Plot mean inspections per period for each policy.

    Args:
        results: One result per policy.
        save_path: If given, save the figure to this path.
        **kwargs: Forwarded to ``matplotlib`` for styling.

    Returns:
        The figure.
    """
    raise NotImplementedError


def plot_cost_trajectory(
    result: EvaluationResult,
    *,
    save_path: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Plot mean per-period cost over the horizon with a shaded CI band.

    Useful for checking that the truncated horizon has reached steady state.

    Args:
        result: A single policy's evaluation result.
        save_path: If given, save the figure to this path.
        **kwargs: Forwarded to ``matplotlib`` for styling.

    Returns:
        The figure.
    """
    raise NotImplementedError
