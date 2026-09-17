"""Problem instance definition for the multi-item food bank inventory routing problem.

This module defines the static data of a single problem instance (the network,
item set, demand model, and cost structure) and a helper to generate small
random instances for testing and experimentation.

Core logic (instance generation, sampling, validation) is intentionally left
unimplemented as ``NotImplementedError`` stubs. See ``CLAUDE.md`` for the
project's implementation boundary: this module is scaffolding only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DemandDistribution:
    """Parametric demand distribution for a single (agency, item) pair.

    Attributes:
        family: Name of the distribution family (e.g. ``"poisson"``,
            ``"normal"``, ``"negative_binomial"``).
        params: Distribution parameters keyed by name (e.g. ``{"mean": 10.0}``
            or ``{"mean": 10.0, "std": 3.0}``), interpreted according to
            ``family``.
    """

    family: str
    params: dict[str, float]


@dataclass(frozen=True)
class Agency:
    """A single agency (delivery destination) in the network.

    Attributes:
        agency_id: Unique identifier for the agency.
        name: Human-readable label.
        coordinates: ``(x, y)`` planar coordinates, used to derive travel
            distances/times when an explicit distance matrix is not supplied.
        storage_capacity: Maximum total units the agency can hold, summed
            across item types (``None`` if uncapacitated).
    """

    agency_id: int
    name: str
    coordinates: tuple[float, float]
    storage_capacity: float | None = None


@dataclass(frozen=True)
class Item:
    """A single item type distributed by the food bank.

    Attributes:
        item_id: Unique identifier for the item type.
        name: Human-readable label (e.g. ``"canned_vegetables"``).
        unit_weight: Weight or volume per unit, used for vehicle capacity
            constraints.
        holding_cost: Per-unit, per-period cost of inventory held at an
            agency in excess of target/need.
        shortage_cost: Per-unit, per-period penalty for unmet demand
            (stockout) at an agency.
    """

    item_id: int
    name: str
    unit_weight: float
    holding_cost: float
    shortage_cost: float


@dataclass(frozen=True)
class FoodBankInstance:
    """Static data describing one instance of the inventory routing problem.

    Attributes:
        agencies: All agencies in the network, excluding the depot.
        items: All item types distributed by the food bank.
        depot_coordinates: Planar coordinates of the food bank depot.
        distances: Travel distance/time matrix indexed by
            ``(origin_id, destination_id)``, where ``origin_id``/
            ``destination_id`` are agency ids or the sentinel depot id
            ``-1``. If ``None``, distances are derived from coordinates.
        demand: Demand distribution for each ``(agency_id, item_id)`` pair.
        inspection_cost: Cost of paying to observe an agency's true
            inventory state in a given period.
        vehicle_capacity: Maximum total weight/volume the delivery vehicle(s)
            can carry per period.
        discount_factor: Infinite-horizon discount factor in ``(0, 1]``.
        num_vehicles: Number of delivery vehicles available per period.
        metadata: Free-form provenance info (e.g. random seed, generator
            version) attached when the instance is created.
    """

    agencies: list[Agency]
    items: list[Item]
    depot_coordinates: tuple[float, float]
    distances: dict[tuple[int, int], float] | None
    demand: dict[tuple[int, int], DemandDistribution]
    inspection_cost: float
    vehicle_capacity: float
    discount_factor: float
    num_vehicles: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def distance(self, origin_id: int, destination_id: int) -> float:
        """Return the travel distance/time between two nodes.

        Uses ``self.distances`` if provided, otherwise falls back to
        Euclidean distance between ``self.depot_coordinates`` and/or agency
        coordinates (depot is identified by ``origin_id``/``destination_id``
        ``== -1``).

        Args:
            origin_id: Agency id, or ``-1`` for the depot.
            destination_id: Agency id, or ``-1`` for the depot.

        Returns:
            The travel distance/time between the two nodes.
        """
        raise NotImplementedError

    def num_agencies(self) -> int:
        """Return the number of agencies in the instance (excluding the depot)."""
        raise NotImplementedError

    def num_items(self) -> int:
        """Return the number of item types in the instance."""
        raise NotImplementedError


def generate_random_instance(
    num_agencies: int,
    num_items: int,
    *,
    seed: int | None = None,
    demand_family: str = "poisson",
    area_size: float = 100.0,
    inspection_cost: float = 1.0,
    vehicle_capacity: float = 1000.0,
    discount_factor: float = 0.95,
    num_vehicles: int = 1,
) -> FoodBankInstance:
    """Generate a small random :class:`FoodBankInstance` for testing/experiments.

    Agency and depot coordinates are drawn uniformly at random over a square
    of side ``area_size``; demand parameters and cost coefficients are drawn
    from instance-generator-defined ranges intended to produce a well-scaled,
    non-degenerate instance.

    Args:
        num_agencies: Number of agencies to generate.
        num_items: Number of item types to generate.
        seed: Random seed for reproducibility.
        demand_family: Distribution family to use for all
            ``DemandDistribution`` objects (see :class:`DemandDistribution`).
        area_size: Side length of the square region agencies/depot are
            placed in.
        inspection_cost: Inspection cost to assign to the instance.
        vehicle_capacity: Vehicle capacity to assign to the instance.
        discount_factor: Discount factor to assign to the instance.
        num_vehicles: Number of vehicles to assign to the instance.

    Returns:
        A randomly generated :class:`FoodBankInstance`.

    Raises:
        ValueError: If ``num_agencies`` or ``num_items`` is not positive.
    """
    raise NotImplementedError


def load_instance(path: str) -> FoodBankInstance:
    """Load a :class:`FoodBankInstance` from a config/data file on disk.

    Args:
        path: Path to a YAML or JSON file describing the instance (see
            ``experiments/configs/`` for the expected schema).

    Returns:
        The deserialized :class:`FoodBankInstance`.
    """
    raise NotImplementedError


def save_instance(instance: FoodBankInstance, path: str) -> None:
    """Serialize a :class:`FoodBankInstance` to disk.

    Args:
        instance: The instance to serialize.
        path: Destination path (YAML or JSON, inferred from extension).
    """
    raise NotImplementedError
