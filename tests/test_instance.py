"""Tests for foodbank_ir.instance."""

import pytest

from foodbank_ir.instance import generate_random_instance


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_generate_random_instance_shapes():
    """A 5-agency, 3-item instance has the expected dimensions and a demand entry per pair."""
    instance = generate_random_instance(num_agencies=5, num_items=3, seed=0)
    assert instance.num_agencies() == 5
    assert instance.num_items() == 3
    assert len(instance.demand) == 15
    assert 0.0 < instance.discount_factor <= 1.0


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_generate_random_instance_is_reproducible():
    """Same seed yields identical instances."""
    a = generate_random_instance(num_agencies=4, num_items=2, seed=123)
    b = generate_random_instance(num_agencies=4, num_items=2, seed=123)
    assert a == b


@pytest.mark.xfail(raises=NotImplementedError, reason="core logic not yet implemented")
def test_distance_is_symmetric_and_zero_on_diagonal():
    """Derived Euclidean distances are symmetric with zero self-distance."""
    instance = generate_random_instance(num_agencies=3, num_items=1, seed=1)
    for i in range(3):
        assert instance.distance(i, i) == 0.0
        for j in range(3):
            assert instance.distance(i, j) == pytest.approx(instance.distance(j, i))
