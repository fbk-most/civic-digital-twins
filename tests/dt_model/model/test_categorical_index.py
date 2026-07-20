"""Tests for CategoricalIndex."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.model.index import CategoricalIndex

# ===========================================================================
# Construction
# ===========================================================================


def test_construction_basic():
    """CategoricalIndex constructs with valid outcomes."""
    ci = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    assert ci.name == "mode"
    assert ci.is_abstract  # always abstract


def test_node_is_placeholder():
    """Underlying graph node is a placeholder (abstract index)."""
    ci = CategoricalIndex("mode", {"a": 0.5, "b": 0.5})
    assert isinstance(ci.node, graph.placeholder)


def test_support_returns_keys_in_order():
    """Support returns outcome keys in insertion order."""
    ci = CategoricalIndex("mode", {"bike": 0.3, "train": 0.5, "bus": 0.2})
    assert ci.support == ["bike", "train", "bus"]


def test_outcomes_returns_copy():
    """Outcomes property returns a copy (mutation does not affect internal state)."""
    ci = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    d = ci.outcomes
    d["bike"] = 0.0
    assert ci.outcomes["bike"] == pytest.approx(0.4)


def test_is_abstract():
    """CategoricalIndex is treated as abstract (value is None)."""
    ci = CategoricalIndex("mode", {"a": 1.0})
    assert ci.is_abstract


def test_repr():
    """Repr includes name and outcomes dict."""
    ci = CategoricalIndex("x", {"p": 0.6, "q": 0.4})
    r = repr(ci)
    assert "CategoricalIndex" in r
    assert "x" in r


# ===========================================================================
# Validation errors
# ===========================================================================


def test_empty_outcomes_raises():
    """Empty outcomes dict raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        CategoricalIndex("mode", {})


def test_non_positive_probability_raises():
    """Zero or negative probability raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        CategoricalIndex("mode", {"bike": 0.0, "train": 1.0})
    with pytest.raises(ValueError, match="strictly positive"):
        CategoricalIndex("mode", {"bike": -0.1, "train": 1.1})


def test_probabilities_not_summing_to_one_raises():
    """Probabilities that do not sum to 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        CategoricalIndex("mode", {"bike": 0.3, "train": 0.3})


# ===========================================================================
# Sampling
# ===========================================================================


def test_sample_returns_valid_key():
    """sample() always returns a key from support."""
    ci = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    rng = np.random.default_rng(42)
    for _ in range(50):
        key = ci.sample(rng, size=1)[0]
        assert key in ci.support


def test_sample_without_rng_returns_valid_key():
    """sample() without rng still returns a valid key."""
    ci = CategoricalIndex("mode", {"a": 0.5, "b": 0.5})
    assert ci.sample(size=1)[0] in {"a", "b"}


def test_sample_with_rng_is_reproducible():
    """Same rng seed produces same sequence of samples."""
    ci = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    samples1 = list(ci.sample(rng1, size=20))
    samples2 = list(ci.sample(rng2, size=20))
    assert samples1 == samples2


def test_sample_distribution_is_approximately_correct():
    """Sample frequency over many draws approximates the declared probabilities."""
    ci = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    rng = np.random.default_rng(1234)
    N = 10_000
    counts = {"bike": 0, "train": 0}
    for key in ci.sample(rng, size=N):
        counts[key] += 1
    assert abs(counts["bike"] / N - 0.3) < 0.02
    assert abs(counts["train"] / N - 0.7) < 0.02


# ===========================================================================
# Graph integration — CategoricalIndex as guard condition
# ===========================================================================


def test_equality_creates_graph_node():
    """CategoricalIndex.__eq__ returns a graph Node (not a bool)."""
    ci = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    result = ci == "bike"
    assert isinstance(result, graph.Node)


def test_importable_from_dt_model():
    """CategoricalIndex is importable from civic_digital_twins.dt_model."""
    from civic_digital_twins.dt_model import CategoricalIndex as CI  # noqa: PLC0415

    assert CI is CategoricalIndex


# ===========================================================================
# Hash / identity convention
# ===========================================================================


def test_categorical_index_hash_is_id_based():
    """CategoricalIndex.__hash__ == id(self): different objects don't collide in sets.

    CategoricalIndex.__eq__ returns a graph node (not bool), so the class
    declares __hash__ = id to prevent silent misbehaviour in sets and dicts.
    This test guards that invariant explicitly: two distinct objects with the
    same parameters are different set members, while the same object is always
    found in its own singleton set.
    """
    idx_a = CategoricalIndex("mode", {"bike": 0.5, "car": 0.5})
    idx_b = CategoricalIndex("mode", {"bike": 0.5, "car": 0.5})

    assert idx_a in {idx_a}, "same object must be found in its own singleton set"
    assert idx_b not in {idx_a}, "distinct objects with same params must not share set membership"
    assert not isinstance(idx_a == idx_b, bool), "__eq__ must return a graph node, not bool"
