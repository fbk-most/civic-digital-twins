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
    assert ci.value is None  # always abstract


def test_node_is_placeholder():
    """Underlying graph node is a placeholder (abstract index)."""
    ci = CategoricalIndex("mode", {"a": 0.5, "b": 0.5})
    assert isinstance(ci.node, graph.placeholder)


def test_support_returns_keys_in_order():
    """support returns outcome keys in insertion order."""
    ci = CategoricalIndex("mode", {"bike": 0.3, "train": 0.5, "bus": 0.2})
    assert ci.support == ["bike", "train", "bus"]


def test_outcomes_returns_copy():
    """outcomes property returns a copy (mutation does not affect internal state)."""
    ci = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    d = ci.outcomes
    d["bike"] = 0.0
    assert ci.outcomes["bike"] == pytest.approx(0.4)


def test_is_abstract():
    """CategoricalIndex is treated as abstract (value is None)."""
    ci = CategoricalIndex("mode", {"a": 1.0})
    assert ci.value is None


def test_repr():
    """repr includes name and outcomes dict."""
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
        key = ci.sample(rng)
        assert key in ci.support


def test_sample_without_rng_returns_valid_key():
    """sample() without rng still returns a valid key."""
    ci = CategoricalIndex("mode", {"a": 0.5, "b": 0.5})
    assert ci.sample() in {"a", "b"}


def test_sample_with_rng_is_reproducible():
    """Same rng seed produces same sequence of samples."""
    ci = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    samples1 = [ci.sample(rng1) for _ in range(20)]
    samples2 = [ci.sample(rng2) for _ in range(20)]
    assert samples1 == samples2


def test_sample_distribution_is_approximately_correct():
    """Sample frequency over many draws approximates the declared probabilities."""
    ci = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    rng = np.random.default_rng(1234)
    N = 10_000
    counts = {"bike": 0, "train": 0}
    for _ in range(N):
        counts[ci.sample(rng)] += 1
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
