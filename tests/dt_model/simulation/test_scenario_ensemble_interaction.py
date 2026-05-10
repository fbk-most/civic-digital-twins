# SPDX-License-Identifier: Apache-2.0
"""Tests for Scenario × Ensemble interaction covering all six override cases."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model.model.index import GenericIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(*indexes: GenericIndex) -> Model:
    """Wrap *indexes* in a minimal named model."""
    return Model("test", list(indexes))


# ---------------------------------------------------------------------------
# Case 1 — concrete index, no override (baseline)
# ---------------------------------------------------------------------------


def test_concrete_no_override():
    """Index('x', 5.0) with no scenario override evaluates to 5.0."""
    x = Index("x", 5.0)
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)
    scenario = Scenario(model)

    ev = Evaluation(scenario).evaluate(ensemble=None)
    assert float(ev[result]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Case 2 — concrete index overridden with a concrete value (regression guard)
# ---------------------------------------------------------------------------


def test_concrete_overridden_with_concrete():
    """Scenario override of 12.0 wins over model value of 5.0."""
    x = Index("x", 5.0)
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)
    scenario = Scenario(model, overrides={x: 12.0})

    ev = Evaluation(scenario).evaluate(ensemble=None)
    assert float(ev[result]) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Case 3 — abstract index, no override, sampled by DistributionEnsemble
# ---------------------------------------------------------------------------


def test_abstract_no_override():
    """Index('x', norm(0,1)) with DistributionEnsemble(size=100) yields 100 float samples."""
    x = Index("x", stats.norm(0, 1))
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)
    scenario = Scenario(model)

    ens = DistributionEnsemble(scenario, size=100, rng=np.random.default_rng(42))
    ev = Evaluation(scenario).evaluate(ensemble=ens)

    arr = ev[result]
    assert arr.shape == (100,), f"Expected shape (100,), got {arr.shape}"
    # All samples should be finite floats, not the distribution object
    assert arr.dtype.kind == "f"
    assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# Case 4 — abstract index overridden with a concrete value
# ---------------------------------------------------------------------------


def test_abstract_overridden_with_concrete():
    """Concrete override 7.0 silences the distribution; no ensemble sampling needed."""
    x = Index("x", stats.norm(0, 1))
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)
    scenario = Scenario(model, overrides={x: 7.0})

    # scenario.abstract_indexes() must be empty — x is now concrete.
    assert scenario.abstract_indexes() == [], f"Expected no abstract indexes, got {scenario.abstract_indexes()}"

    # DistributionEnsemble should not raise (it just has nothing to sample).
    ens = DistributionEnsemble(scenario, size=10, rng=np.random.default_rng(0))

    # assignments() returns an empty dict because there's nothing to sample.
    assert dict(ens.assignments()) == {}

    # Evaluate without an ensemble — x is concrete via the scenario override.
    ev = Evaluation(scenario).evaluate(ensemble=None)
    assert float(ev[result]) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Case 5 — abstract index overridden with a different distribution
# ---------------------------------------------------------------------------


def test_abstract_overridden_with_different_distribution():
    """Override distribution (norm(100, 0.001)) is used for sampling, not model's norm(0,1)."""
    x = Index("x", stats.norm(0, 1))
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)

    override_dist = stats.norm(100, 0.001)
    scenario = Scenario(model, overrides={x: override_dist})

    # x must still be abstract (but with the override distribution).
    assert x in scenario.abstract_indexes()
    assert scenario.effective_distribution(x) is override_dist

    ens = DistributionEnsemble(scenario, size=200, rng=np.random.default_rng(0))
    ev = Evaluation(scenario).evaluate(ensemble=ens)

    arr = ev[result]
    assert arr.shape == (200,)
    # Mean should be close to 100 (the override dist centre), not 0.
    assert np.mean(arr) == pytest.approx(100.0, abs=0.1), (
        f"Mean {np.mean(arr)} is not close to 100 — model's distribution was used instead of override"
    )


# ---------------------------------------------------------------------------
# Case 6 — concrete index promoted to abstract by a Distribution override
# ---------------------------------------------------------------------------


def test_concrete_promoted_to_abstract():
    """Concrete Index('x', 5.0) promoted to abstract by a Distribution override in scenario."""
    x = Index("x", 5.0)
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)

    promo_dist = stats.norm(50, 0.001)
    scenario = Scenario(model, overrides={x: promo_dist})

    # x must appear as abstract in the scenario even though the model treats it as concrete.
    assert x in scenario.abstract_indexes(), (
        "x should be scenario-abstract after being promoted by a Distribution override"
    )

    ens = DistributionEnsemble(scenario, size=200, rng=np.random.default_rng(0))
    ev = Evaluation(scenario).evaluate(ensemble=ens)

    arr = ev[result]
    assert arr.shape == (200,)
    # Mean should be close to 50 (the promoted distribution centre), not 5.
    assert np.mean(arr) == pytest.approx(50.0, abs=0.1), (
        f"Mean {np.mean(arr)} is not close to 50 — concrete value 5.0 was used instead of the promoted distribution"
    )
