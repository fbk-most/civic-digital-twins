# SPDX-License-Identifier: Apache-2.0
"""Tests for Scenario × Ensemble interaction covering all six override cases."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import ConstIndex, ConstTimeseriesIndex, TimeseriesIndex, define, inputs, outputs
from civic_digital_twins.dt_model.model.index import (
    CategoricalIndex,
    ConditionalCategoricalIndex,
    ConditionalDistributionIndex,
    DistributionIndex,
    GenericIndex,
    Index,
)
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.ensemble import CrossProductEnsemble, DistributionEnsemble
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@define("TestModel")
class _TestModel(Model):
    """Minimal model wrapping an arbitrary set of indexes."""

    @inputs
    class Inputs:
        indexes: list[GenericIndex]

    @outputs
    class Outputs:
        pass

    def compute(self, inputs: Inputs) -> Outputs:
        """No computation — just expose the wrapped indexes."""
        return _TestModel.Outputs()


def _make_model(*indexes: GenericIndex) -> Model:
    """Wrap *indexes* in a minimal named model."""
    return _TestModel(inputs=_TestModel.Inputs(indexes=list(indexes)))  # type: ignore[call-arg]


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
# Case 3 — distribution-backed index, no override, sampled by DistributionEnsemble
# ---------------------------------------------------------------------------


def test_abstract_no_override():
    """DistributionIndex('x', norm(0,1)) with DistributionEnsemble(size=100) yields 100 float samples."""
    x = DistributionIndex("x", stats.norm, {"loc": 0, "scale": 1})
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)
    scenario = Scenario(model)

    ens = DistributionEnsemble(scenario, size=100, rng=np.random.default_rng(42))
    ev = Evaluation(scenario).evaluate(ensemble=ens)

    arr = ev[result]
    assert arr.shape == (100,), f"Expected shape (100,), got {arr.shape}"
    assert arr.dtype.kind == "f"
    assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# Case 4 — abstract placeholder index overridden with a concrete value
# ---------------------------------------------------------------------------


def test_abstract_overridden_with_concrete():
    """Concrete override 7.0 on a placeholder Index; no ensemble sampling needed."""
    x = Index("x", None)  # bare placeholder — abstract until a value is supplied
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)
    scenario = Scenario(model, overrides={x: 7.0})

    # scenario.abstract_indexes() must be empty — x is now concrete.
    assert scenario.abstract_indexes() == [], f"Expected no abstract indexes, got {scenario.abstract_indexes()}"

    # DistributionEnsemble has nothing to sample.
    ens = DistributionEnsemble(scenario, size=10, rng=np.random.default_rng(0))
    assert dict(ens.assignments()) == {}

    # Evaluate without an ensemble — x is concrete via the scenario override.
    ev = Evaluation(scenario).evaluate(ensemble=None)
    assert float(ev[result]) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Case 5 — distribution-backed index overridden with a different distribution
# ---------------------------------------------------------------------------


def test_abstract_overridden_with_different_distribution():
    """Override distribution (norm(100, 0.001)) is used for sampling, not the model's norm(0,1)."""
    x = DistributionIndex("x", stats.norm, {"loc": 0, "scale": 1})
    result = Index("result", x.node * 1.0)
    model = _make_model(x, result)

    override_dist = stats.norm(100, 0.001)
    scenario = Scenario(model, overrides={x: override_dist})  # type: ignore[arg-type]

    # x must still be abstract (but with the override distribution).
    assert any(idx is x for idx in scenario.abstract_indexes())
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
# Case 6 — type-mismatch overrides rejected with TypeError
# ---------------------------------------------------------------------------


def test_scalar_index_rejects_distribution_override():
    """Overriding a scalar Index with a Distribution raises TypeError."""
    x = Index("x", 5.0)
    model = _make_model(x)
    with pytest.raises(TypeError, match="distribution"):
        Scenario(model, overrides={x: stats.norm(50, 1)})  # type: ignore[arg-type]


def test_distribution_index_rejects_scalar_override():
    """Overriding a DistributionIndex with a scalar raises TypeError."""
    x = DistributionIndex("x", stats.norm, {"loc": 0, "scale": 1})
    model = _make_model(x)
    with pytest.raises(TypeError, match="distribution-backed"):
        Scenario(model, overrides={x: 7.0})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Categorical index overrides — support must be preserved
# ---------------------------------------------------------------------------


def test_categorical_concrete_pin():
    """CategoricalIndex overridden with a valid str becomes concrete."""
    mode = CategoricalIndex("mode", {"electric": 0.6, "diesel": 0.4})
    model = _make_model(mode)
    scenario = Scenario(model, overrides={mode: "electric"})  # type: ignore[dict-item]

    # Use identity comparison: GenericIndex.__eq__ returns a graph node, not bool.
    assert not any(idx is mode for idx in scenario.abstract_indexes())


def test_categorical_dict_override_full_support():
    """CategoricalIndex overridden with a full-support dict stays abstract with new weights."""
    mode = CategoricalIndex("mode", {"electric": 0.6, "diesel": 0.4})
    model = _make_model(mode)
    scenario = Scenario(model, overrides={mode: {"electric": 0.9, "diesel": 0.1}})  # type: ignore[dict-item]

    assert any(idx is mode for idx in scenario.abstract_indexes())
    assert scenario.effective_outcomes(mode) == {"electric": 0.9, "diesel": 0.1}


def test_categorical_dict_override_subset():
    """CategoricalIndex overridden with a strict subset of the support restricts the domain."""
    mode = CategoricalIndex("mode", {"electric": 0.5, "diesel": 0.3, "hybrid": 0.2})
    model = _make_model(mode)
    scenario = Scenario(model, overrides={mode: {"electric": 0.7, "hybrid": 0.3}})  # type: ignore[dict-item]

    assert any(idx is mode for idx in scenario.abstract_indexes())
    assert scenario.effective_outcomes(mode) == {"electric": 0.7, "hybrid": 0.3}


def test_categorical_effective_outcomes_fallback():
    """effective_outcomes() falls back to idx.outcomes when no override is set."""
    mode = CategoricalIndex("mode", {"electric": 0.6, "diesel": 0.4})
    model = _make_model(mode)
    scenario = Scenario(model)

    assert scenario.effective_outcomes(mode) == {"electric": 0.6, "diesel": 0.4}


def test_categorical_effective_outcomes_str_pin_singleton():
    """effective_outcomes() returns a degenerate singleton when the index is str-pinned."""
    mode = CategoricalIndex("mode", {"electric": 0.6, "diesel": 0.4})
    model = _make_model(mode)
    scenario = Scenario(model, overrides={mode: "electric"})  # type: ignore[dict-item]

    assert scenario.effective_outcomes(mode) == {"electric": 1.0}


def test_categorical_rejects_extra_keys_dict():
    """Dict override with keys outside the support raises ValueError."""
    mode = CategoricalIndex("mode", {"electric": 0.6, "diesel": 0.4})
    model = _make_model(mode)
    with pytest.raises(ValueError, match="outside its support"):
        Scenario(model, overrides={mode: {"electric": 0.5, "hybrid": 0.5}})  # type: ignore[dict-item]


def test_categorical_rejects_out_of_support_str():
    """Str override outside support raises ValueError."""
    mode = CategoricalIndex("mode", {"electric": 0.6, "diesel": 0.4})
    model = _make_model(mode)
    with pytest.raises(ValueError, match="not in the support"):
        Scenario(model, overrides={mode: "hybrid"})  # type: ignore[dict-item]


def test_categorical_rejects_wrong_type():
    """Non-str, non-dict override for CategoricalIndex raises TypeError."""
    mode = CategoricalIndex("mode", {"electric": 0.6, "diesel": 0.4})
    model = _make_model(mode)
    with pytest.raises(TypeError, match="str.*dict"):
        Scenario(model, overrides={mode: 42.0})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# CategoricalIndex: list[str] override
# ---------------------------------------------------------------------------


def test_categorical_list_override_restricts_and_renormalises():
    """list[str] override restricts to subset and renormalises original probabilities."""
    # Model: good=0.3, unsettled=0.5, bad=0.2
    cat = CategoricalIndex("weather", {"good": 0.3, "unsettled": 0.5, "bad": 0.2})
    model = _make_model(cat)
    scenario = Scenario(model, overrides={cat: ["good", "unsettled"]})  # type: ignore[dict-item]
    # Renormalised: total=0.8, good=0.375, unsettled=0.625
    outcomes = scenario.effective_outcomes(cat)
    assert outcomes == pytest.approx({"good": 0.375, "unsettled": 0.625})


def test_categorical_list_override_stored_as_dict():
    """list[str] override is stored internally as a dict[str, float]."""
    cat = CategoricalIndex("weather", {"good": 0.3, "unsettled": 0.5, "bad": 0.2})
    model = _make_model(cat)
    scenario = Scenario(model, overrides={cat: ["good", "unsettled"]})  # type: ignore[dict-item]
    stored = scenario.overrides[cat]
    assert isinstance(stored, dict)
    assert set(stored.keys()) == {"good", "unsettled"}


def test_categorical_list_override_single_value():
    """list[str] with one element gives weight 1.0 for that element."""
    cat = CategoricalIndex("weather", {"good": 0.3, "unsettled": 0.5, "bad": 0.2})
    model = _make_model(cat)
    scenario = Scenario(model, overrides={cat: ["bad"]})  # type: ignore[dict-item]
    outcomes = scenario.effective_outcomes(cat)
    assert outcomes == pytest.approx({"bad": 1.0})


def test_categorical_list_override_empty_raises():
    """Empty list[str] override raises ValueError."""
    cat = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})
    model = _make_model(cat)
    with pytest.raises(ValueError, match="must not be empty"):
        Scenario(model, overrides={cat: []})  # type: ignore[dict-item]


def test_categorical_list_override_unknown_outcome_raises():
    """list[str] with unknown outcome raises ValueError."""
    cat = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})
    model = _make_model(cat)
    with pytest.raises(ValueError, match="outside its support"):
        Scenario(model, overrides={cat: ["good", "unknown"]})  # type: ignore[dict-item]


def test_categorical_list_override_in_cross_product_ensemble():
    """CrossProductEnsemble respects list[str] override from Scenario."""
    cat = CategoricalIndex("weather", {"good": 0.3, "unsettled": 0.5, "bad": 0.2})
    model = _make_model(cat)
    scenario = Scenario(model, overrides={cat: ["good", "unsettled"]})  # type: ignore[dict-item]
    ens = CrossProductEnsemble(scenario)
    assert ens.size == 2
    assigned = set(ens.assignments()[cat].tolist())
    assert assigned == {"good", "unsettled"}
    weights = ens.ensemble_weights[0]
    weight_by_value = {str(v): float(w) for v, w in zip(ens.assignments()[cat], weights)}
    assert weight_by_value == pytest.approx({"good": 0.375, "unsettled": 0.625})


def test_conditional_categorical_concrete_pin():
    """ConditionalCategoricalIndex overridden with a valid str becomes concrete."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    weather = ConditionalCategoricalIndex(
        "weather",
        parents=[season],
        support=["good", "bad"],
        factory=lambda season: {"good": 0.8, "bad": 0.2} if season == "summer" else {"good": 0.3, "bad": 0.7},
    )
    model = _make_model(season, weather)
    scenario = Scenario(model, overrides={weather: "good"})  # type: ignore[dict-item]

    # Use identity comparison: GenericIndex.__eq__ returns a graph node, not bool.
    assert not any(idx is weather for idx in scenario.abstract_indexes())
    # effective_outcomes returns the singleton — consistent with CategoricalIndex, str.
    assert scenario.effective_outcomes(weather) == {"good": 1.0}


def test_conditional_categorical_effective_outcomes_no_override():
    """effective_outcomes() returns None for ConditionalCategoricalIndex with no override."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    weather = ConditionalCategoricalIndex(
        "weather",
        parents=[season],
        support=["good", "bad"],
        factory=lambda season: {"good": 0.8, "bad": 0.2} if season == "summer" else {"good": 0.3, "bad": 0.7},
    )
    model = _make_model(season, weather)
    scenario = Scenario(model)

    assert scenario.effective_outcomes(weather) is None


def test_conditional_categorical_rejects_dict_override():
    """ConditionalCategoricalIndex does not accept dict overrides (only str)."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    weather = ConditionalCategoricalIndex(
        "weather",
        parents=[season],
        support=["good", "bad"],
        factory=lambda season: {"good": 0.8, "bad": 0.2} if season == "summer" else {"good": 0.3, "bad": 0.7},
    )
    model = _make_model(season, weather)
    with pytest.raises(TypeError, match="str"):
        Scenario(model, overrides={weather: {"good": 0.5, "bad": 0.5}})  # type: ignore[dict-item]


def test_conditional_distribution_index_rejects_all_overrides():
    """ConditionalDistributionIndex does not support any overrides."""
    from scipy import stats as scipy_stats

    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    temp = ConditionalDistributionIndex(
        "temperature",
        parents=[season],
        factory=lambda season: (
            scipy_stats.norm(loc=25, scale=3) if season == "summer" else scipy_stats.norm(loc=5, scale=5)
        ),
    )
    model = _make_model(season, temp)
    with pytest.raises(TypeError, match="not support"):
        Scenario(model, overrides={temp: scipy_stats.norm(loc=20, scale=2)})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Scenario validation — structural constants cannot be overridden
# ---------------------------------------------------------------------------


def test_const_index_cannot_be_overridden():
    """Scenario rejects overrides of ConstIndex with TypeError."""
    c = ConstIndex("c", 5.0)
    model = _make_model(c)
    with pytest.raises(TypeError, match="structural constant"):
        Scenario(model, overrides={c: 10.0})  # type: ignore[dict-item]


def test_const_timeseries_index_cannot_be_overridden():
    """Scenario rejects overrides of ConstTimeseriesIndex with TypeError."""
    c = ConstTimeseriesIndex("c", np.array([1.0, 2.0]))
    model = _make_model(c)
    with pytest.raises(TypeError, match="structural constant"):
        Scenario(model, overrides={c: np.array([3.0, 4.0])})  # type: ignore[dict-item]


def test_timeseries_index_override_must_be_1d_ndarray():
    """Scenario rejects non-1-D ndarray override for TimeseriesIndex."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0]))
    model = _make_model(ts)
    with pytest.raises(TypeError, match="1-D ndarray"):
        Scenario(model, overrides={ts: np.array([[1.0, 2.0]])})  # type: ignore[dict-item]


def test_timeseries_index_override_must_be_ndarray_not_scalar():
    """Scenario rejects a plain float override for TimeseriesIndex."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0]))
    model = _make_model(ts)
    with pytest.raises(TypeError, match="1-D ndarray"):
        Scenario(model, overrides={ts: 3.0})  # type: ignore[dict-item]


def test_timeseries_index_valid_override_accepted():
    """Scenario accepts a valid 1-D ndarray override for TimeseriesIndex."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0]))
    model = _make_model(ts)
    override = np.array([3.0, 4.0])
    scenario = Scenario(model, overrides={ts: override})
    subs = scenario.base_substitutions()
    assert ts.node in subs
    assert np.array_equal(subs[ts.node], override)


def test_categorical_index_override_dict_must_not_be_empty():
    """Scenario rejects an empty dict override for CategoricalIndex."""
    cat = CategoricalIndex("cat", {"a": 0.5, "b": 0.5})
    model = _make_model(cat)
    with pytest.raises(ValueError, match="must not be empty"):
        Scenario(model, overrides={cat: {}})  # type: ignore[dict-item]


def test_categorical_index_override_dict_non_positive_prob():
    """Scenario rejects a dict override with a non-positive probability."""
    cat = CategoricalIndex("cat", {"a": 0.5, "b": 0.5})
    model = _make_model(cat)
    with pytest.raises(ValueError, match="strictly positive"):
        Scenario(model, overrides={cat: {"a": 0.0, "b": 1.0}})  # type: ignore[dict-item]


def test_categorical_index_override_dict_probs_not_sum_to_one():
    """Scenario rejects a dict override whose probabilities don't sum to 1."""
    cat = CategoricalIndex("cat", {"a": 0.5, "b": 0.5})
    model = _make_model(cat)
    with pytest.raises(ValueError, match="sum to 1"):
        Scenario(model, overrides={cat: {"a": 0.4, "b": 0.4}})  # type: ignore[dict-item]


def test_conditional_categorical_override_key_not_in_support():
    """Scenario rejects a str override not in the ConditionalCategoricalIndex support."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    weather = ConditionalCategoricalIndex(
        "weather",
        parents=[season],
        support=["good", "bad"],
        factory=lambda _season: {"good": 0.8, "bad": 0.2},
    )
    model = _make_model(season, weather)
    with pytest.raises(ValueError, match="not in the support"):
        Scenario(model, overrides={weather: "rainy"})


def test_scalar_index_override_must_be_scalar():
    """Scenario rejects an ndarray override for a scalar Index."""
    x = Index("x", 1.0)
    model = _make_model(x)
    with pytest.raises(TypeError, match="scalar"):
        Scenario(model, overrides={x: np.array([1.0, 2.0])})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# Evaluation validation — uncovered abstract indexes and overlap
# ---------------------------------------------------------------------------


def test_evaluate_raises_on_uncovered_abstract_index():
    """Evaluation.evaluate raises ValueError when an abstract index has no value."""
    x = Index("x", None)
    result = Index("result", x.node * 2.0)
    model = _make_model(x, result)
    scenario = Scenario(model)
    with pytest.raises(ValueError, match="abstract indexes are not covered"):
        Evaluation(scenario).evaluate(ensemble=None)


def test_evaluate_raises_on_parameters_and_overrides_overlap():
    """Evaluation.evaluate raises ValueError when parameters= and Scenario.overrides share an index."""
    x = Index("x", 1.0)
    result = Index("result", x.node * 2.0)
    model = _make_model(x, result)
    scenario = Scenario(model, overrides={x: 5.0})
    with pytest.raises(ValueError, match="both parameters= and Scenario.overrides"):
        Evaluation(scenario).evaluate(ensemble=None, parameters={x: np.array([1.0, 2.0])})


# ---------------------------------------------------------------------------
# DistributionEnsemble — Scenario path and wrong-type guard
# ---------------------------------------------------------------------------


def test_distribution_ensemble_accepts_scenario():
    """DistributionEnsemble can be constructed directly from a Scenario."""
    x = DistributionIndex("x", stats.norm, {"loc": 0.0, "scale": 1.0})
    model = _make_model(x)
    scenario = Scenario(model)
    ens = DistributionEnsemble(scenario, 5, rng=np.random.default_rng(0))
    assert len(ens.ensemble_weights) == 1
    assert ens.ensemble_weights[0].shape == (5,)


def test_distribution_ensemble_rejects_wrong_type():
    """DistributionEnsemble raises TypeError when passed something other than a Scenario."""
    with pytest.raises(TypeError, match="expects a Scenario"):
        DistributionEnsemble("not a model", 5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CrossProductEnsemble — wrong-type guard
# ---------------------------------------------------------------------------


def test_cross_product_ensemble_rejects_wrong_type():
    """CrossProductEnsemble raises TypeError when passed something other than a Scenario."""
    with pytest.raises(TypeError, match="expects a Scenario"):
        CrossProductEnsemble("not a model")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CrossProductEnsemble - scenario dict override weights respected
# ---------------------------------------------------------------------------


def test_cross_product_ensemble_honors_dict_override_weights():
    """CrossProductEnsemble uses scenario dict override probabilities, not model probs.

    Regression for the bug where cat.outcomes was always used, silently
    discarding any dict[str, float] override probability set on the scenario.
    With max_categorical_size >= support size the ensemble enumerates exactly,
    so weights are deterministic.
    """
    # Model declares heavily skewed probs: a=0.9, b=0.1.
    cat = CategoricalIndex("cat", {"a": 0.9, "b": 0.1})
    model = _make_model(cat)

    # Override to equal weights - the ensemble must reflect these, not the model probs.
    scenario = Scenario(model, overrides={cat: {"a": 0.5, "b": 0.5}})  # type: ignore[dict-item]
    ens = CrossProductEnsemble(scenario, max_categorical_size=20)

    weights = ens.ensemble_weights[0]  # shape (2,) - one entry per outcome
    assignments = ens.assignments()[cat]  # shape (2,)

    # Build a mapping outcome -> weight for order-independent comparison.
    weight_by_value = {str(v): float(w) for v, w in zip(assignments, weights)}
    assert weight_by_value == pytest.approx({"a": 0.5, "b": 0.5})


def test_cross_product_ensemble_dict_override_restricts_support():
    """CrossProductEnsemble auto-restricts to the dict override's keys.

    When the scenario dict override covers only a strict subset of the model's
    support, the ensemble must sample only those outcomes and must not include
    any value absent from the override.
    """
    # Model has three outcomes: a, b, c.
    cat = CategoricalIndex("cat", {"a": 0.5, "b": 0.3, "c": 0.2})
    model = _make_model(cat)

    # Override restricts to {a, b} with custom weights.
    scenario = Scenario(model, overrides={cat: {"a": 0.7, "b": 0.3}})  # type: ignore[dict-item]
    ens = CrossProductEnsemble(scenario, max_categorical_size=20)

    assignments = ens.assignments()[cat]
    sampled_values = {str(v) for v in assignments}
    assert sampled_values == {"a", "b"}, f"unexpected values in ensemble: {sampled_values}"
    assert ens.size == 2  # exactly the two restricted outcomes

    weights = ens.ensemble_weights[0]
    weight_by_value = {str(v): float(w) for v, w in zip(assignments, weights)}
    assert weight_by_value == pytest.approx({"a": 0.7, "b": 0.3})


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_cross_product_ensemble_explicit_restriction_overrides_dict_keys():
    """An explicit restrictions= entry takes precedence over the dict override's key set.

    The restriction controls which values are sampled; probabilities still come
    from the scenario dict override (not the model).
    """
    cat = CategoricalIndex("cat", {"a": 0.5, "b": 0.3, "c": 0.2})
    model = _make_model(cat)
    # Scenario override covers {a, b, c} with custom probs.
    scenario = Scenario(model, overrides={cat: {"a": 0.6, "b": 0.3, "c": 0.1}})  # type: ignore[dict-item]

    # Explicit restriction further limits to just {a, b}.
    ens = CrossProductEnsemble(scenario, restrictions={cat: ["a", "b"]}, max_categorical_size=20)

    assignments = ens.assignments()[cat]
    sampled_values = {str(v) for v in assignments}
    assert sampled_values == {"a", "b"}

    # Weights must come from the override dict, renormalised over {a, b}: 0.6/0.9, 0.3/0.9.
    weights = ens.ensemble_weights[0]
    weight_by_value = {str(v): float(w) for v, w in zip(assignments, weights)}
    assert weight_by_value == pytest.approx({"a": 0.6 / 0.9, "b": 0.3 / 0.9})


# ---------------------------------------------------------------------------
# Foreign-index checks — override keys not in model.indexes
# ---------------------------------------------------------------------------


def test_scenario_override_foreign_index_raises():
    """Scenario raises ValueError when an override key is not in the model's indexes."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)  # NOT added to the model
    model = _make_model(a)
    with pytest.raises(ValueError, match="not part of this model"):
        Scenario(model, overrides={b: 3.0})


def test_scenario_override_foreign_index_error_names_the_culprit():
    """Error message includes the name of the foreign index."""
    a = Index("a", 1.0)
    foreign = Index("foreign_idx", 99.0)  # NOT in the model
    model = _make_model(a)
    with pytest.raises(ValueError, match="foreign_idx"):
        Scenario(model, overrides={foreign: 42.0})


def test_scenario_override_index_in_model_does_not_raise():
    """Scenario does not raise when all override keys belong to the model."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    model = _make_model(a, b)
    # Both a and b are in the model — no error expected.
    scenario = Scenario(model, overrides={a: 10.0, b: 20.0})
    assert scenario.overrides == {a: 10.0, b: 20.0}


# ---------------------------------------------------------------------------
# CrossProductEnsemble deprecation warnings
# ---------------------------------------------------------------------------


def test_cross_product_ensemble_restrictions_deprecated():
    """CrossProductEnsemble.restrictions= emits DeprecationWarning."""
    cat = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})
    model = _make_model(cat)
    with pytest.warns(DeprecationWarning, match="restrictions="):
        CrossProductEnsemble(Scenario(model), restrictions={cat: ["good"]})


def test_cross_product_ensemble_exclude_deprecated():
    """CrossProductEnsemble.exclude= emits DeprecationWarning."""
    cat = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})
    pv = ConditionalDistributionIndex(
        "presence",
        parents=[cat],
        factory=lambda weather: stats.norm(loc=100.0, scale=10.0),
    )
    model = _make_model(cat, pv)
    with pytest.warns(DeprecationWarning, match="exclude="):
        CrossProductEnsemble(Scenario(model), exclude=[pv])


# ---------------------------------------------------------------------------
# CrossProductEnsemble: Scenario.parameter_axes auto-exclusion
# ---------------------------------------------------------------------------


def test_cross_product_ensemble_skips_parameter_axes_index():
    """CrossProductEnsemble automatically skips indexes listed in Scenario.parameter_axes."""
    import warnings  # noqa: PLC0415

    cat = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})
    pv = ConditionalDistributionIndex(
        "presence",
        parents=[cat],
        factory=lambda weather: stats.norm(loc=100.0, scale=10.0),
    )
    model = _make_model(cat, pv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ens = CrossProductEnsemble(Scenario(model, parameter_axes=[pv]))
    assert pv not in ens.assignments()
    assert cat in ens.assignments()


def test_cross_product_ensemble_parameter_axes_no_warning():
    """CrossProductEnsemble does not emit DeprecationWarning for Scenario.parameter_axes indexes."""
    import warnings  # noqa: PLC0415

    cat = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})
    pv = ConditionalDistributionIndex(
        "presence",
        parents=[cat],
        factory=lambda weather: stats.norm(loc=100.0, scale=10.0),
    )
    model = _make_model(cat, pv)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CrossProductEnsemble(Scenario(model, parameter_axes=[pv]))
    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 0, f"Unexpected DeprecationWarning: {deprecation_warnings}"


# ---------------------------------------------------------------------------
# Scenario.parameter_axes validation
# ---------------------------------------------------------------------------


def test_scenario_parameter_axes_foreign_index_raises():
    """Scenario rejects a parameter_axes entry from a different model."""
    cat = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})
    pv = ConditionalDistributionIndex("presence", parents=[cat], factory=lambda weather: stats.norm())
    _make_model(cat, pv)  # pv belongs to this model only
    model_b = _make_model(cat)  # pv not in this model
    with pytest.raises(ValueError, match="not part of this model"):
        Scenario(model_b, parameter_axes=[pv])


def test_evaluation_raises_when_parameter_axes_not_in_parameters():
    """Evaluation.evaluate() raises ValueError when scenario.parameter_axes are absent from parameters=."""
    cat = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})
    pv = ConditionalDistributionIndex(
        "presence", parents=[cat], factory=lambda weather: stats.norm(loc=100.0, scale=10.0)
    )
    model = _make_model(cat, pv)
    scenario = Scenario(model, parameter_axes=[pv])
    ens = CrossProductEnsemble(scenario)  # pv excluded from ensemble
    with pytest.raises(ValueError, match="parameter_axes"):
        Evaluation(scenario).evaluate(ensemble=ens)  # pv not in parameters=
