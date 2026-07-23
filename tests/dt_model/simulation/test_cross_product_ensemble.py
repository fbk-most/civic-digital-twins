"""Tests for CrossProductEnsemble."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    ConditionalCategoricalIndex,
    ConditionalDistributionIndex,
    CrossProductEnsemble,
    DistributionIndex,
    GenericIndex,
    Index,
    Model,
    define,
    inputs,
    outputs,
    sample_across,
)
from civic_digital_twins.dt_model.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# Minimal model helpers
# ---------------------------------------------------------------------------


@define("M")
class _M(Model):
    """Minimal model wrapping an arbitrary set of indexes."""

    @inputs
    class Inputs:
        indexes: list[GenericIndex]

    @outputs
    class Outputs:
        pass

    def compute(self, inputs: Inputs) -> Outputs:
        """No computation — just expose the wrapped indexes."""
        return _M.Outputs()


def _simple_model(*abstract_indexes: GenericIndex) -> Model:
    """Return a minimal Model wrapping the given abstract indexes."""
    return _M(inputs=_M.Inputs(indexes=list(abstract_indexes)))


# ---------------------------------------------------------------------------
# Basic construction — unconditional CategoricalIndex
# ---------------------------------------------------------------------------


def test_cpe_single_categorical_enumerated():
    """Single 2-outcome CategoricalIndex is fully enumerated; weights sum to 1."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    model = _simple_model(season)
    ens = CrossProductEnsemble(Scenario(model))
    assert ens.size == 2
    weights = ens.ensemble_weights[0]
    assert pytest.approx(weights.sum()) == 1.0
    assert set(ens.assignments()[season].tolist()) == {"summer", "winter"}


def test_cpe_two_categoricals_cross_product():
    """Two categoricals produce size = product of support sizes."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    weather = CategoricalIndex("weather", {"good": 0.7, "bad": 0.3})
    model = _simple_model(season, weather)
    ens = CrossProductEnsemble(Scenario(model))
    assert ens.size == 4
    assert pytest.approx(ens.ensemble_weights[0].sum()) == 1.0


def test_cpe_weights_match_joint_probability():
    """Weights for enumerated cross-product equal the joint probability."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    weather = CategoricalIndex("weather", {"good": 0.7, "bad": 0.3})
    model = _simple_model(season, weather)
    ens = CrossProductEnsemble(Scenario(model))
    a = ens.assignments()
    weights = ens.ensemble_weights[0]
    # Find the (summer, good) scenario.
    for i in range(ens.size):
        if a[season][i] == "summer" and a[weather][i] == "good":
            assert pytest.approx(weights[i], rel=1e-6) == 0.6 * 0.7


def test_cpe_no_abstract_indexes():
    """Model with no enumerable/sampleable abstract indexes produces size=1."""
    model = _simple_model()  # no abstract indexes
    ens = CrossProductEnsemble(Scenario(model))
    assert ens.size == 1
    assert pytest.approx(ens.ensemble_weights[0].sum()) == 1.0


def test_cpe_len():
    """__len__ returns the number of scenarios."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    model = _simple_model(season)
    ens = CrossProductEnsemble(Scenario(model))
    assert len(ens) == ens.size


# ---------------------------------------------------------------------------
# MC sampling (max_categorical_size < support size)
# ---------------------------------------------------------------------------


def test_cpe_mc_sampling_when_support_exceeds_size():
    """When support > max_categorical_size, MC sampling is used; size = max_categorical_size."""
    season = CategoricalIndex("season", {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2})
    model = _simple_model(season)
    rng = np.random.default_rng(0)
    ens = CrossProductEnsemble(Scenario(model), max_categorical_size=3, rng=rng)
    assert ens.size == 3
    assert pytest.approx(ens.ensemble_weights[0].sum()) == 1.0


# ---------------------------------------------------------------------------
# Distribution-backed indexes
# ---------------------------------------------------------------------------


def test_cpe_distribution_index_sampled():
    """DistributionIndex is sampled and present in assignments."""
    cap = DistributionIndex("cap", stats.uniform, {"loc": 90.0, "scale": 20.0})
    model = _simple_model(cap)
    ens = CrossProductEnsemble(Scenario(model), rng=np.random.default_rng(42))
    a = ens.assignments()
    assert cap in a
    assert len(a[cap]) == ens.size
    assert np.all((a[cap] >= 90.0) & (a[cap] <= 110.0))


def test_cpe_mixed_categorical_and_distribution():
    """CategoricalIndex cross-product combined with DistributionIndex sampling."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    cap = DistributionIndex("cap", stats.uniform, {"loc": 100.0, "scale": 50.0})
    model = _simple_model(season, cap)
    ens = CrossProductEnsemble(Scenario(model), rng=np.random.default_rng(1))
    assert ens.size == 2  # 2 seasons
    a = ens.assignments()
    assert season in a
    assert cap in a
    assert len(a[cap]) == 2


# ---------------------------------------------------------------------------
# ConditionalCategoricalIndex
# ---------------------------------------------------------------------------


def test_cpe_conditional_categorical_enumerated():
    """ConditionalCategoricalIndex is enumerated per parent configuration."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})

    def weather_factory(season: str) -> dict[str, float]:
        """Weather probabilities depend on season."""
        if season == "summer":
            return {"good": 0.8, "bad": 0.2}
        return {"good": 0.3, "bad": 0.7}

    weather = ConditionalCategoricalIndex("weather", parents=[season], support=["good", "bad"], factory=weather_factory)
    model = _simple_model(season, weather)
    ens = CrossProductEnsemble(Scenario(model))
    # 2 seasons × 2 weather outcomes = 4 combos.
    assert ens.size == 4
    assert pytest.approx(ens.ensemble_weights[0].sum()) == 1.0


def test_cpe_conditional_categorical_weights():
    """Joint weights for ConditionalCategoricalIndex match conditional probability product."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    weather = ConditionalCategoricalIndex(
        "weather",
        parents=[season],
        support=["good", "bad"],
        factory=lambda season: {"good": 0.8, "bad": 0.2} if season == "summer" else {"good": 0.3, "bad": 0.7},
    )
    model = _simple_model(season, weather)
    ens = CrossProductEnsemble(Scenario(model))
    a = ens.assignments()
    weights = ens.ensemble_weights[0]
    # (summer, good): P = 0.5 × 0.8 = 0.4
    for i in range(ens.size):
        if a[season][i] == "summer" and a[weather][i] == "good":
            assert pytest.approx(weights[i], rel=1e-6) == 0.5 * 0.8


# ---------------------------------------------------------------------------
# ConditionalDistributionIndex
# ---------------------------------------------------------------------------


def test_cpe_conditional_distribution_sampled_per_categorical():
    """ConditionalDistributionIndex is sampled with the correct parent config per scenario."""
    weather = CategoricalIndex("weather", {"hot": 0.5, "cold": 0.5})
    temp = ConditionalDistributionIndex(
        "temp",
        parents=[weather],
        factory=lambda weather: stats.norm(loc=30.0, scale=1.0) if weather == "hot" else stats.norm(loc=5.0, scale=1.0),
    )
    model = _simple_model(weather, temp)
    rng = np.random.default_rng(99)
    ens = CrossProductEnsemble(Scenario(model), rng=rng)
    a = ens.assignments()
    assert ens.size == 2
    assert temp in a
    for i in range(ens.size):
        w = a[weather][i]
        t = float(a[temp][i])
        if w == "hot":
            assert 25.0 < t < 35.0, f"Expected hot temp near 30, got {t}"
        else:
            assert 0.0 < t < 10.0, f"Expected cold temp near 5, got {t}"


def test_cpe_conditional_dist_with_distribution_parent():
    """ConditionalDistributionIndex whose parent is a DistributionIndex.

    Exercises _topo_sort_dists recursive visit (line 456), the already-visited
    early return (line 451), and the non-categorical parent lookup (line 609).
    """
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    base = DistributionIndex("base", stats.uniform, {"loc": 1.0, "scale": 2.0})
    derived = ConditionalDistributionIndex(
        "derived",
        parents=[season, base],
        factory=lambda **kw: stats.norm(loc=float(kw["base"]), scale=0.1),
    )
    model = _simple_model(season, base, derived)
    rng = np.random.default_rng(0)
    ens = CrossProductEnsemble(Scenario(model), rng=rng)
    assert ens.size == 2  # 2 seasons
    a = ens.assignments()
    assert base in a
    assert derived in a
    assert a[derived].shape == (2,)
    assert np.all(np.isfinite(a[derived]))


# ---------------------------------------------------------------------------
# PresenceVariable / plain placeholder Index excluded
# ---------------------------------------------------------------------------


def test_cpe_plain_placeholder_index_excluded():
    """Plain abstract Index (no distribution, not categorical) is excluded from ensemble."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    pv = Index("presence", None)  # PresenceVariable-like: abstract placeholder
    model = _simple_model(season, pv)
    ens = CrossProductEnsemble(Scenario(model))
    # season is enumerated; pv is skipped — assignments has only season.
    a = ens.assignments()
    assert season in a
    assert pv not in a


# ---------------------------------------------------------------------------
# AxisEnsemble protocol
# ---------------------------------------------------------------------------


def test_cpe_implements_axis_ensemble_protocol():
    """CrossProductEnsemble satisfies the AxisEnsemble protocol."""
    from civic_digital_twins.dt_model import AxisEnsemble  # noqa: PLC0415

    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    model = _simple_model(season)
    ens = CrossProductEnsemble(Scenario(model))
    assert isinstance(ens, AxisEnsemble)


def test_cpe_single_ensemble_axis():
    """CrossProductEnsemble reports exactly one ENSEMBLE axis."""
    from civic_digital_twins.dt_model.axes import ENSEMBLE  # noqa: PLC0415

    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    model = _simple_model(season)
    ens = CrossProductEnsemble(Scenario(model))
    axes = ens.ensemble_axes
    assert len(axes) == 1
    assert axes[0].role == ENSEMBLE


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_cpe_reproducible_with_rng():
    """Same rng seed produces identical assignments."""
    season = CategoricalIndex("season", {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2})
    cap = DistributionIndex("cap", stats.norm, {"loc": 100.0, "scale": 10.0})
    model = _simple_model(season, cap)
    scenario = Scenario(model)
    a1 = CrossProductEnsemble(scenario, max_categorical_size=3, rng=np.random.default_rng(7)).assignments()
    a2 = CrossProductEnsemble(scenario, max_categorical_size=3, rng=np.random.default_rng(7)).assignments()
    np.testing.assert_array_equal(a1[season], a2[season])
    np.testing.assert_array_equal(a1[cap], a2[cap])


# ---------------------------------------------------------------------------
# sample_across
# ---------------------------------------------------------------------------


def test_sample_across_basic():
    """sample_across returns approximately total samples per index."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    temp = ConditionalDistributionIndex(
        "temp",
        parents=[season],
        factory=lambda season: (
            stats.norm(loc=30.0, scale=1.0) if season == "summer" else stats.norm(loc=5.0, scale=1.0)
        ),
    )
    model = _simple_model(season, temp)  # temp is skipped by CrossProductEnsemble (CDI)
    ens = CrossProductEnsemble(Scenario(model))
    samples = sample_across(ens, [temp], total=100, rng=np.random.default_rng(0))
    assert temp in samples
    # Approximately 100 samples (may be 100 or 102 due to rounding).
    assert 90 <= len(samples[temp]) <= 110


def test_sample_across_respects_weights():
    """Samples are drawn proportionally to scenario weights."""
    season = CategoricalIndex("season", {"hot": 0.8, "cold": 0.2})
    temp = ConditionalDistributionIndex(
        "temp",
        parents=[season],
        factory=lambda season: stats.norm(loc=35.0, scale=0.5) if season == "hot" else stats.norm(loc=0.0, scale=0.5),
    )
    model = _simple_model(season, temp)
    ens = CrossProductEnsemble(Scenario(model))
    samples = sample_across(ens, [temp], total=1000, rng=np.random.default_rng(42))
    arr = samples[temp]
    # ~80% of samples should be from the hot distribution (mean 35), ~20% from cold (mean 0).
    # Weighted mean ≈ 0.8×35 + 0.2×0 = 28.
    assert 25.0 < float(arr.mean()) < 31.0


def test_sample_across_reproducible():
    """Same rng seed produces identical samples."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    temp = ConditionalDistributionIndex(
        "temp",
        parents=[season],
        factory=lambda season: (
            stats.norm(loc=30.0, scale=1.0) if season == "summer" else stats.norm(loc=5.0, scale=1.0)
        ),
    )
    model = _simple_model(season, temp)
    ens = CrossProductEnsemble(Scenario(model), rng=np.random.default_rng(1))
    s1 = sample_across(ens, [temp], total=50, rng=np.random.default_rng(99))
    s2 = sample_across(ens, [temp], total=50, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(s1[temp], s2[temp])


def test_sample_across_missing_parent_raises():
    """Raises ValueError when a parent index is not in the ensemble."""
    orphan_season = CategoricalIndex("orphan_season", {"a": 0.5, "b": 0.5})
    temp = ConditionalDistributionIndex(
        "temp",
        parents=[orphan_season],
        factory=lambda **_kw: stats.norm(loc=10.0, scale=1.0),
    )
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    model = _simple_model(season)  # orphan_season not in model
    ens = CrossProductEnsemble(Scenario(model))
    with pytest.raises(ValueError, match="not present in the ensemble"):
        sample_across(ens, [temp])


def test_sample_across_multi_axis_raises():
    """Raises ValueError for multi-axis ensembles."""
    from civic_digital_twins.dt_model import EnsembleAxisSpec, PartitionedEnsemble  # noqa: PLC0415

    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    cap = DistributionIndex("cap", stats.uniform, {"loc": 0.0, "scale": 1.0})
    model = _simple_model(season, cap)
    pens = PartitionedEnsemble(
        Scenario(model),
        axes=[EnsembleAxisSpec("cats", [season], size=2), EnsembleAxisSpec("dists", [cap], size=5)],
    )
    temp = ConditionalDistributionIndex("temp", parents=[season], factory=lambda **_kw: stats.norm())
    with pytest.raises(ValueError, match="single-axis"):
        sample_across(pens, [temp])


# ---------------------------------------------------------------------------
# n_samples_per_combo
# ---------------------------------------------------------------------------


def test_cpe_n_samples_per_combo_size():
    """Total size equals |categorical cross-product| × n_samples_per_combo."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    weather = CategoricalIndex("weather", {"good": 0.7, "bad": 0.3})
    model = _simple_model(season, weather)
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=10)
    assert ens.size == 4 * 10
    assert len(ens) == 4 * 10


def test_cpe_n_samples_per_combo_weights_sum_to_one():
    """Weights still sum to 1.0 with n_samples_per_combo > 1."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    model = _simple_model(season)
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=7)
    assert pytest.approx(ens.ensemble_weights[0].sum()) == 1.0


def test_cpe_n_samples_per_combo_equal_weight_within_combo():
    """Within each categorical combo all replicates share equal weight w_combo / N."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    model = _simple_model(season)
    N = 5
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=N)
    weights = ens.ensemble_weights[0]
    cat_arr = ens.assignments()[season]
    for value, expected_combo_weight in [("summer", 0.6), ("winter", 0.4)]:
        mask = cat_arr == value
        replicate_weights = weights[mask]
        assert len(replicate_weights) == N
        np.testing.assert_allclose(replicate_weights, expected_combo_weight / N, rtol=1e-9)


def test_cpe_n_samples_per_combo_cat_values_repeated():
    """Categorical assignments contain each combo value exactly n_samples_per_combo times."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    model = _simple_model(season)
    N = 4
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=N)
    cat_arr = ens.assignments()[season].tolist()
    assert cat_arr.count("summer") == N
    assert cat_arr.count("winter") == N


def test_cpe_n_samples_per_combo_dist_samples_vary():
    """Distribution samples are drawn independently for each replicate (not all equal)."""
    season = CategoricalIndex("season", {"summer": 1.0})  # single combo
    cap = DistributionIndex("cap", stats.norm, {"loc": 100.0, "scale": 10.0})
    model = _simple_model(season, cap)
    N = 50
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=N, rng=np.random.default_rng(0))
    assert ens.size == N
    # All samples should not be identical (astronomically unlikely with N=50).
    assert len(set(ens.assignments()[cap].tolist())) > 1


def test_cpe_n_samples_per_combo_reduces_variance():
    """Larger n_samples_per_combo reduces variance of the weighted-mean estimate."""
    season = CategoricalIndex("season", {"a": 1.0})  # single combo
    cap = DistributionIndex("cap", stats.norm, {"loc": 0.0, "scale": 1.0})
    model = _simple_model(season, cap)
    scenario = Scenario(model)

    def weighted_mean(ens: CrossProductEnsemble) -> float:
        w = ens.ensemble_weights[0]
        v = ens.assignments()[cap]
        return float(np.dot(w, v))

    rng_seed = 12345
    n_trials = 200
    means_small = [
        weighted_mean(CrossProductEnsemble(scenario, n_samples_per_combo=1, rng=np.random.default_rng(rng_seed + i)))
        for i in range(n_trials)
    ]
    means_large = [
        weighted_mean(CrossProductEnsemble(scenario, n_samples_per_combo=100, rng=np.random.default_rng(rng_seed + i)))
        for i in range(n_trials)
    ]
    # Variance with N=100 should be ~100× smaller than with N=1.
    assert np.var(means_large) < np.var(means_small) / 10


def test_cpe_n_samples_per_combo_conditional_dist_per_categorical():
    """ConditionalDistributionIndex gets N independent samples per categorical combo."""
    weather = CategoricalIndex("weather", {"hot": 0.5, "cold": 0.5})
    temp = ConditionalDistributionIndex(
        "temp",
        parents=[weather],
        factory=lambda weather: stats.norm(loc=30.0, scale=1.0) if weather == "hot" else stats.norm(loc=5.0, scale=1.0),
    )
    model = _simple_model(weather, temp)
    N = 20
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=N, rng=np.random.default_rng(7))
    assert ens.size == 2 * N
    a = ens.assignments()
    # Hot replicates should cluster near 30; cold near 5.
    hot_mask = a[weather] == "hot"
    cold_mask = a[weather] == "cold"
    assert np.all((a[temp][hot_mask] > 25.0) & (a[temp][hot_mask] < 35.0))
    assert np.all((a[temp][cold_mask] > 0.0) & (a[temp][cold_mask] < 10.0))


def test_cpe_n_samples_per_combo_dist_parent_uses_replicate_value():
    """When a ConditionalDistributionIndex has a DistributionIndex parent.

    Each replicate uses that replicate's parent sample, not a shared one.
    """
    season = CategoricalIndex("season", {"summer": 1.0})  # one combo
    base = DistributionIndex("base", stats.uniform, {"loc": 0.0, "scale": 1.0})
    derived = ConditionalDistributionIndex(
        "derived",
        parents=[season, base],
        # derived = base + tiny noise; so derived ≈ base per replicate
        factory=lambda **kw: stats.norm(loc=float(kw["base"]), scale=1e-6),
    )
    model = _simple_model(season, base, derived)
    N = 30
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=N, rng=np.random.default_rng(42))
    a = ens.assignments()
    assert a[base].shape == (N,)
    assert a[derived].shape == (N,)
    # Each derived value should be very close to its corresponding base value.
    np.testing.assert_allclose(a[derived], a[base], atol=1e-4)


def test_cpe_n_samples_per_combo_one_is_default():
    """n_samples_per_combo=1 (default) matches behaviour of omitting the parameter."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    cap = DistributionIndex("cap", stats.norm, {"loc": 0.0, "scale": 1.0})
    model = _simple_model(season, cap)
    scenario = Scenario(model)
    rng = np.random.default_rng(0)
    ens_default = CrossProductEnsemble(scenario, rng=np.random.default_rng(0))
    ens_explicit = CrossProductEnsemble(scenario, n_samples_per_combo=1, rng=np.random.default_rng(0))
    assert ens_default.size == ens_explicit.size
    np.testing.assert_array_equal(ens_default.ensemble_weights[0], ens_explicit.ensemble_weights[0])
    np.testing.assert_array_equal(ens_default.assignments()[season], ens_explicit.assignments()[season])
    np.testing.assert_array_equal(ens_default.assignments()[cap], ens_explicit.assignments()[cap])
    del rng  # unused; silence linter


def test_cpe_n_samples_per_combo_invalid_raises():
    """n_samples_per_combo < 1 raises ValueError."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    model = _simple_model(season)
    with pytest.raises(ValueError, match="n_samples_per_combo"):
        CrossProductEnsemble(Scenario(model), n_samples_per_combo=0)


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------


def test_importable_from_dt_model():
    """CrossProductEnsemble and sample_across are importable from civic_digital_twins.dt_model."""
    from civic_digital_twins.dt_model import CrossProductEnsemble as CPE  # noqa: PLC0415
    from civic_digital_twins.dt_model import sample_across as sa  # noqa: PLC0415

    assert CPE is CrossProductEnsemble
    assert sa is sample_across


def test_cross_product_ensemble_accepts_scenario():
    """CrossProductEnsemble can be constructed directly from a Scenario."""
    cat = CategoricalIndex("mode", {"walk": 0.5, "bike": 0.5})
    model = _simple_model(cat)
    scenario = Scenario(model)
    ens = CrossProductEnsemble(scenario)
    assert len(ens.ensemble_axes) == 1


# ---------------------------------------------------------------------------
# draw_batch
# ---------------------------------------------------------------------------


def test_cpe_draw_batch_returns_frozen_ensemble():
    """draw_batch returns a FrozenEnsemble with one ENSEMBLE axis and the requested size."""
    from scipy import stats  # noqa: PLC0415

    from civic_digital_twins.dt_model.simulation.ensemble import FrozenEnsemble  # noqa: PLC0415

    cap = DistributionIndex("cap", stats.uniform, {"loc": 0.0, "scale": 1.0})
    model = _simple_model(cap)
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=2, rng=np.random.default_rng(0))
    batch = ens.draw_batch(3, np.random.default_rng(1))
    assert isinstance(batch, FrozenEnsemble)
    assert len(batch.ensemble_axes) == 1
    # No categorical combos → 1 combo; draw_batch with size=3 → n_samples_per_combo=3 → 3 rows.
    assert batch.ensemble_weights[0].shape == (3,)


def test_cpe_draw_batch_conditional_dist_all_cat_parents():
    """draw_batch re-samples a ConditionalDistributionIndex whose parents are all categorical.

    Exercises _draw_from_combos: the CDI / no-dist-parents branch.
    """
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    temp = ConditionalDistributionIndex(
        "temp",
        parents=[season],
        factory=lambda season: (
            stats.norm(loc=30.0, scale=1.0) if season == "summer" else stats.norm(loc=5.0, scale=1.0)
        ),
    )
    model = _simple_model(season, temp)
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=1, rng=np.random.default_rng(0))

    # 2 combos × 3 samples_per_combo = 6 total
    batch = ens.draw_batch(3, np.random.default_rng(1))

    assert len(batch.ensemble_axes) == 1
    assert batch.ensemble_weights[0].shape == (6,)
    assert abs(batch.ensemble_weights[0].sum() - 1.0) < 1e-12
    assert set(batch.assignments()[season].tolist()) <= {"summer", "winter"}
    # temp values must be finite (sampled from the correct distribution per combo)
    assert np.all(np.isfinite(batch.assignments()[temp].astype(float)))


def test_cpe_draw_batch_conditional_dist_distribution_parent():
    """draw_batch re-samples a ConditionalDistributionIndex with a distribution parent.

    Exercises _draw_from_combos: the CDI / has-dist-parents branch (per-replicate sampling).
    """
    x = DistributionIndex("x", stats.norm, {"loc": 0.0, "scale": 1.0})
    y = ConditionalDistributionIndex(
        "y",
        parents=[x],
        factory=lambda x: stats.norm(loc=float(x), scale=0.1),
    )
    model = _simple_model(x, y)
    ens = CrossProductEnsemble(Scenario(model), n_samples_per_combo=2, rng=np.random.default_rng(0))

    # 1 combo (no categoricals) × 3 samples_per_combo = 3 total
    batch = ens.draw_batch(3, np.random.default_rng(7))

    assert batch.ensemble_weights[0].shape == (3,)
    assert abs(batch.ensemble_weights[0].sum() - 1.0) < 1e-12
    assert np.all(np.isfinite(batch.assignments()[x].astype(float)))
    assert np.all(np.isfinite(batch.assignments()[y].astype(float)))


def test_cpe_draw_batch_axis_not_none_raises():
    """draw_batch raises ValueError when axis= is not None."""
    from scipy import stats  # noqa: PLC0415

    cap = DistributionIndex("cap", stats.uniform, {"loc": 0.0, "scale": 1.0})
    model = _simple_model(cap)
    ens = CrossProductEnsemble(Scenario(model), rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="single ENSEMBLE axis"):
        ens.draw_batch(3, np.random.default_rng(1), axis="unc")


def test_cpe_no_rng_with_distributions():
    """CrossProductEnsemble with no rng samples distribution indexes deterministically."""
    from scipy import stats  # noqa: PLC0415

    cap = DistributionIndex("cap", stats.uniform, {"loc": 0.0, "scale": 1.0})
    model = _simple_model(cap)
    ens = CrossProductEnsemble(Scenario(model))  # no rng argument
    a = ens.assignments()
    # 1 combo × n_samples_per_combo=1 → shape (1,)
    assert a[cap].shape == (1,)


def test_cat_samples_no_rng_monte_carlo():
    """_cat_samples with rng=None falls back to np.random.choice for MC sampling."""
    # max_categorical_size=2 < len(values)=5 triggers the MC branch; rng=None uses np.random.choice.
    season = CategoricalIndex("season", {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2})
    model = _simple_model(season)
    # rng=None means _cat_samples uses np.random.choice
    ens = CrossProductEnsemble(Scenario(model), max_categorical_size=2, rng=None)
    assert ens.assignments()[season].shape == (2,)


# ---------------------------------------------------------------------------
# Scenario.parameter_axes auto-exclusion
# ---------------------------------------------------------------------------


def test_cpe_parameter_axes_auto_excluded():
    """CrossProductEnsemble auto-excludes indexes listed in Scenario.parameter_axes."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    pv = Index("presence", None)
    model = _simple_model(season, pv)
    ens = CrossProductEnsemble(Scenario(model, parameter_axes=[pv]))
    a = ens.assignments()
    assert season in a
    assert pv not in a


# ---------------------------------------------------------------------------
# Deprecation warnings
# ---------------------------------------------------------------------------


def test_cpe_draw_batch_categorical_stable_across_batches():
    """draw_batch reuses the categorical combos fixed at construction.

    When max_categorical_size < support size, categories are MC-sampled rather
    than enumerated.  Every call to draw_batch must return the *same* categorical
    values as the initial construction (_combo_cats), not a freshly re-sampled set.
    Regression test for the bug where draw_batch constructed a new
    CrossProductEnsemble internally (re-sampling categoricals with a different rng).
    """
    season = CategoricalIndex("season", {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2})
    model = _simple_model(season)
    # max_categorical_size=3 < 5 → MC sampling of categories (not full enumeration)
    ens = CrossProductEnsemble(Scenario(model), max_categorical_size=3, rng=np.random.default_rng(7))

    # Record which 3 category values were chosen at construction time.
    initial_cats = ens.assignments()[season].copy()  # shape (3,)

    # draw_batch with a *different* rng must return the same categorical combos,
    # only re-sampling distributions (none here).
    batch = ens.draw_batch(1, np.random.default_rng(99))  # 1 sample per combo → 3 rows
    batch_cats = batch.assignments()[season]

    np.testing.assert_array_equal(batch_cats, initial_cats)


def test_cpe_pinned_categorical_parent_of_conditional_dist():
    """CrossProductEnsemble handles a ConditionalDistributionIndex whose categorical parent is pinned.

    When a CategoricalIndex parent is pinned via Scenario(overrides={cat: "value"}),
    it is removed from abstract_indexes() and therefore absent from combo dicts.
    _compute_assignments must fall back to the scenario override value instead of
    raising KeyError.
    """
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    temp = ConditionalDistributionIndex(
        "temp",
        parents=[season],
        factory=lambda season: (
            stats.norm(loc=30.0, scale=1.0) if season == "summer" else stats.norm(loc=5.0, scale=1.0)
        ),
    )
    model = _simple_model(season, temp)
    # Pin season to "summer" — season is no longer abstract, but is still a parent of temp.
    scenario = Scenario(model, overrides={season: "summer"})
    ens = CrossProductEnsemble(scenario, rng=np.random.default_rng(0))

    # Should construct without KeyError and sample temp from the "summer" distribution.
    assert ens.size > 0
    temps = ens.assignments()[temp].astype(float)
    assert np.all(np.isfinite(temps))
    # All samples should come from summer distribution (mean ~30), not winter (mean ~5).
    assert np.mean(temps) > 20.0
