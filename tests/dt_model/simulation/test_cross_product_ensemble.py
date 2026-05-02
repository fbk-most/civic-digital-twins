"""Tests for CrossProductEnsemble."""

# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass

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
    sample_across,
)

# ---------------------------------------------------------------------------
# Minimal model helpers
# ---------------------------------------------------------------------------


def _simple_model(
    *abstract_indexes: GenericIndex,
) -> Model:
    """Return a minimal Model wrapping the given abstract indexes."""

    @dataclass
    class Inputs:
        indexes: list[GenericIndex]

    @dataclass
    class Outputs:
        pass

    class _M(Model):
        def __init__(self, idxs: tuple[GenericIndex, ...]) -> None:
            super().__init__(
                "test",
                inputs=Inputs(indexes=list(idxs)),
                outputs=Outputs(),
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _M(abstract_indexes)


# ---------------------------------------------------------------------------
# Basic construction — unconditional CategoricalIndex
# ---------------------------------------------------------------------------


def test_cpe_single_categorical_enumerated():
    """Single 2-outcome CategoricalIndex is fully enumerated; weights sum to 1."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    model = _simple_model(season)
    ens = CrossProductEnsemble(model)
    assert ens.size == 2
    weights = ens.ensemble_weights[0]
    assert pytest.approx(weights.sum()) == 1.0
    assert set(ens.assignments()[season].tolist()) == {"summer", "winter"}


def test_cpe_two_categoricals_cross_product():
    """Two categoricals produce size = product of support sizes."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    weather = CategoricalIndex("weather", {"good": 0.7, "bad": 0.3})
    model = _simple_model(season, weather)
    ens = CrossProductEnsemble(model)
    assert ens.size == 4
    assert pytest.approx(ens.ensemble_weights[0].sum()) == 1.0


def test_cpe_weights_match_joint_probability():
    """Weights for enumerated cross-product equal the joint probability."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    weather = CategoricalIndex("weather", {"good": 0.7, "bad": 0.3})
    model = _simple_model(season, weather)
    ens = CrossProductEnsemble(model)
    a = ens.assignments()
    weights = ens.ensemble_weights[0]
    # Find the (summer, good) scenario.
    for i in range(ens.size):
        if a[season][i] == "summer" and a[weather][i] == "good":
            assert pytest.approx(weights[i], rel=1e-6) == 0.6 * 0.7


def test_cpe_no_abstract_indexes():
    """Model with no enumerable/sampleable abstract indexes produces size=1."""
    model = _simple_model()  # no abstract indexes
    ens = CrossProductEnsemble(model)
    assert ens.size == 1
    assert pytest.approx(ens.ensemble_weights[0].sum()) == 1.0


def test_cpe_len():
    """__len__ returns the number of scenarios."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    model = _simple_model(season)
    ens = CrossProductEnsemble(model)
    assert len(ens) == ens.size


# ---------------------------------------------------------------------------
# Restrictions
# ---------------------------------------------------------------------------


def test_cpe_restriction_subsets_support():
    """Restricting a CategoricalIndex to a subset reduces ensemble size."""
    season = CategoricalIndex("season", {"summer": 0.5, "spring": 0.3, "winter": 0.2})
    model = _simple_model(season)
    ens = CrossProductEnsemble(model, restrictions={season: ["summer", "winter"]})
    assert ens.size == 2
    assert set(ens.assignments()[season].tolist()) == {"summer", "winter"}


def test_cpe_restriction_single_value():
    """Restricting a categorical to one value gives weight 1 for that value."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    model = _simple_model(season)
    ens = CrossProductEnsemble(model, restrictions={season: ["summer"]})
    assert ens.size == 1
    assert ens.assignments()[season][0] == "summer"
    assert pytest.approx(ens.ensemble_weights[0][0]) == 1.0


def test_cpe_restriction_renormalises_weights():
    """Restricting to a subset renormalises probabilities over that subset."""
    season = CategoricalIndex("season", {"summer": 0.6, "winter": 0.4})
    model = _simple_model(season)
    ens = CrossProductEnsemble(model, restrictions={season: ["summer", "winter"]})
    weights = ens.ensemble_weights[0]
    # Probabilities in the restricted set: summer=0.6, winter=0.4 → already sum to 1.
    idx_summer = list(ens.assignments()[season]).index("summer")
    assert pytest.approx(weights[idx_summer], rel=1e-6) == 0.6


# ---------------------------------------------------------------------------
# MC sampling (max_categorical_size < support size)
# ---------------------------------------------------------------------------


def test_cpe_mc_sampling_when_support_exceeds_size():
    """When support > max_categorical_size, MC sampling is used; size = max_categorical_size."""
    season = CategoricalIndex("season", {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2})
    model = _simple_model(season)
    rng = np.random.default_rng(0)
    ens = CrossProductEnsemble(model, max_categorical_size=3, rng=rng)
    assert ens.size == 3
    assert pytest.approx(ens.ensemble_weights[0].sum()) == 1.0


# ---------------------------------------------------------------------------
# Distribution-backed indexes
# ---------------------------------------------------------------------------


def test_cpe_distribution_index_sampled():
    """DistributionIndex is sampled and present in assignments."""
    cap = DistributionIndex("cap", stats.uniform, {"loc": 90.0, "scale": 20.0})
    model = _simple_model(cap)
    ens = CrossProductEnsemble(model, rng=np.random.default_rng(42))
    a = ens.assignments()
    assert cap in a
    assert len(a[cap]) == ens.size
    assert np.all((a[cap] >= 90.0) & (a[cap] <= 110.0))


def test_cpe_mixed_categorical_and_distribution():
    """CategoricalIndex cross-product combined with DistributionIndex sampling."""
    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    cap = DistributionIndex("cap", stats.uniform, {"loc": 100.0, "scale": 50.0})
    model = _simple_model(season, cap)
    ens = CrossProductEnsemble(model, rng=np.random.default_rng(1))
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
    ens = CrossProductEnsemble(model)
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
    ens = CrossProductEnsemble(model)
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
    ens = CrossProductEnsemble(model, rng=rng)
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
    ens = CrossProductEnsemble(model, rng=rng)
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
    ens = CrossProductEnsemble(model)
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
    ens = CrossProductEnsemble(model)
    assert isinstance(ens, AxisEnsemble)


def test_cpe_single_ensemble_axis():
    """CrossProductEnsemble reports exactly one ENSEMBLE axis."""
    from civic_digital_twins.dt_model.model.axis import ENSEMBLE  # noqa: PLC0415

    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    model = _simple_model(season)
    ens = CrossProductEnsemble(model)
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
    a1 = CrossProductEnsemble(model, max_categorical_size=3, rng=np.random.default_rng(7)).assignments()
    a2 = CrossProductEnsemble(model, max_categorical_size=3, rng=np.random.default_rng(7)).assignments()
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
    ens = CrossProductEnsemble(model)
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
    ens = CrossProductEnsemble(model)
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
    ens = CrossProductEnsemble(model, rng=np.random.default_rng(1))
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
    ens = CrossProductEnsemble(model)
    with pytest.raises(ValueError, match="not present in the ensemble"):
        sample_across(ens, [temp])


def test_sample_across_multi_axis_raises():
    """Raises ValueError for multi-axis ensembles."""
    from civic_digital_twins.dt_model import EnsembleAxisSpec, PartitionedEnsemble  # noqa: PLC0415

    season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    cap = DistributionIndex("cap", stats.uniform, {"loc": 0.0, "scale": 1.0})
    model = _simple_model(season, cap)
    pens = PartitionedEnsemble(
        model,
        axes=[EnsembleAxisSpec("cats", [season], size=2), EnsembleAxisSpec("dists", [cap], size=5)],
    )
    temp = ConditionalDistributionIndex("temp", parents=[season], factory=lambda **_kw: stats.norm())
    with pytest.raises(ValueError, match="single-axis"):
        sample_across(pens, [temp])


# ---------------------------------------------------------------------------
# Public export
# ---------------------------------------------------------------------------


def test_importable_from_dt_model():
    """CrossProductEnsemble and sample_across are importable from civic_digital_twins.dt_model."""
    from civic_digital_twins.dt_model import CrossProductEnsemble as CPE  # noqa: PLC0415
    from civic_digital_twins.dt_model import sample_across as sa  # noqa: PLC0415

    assert CPE is CrossProductEnsemble
    assert sa is sample_across
