"""Tests for ConditionalCategoricalIndex and ConditionalDistributionIndex."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.model.index import (
    CategoricalIndex,
    ConditionalCategoricalIndex,
    ConditionalDistributionIndex,
    DistributionIndex,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_season = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
_weather = CategoricalIndex("weather", {"good": 0.6, "bad": 0.4})


def _weather_given_season(season: str) -> dict[str, float]:
    """Return P(weather | season) — sunny/rainy support."""
    if season == "summer":
        return {"sunny": 0.7, "rainy": 0.3}
    return {"sunny": 0.2, "rainy": 0.8}


def _dist_given_weather(weather: str):
    """Return a temperature distribution conditioned on weather."""
    if weather == "good":
        return stats.norm(loc=25.0, scale=2.0)
    return stats.norm(loc=10.0, scale=5.0)


# ===========================================================================
# ConditionalCategoricalIndex — construction
# ===========================================================================


def test_ccat_construction_basic():
    """ConditionalCategoricalIndex constructs with valid arguments."""
    ccat = ConditionalCategoricalIndex(
        "weather_cond",
        parents=[_season],
        support=["sunny", "rainy"],
        factory=_weather_given_season,
    )
    assert ccat.name == "weather_cond"
    assert ccat.value is None
    assert ccat.support == ["sunny", "rainy"]
    assert ccat.parents == [_season]


def test_ccat_node_is_placeholder():
    """Underlying graph node is a placeholder (abstract index)."""
    ccat = ConditionalCategoricalIndex(
        "w", parents=[_season], support=["a", "b"], factory=lambda **_kw: {"a": 0.5, "b": 0.5}
    )
    assert isinstance(ccat.node, graph.placeholder)


def test_ccat_empty_support_raises():
    """Empty support raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        ConditionalCategoricalIndex("w", parents=[_season], support=[], factory=lambda **_kw: {})


def test_ccat_invalid_parent_type_raises():
    """Non-categorical parent raises TypeError."""
    bad_parent = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 1.0})
    with pytest.raises(TypeError, match="must be a CategoricalIndex"):
        ConditionalCategoricalIndex(
            "w",
            parents=[bad_parent],  # type: ignore[list-item]
            support=["a"],
            factory=lambda **_kw: {"a": 1.0},  # type: ignore[arg-type]
        )


def test_ccat_parents_returns_copy():
    """Mutating the returned parents list does not affect internal state."""
    ccat = ConditionalCategoricalIndex(
        "w", parents=[_season], support=["a", "b"], factory=lambda **_kw: {"a": 0.5, "b": 0.5}
    )
    parents = ccat.parents
    parents.append(_weather)
    assert len(ccat.parents) == 1


def test_ccat_support_returns_copy():
    """Mutating the returned support list does not affect internal state."""
    ccat = ConditionalCategoricalIndex(
        "w", parents=[_season], support=["a", "b"], factory=lambda **_kw: {"a": 0.5, "b": 0.5}
    )
    sup = ccat.support
    sup.append("c")
    assert len(ccat.support) == 2


# ===========================================================================
# ConditionalCategoricalIndex — outcomes_for
# ===========================================================================


def test_ccat_outcomes_for_returns_correct_distribution():
    """outcomes_for returns the factory result for each parent configuration."""
    ccat = ConditionalCategoricalIndex(
        "weather_cond",
        parents=[_season],
        support=["sunny", "rainy"],
        factory=_weather_given_season,
    )
    out_summer = ccat.outcomes_for(season="summer")
    assert pytest.approx(out_summer["sunny"]) == 0.7
    assert pytest.approx(out_summer["rainy"]) == 0.3

    out_winter = ccat.outcomes_for(season="winter")
    assert pytest.approx(out_winter["sunny"]) == 0.2
    assert pytest.approx(out_winter["rainy"]) == 0.8


def test_ccat_outcomes_for_wrong_keys_raises():
    """Factory returning wrong keys raises ValueError."""
    ccat = ConditionalCategoricalIndex(
        "w",
        parents=[_season],
        support=["a", "b"],
        factory=lambda **_kw: {"x": 0.5, "y": 0.5},
    )
    with pytest.raises(ValueError, match="factory returned keys"):
        ccat.outcomes_for(season="summer")


def test_ccat_outcomes_for_non_positive_raises():
    """Factory returning zero probability raises ValueError."""
    ccat = ConditionalCategoricalIndex(
        "w",
        parents=[_season],
        support=["a", "b"],
        factory=lambda **_kw: {"a": 0.0, "b": 1.0},
    )
    with pytest.raises(ValueError, match="non-positive"):
        ccat.outcomes_for(season="summer")


def test_ccat_outcomes_for_bad_sum_raises():
    """Factory returning probabilities that don't sum to 1 raises ValueError."""
    ccat = ConditionalCategoricalIndex(
        "w",
        parents=[_season],
        support=["a", "b"],
        factory=lambda **_kw: {"a": 0.3, "b": 0.3},
    )
    with pytest.raises(ValueError, match="sum"):
        ccat.outcomes_for(season="summer")


# ===========================================================================
# ConditionalCategoricalIndex — conditional parent
# ===========================================================================


def test_ccat_parent_is_conditional_categorical():
    """ConditionalCategoricalIndex accepts another ConditionalCategoricalIndex as parent."""
    ccat1 = ConditionalCategoricalIndex(
        "weather_cond",
        parents=[_season],
        support=["sunny", "rainy"],
        factory=_weather_given_season,
    )
    ccat2 = ConditionalCategoricalIndex(
        "sub_weather",
        parents=[ccat1],
        support=["light", "heavy"],
        factory=lambda **_kw: {"light": 0.5, "heavy": 0.5},
    )
    assert ccat2.parents == [ccat1]


# ===========================================================================
# ConditionalCategoricalIndex — repr
# ===========================================================================


def test_ccat_repr():
    """Repr includes class name and index name."""
    ccat = ConditionalCategoricalIndex(
        "weather_cond", parents=[_season], support=["sunny", "rainy"], factory=_weather_given_season
    )
    r = repr(ccat)
    assert "ConditionalCategoricalIndex" in r
    assert "weather_cond" in r


# ===========================================================================
# ConditionalDistributionIndex — construction
# ===========================================================================


def test_cdist_construction_basic():
    """ConditionalDistributionIndex constructs with valid arguments."""
    cdist = ConditionalDistributionIndex("temp", parents=[_weather], factory=_dist_given_weather)
    assert cdist.name == "temp"
    assert cdist.value is None
    assert cdist.parents == [_weather]


def test_cdist_node_is_placeholder():
    """Underlying graph node is a placeholder (abstract index)."""
    cdist = ConditionalDistributionIndex("temp", parents=[_weather], factory=_dist_given_weather)
    assert isinstance(cdist.node, graph.placeholder)


def test_cdist_zero_parents():
    """ConditionalDistributionIndex with no parents behaves as unconditional."""
    cdist = ConditionalDistributionIndex("noise", parents=[], factory=lambda: stats.norm(loc=0.0, scale=1.0))
    assert cdist.parents == []
    d = cdist.distribution_for()
    assert d.mean() == pytest.approx(0.0)


def test_cdist_invalid_parent_type_raises():
    """Non-ensemble-resolvable parent raises TypeError."""
    from civic_digital_twins.dt_model.model.index import ConstIndex  # noqa: PLC0415

    bad = ConstIndex("c", 1.0)
    with pytest.raises(TypeError, match="must be a CategoricalIndex"):
        ConditionalDistributionIndex("t", parents=[bad], factory=lambda _c: stats.norm())  # type: ignore[list-item]


def test_cdist_parents_returns_copy():
    """Mutating the returned parents list does not affect internal state."""
    cdist = ConditionalDistributionIndex("t", parents=[_weather], factory=_dist_given_weather)
    p = cdist.parents
    p.append(_season)
    assert len(cdist.parents) == 1


# ===========================================================================
# ConditionalDistributionIndex — distribution_for
# ===========================================================================


def test_cdist_distribution_for_returns_frozen_dist():
    """distribution_for returns the correct frozen distribution per parent config."""
    cdist = ConditionalDistributionIndex("temp", parents=[_weather], factory=_dist_given_weather)
    d_good = cdist.distribution_for(weather="good")
    d_bad = cdist.distribution_for(weather="bad")
    assert d_good.mean() == pytest.approx(25.0)
    assert d_bad.mean() == pytest.approx(10.0)


def test_cdist_rvs_produces_samples():
    """Sampling from the returned distribution yields values near the mean."""
    cdist = ConditionalDistributionIndex("temp", parents=[_weather], factory=_dist_given_weather)
    d = cdist.distribution_for(weather="good")
    import numpy as np  # noqa: PLC0415

    samples = np.asarray(d.rvs(size=100))
    assert len(samples) == 100
    assert abs(float(samples.mean()) - 25.0) < 2.0


# ===========================================================================
# ConditionalDistributionIndex — parent types
# ===========================================================================


def test_cdist_accepts_distribution_index_parent():
    """ConditionalDistributionIndex accepts DistributionIndex as parent."""
    di = DistributionIndex("mu", stats.uniform, {"loc": 0.0, "scale": 1.0})
    cdist = ConditionalDistributionIndex("x", parents=[di], factory=lambda mu: stats.norm(loc=mu, scale=1.0))
    assert cdist.parents == [di]


def test_cdist_accepts_conditional_categorical_parent():
    """ConditionalDistributionIndex accepts ConditionalCategoricalIndex as parent."""
    ccat = ConditionalCategoricalIndex(
        "w_cond", parents=[_season], support=["sunny", "rainy"], factory=_weather_given_season
    )
    cdist = ConditionalDistributionIndex(
        "temp",
        parents=[ccat],
        factory=lambda w_cond: stats.norm(loc=20.0 if w_cond == "sunny" else 10.0, scale=2.0),
    )
    assert cdist.parents == [ccat]


def test_cdist_accepts_conditional_distribution_parent():
    """ConditionalDistributionIndex accepts another ConditionalDistributionIndex as parent."""
    cdist1 = ConditionalDistributionIndex("mu", parents=[_weather], factory=lambda _w: stats.norm(loc=25.0, scale=1.0))
    cdist2 = ConditionalDistributionIndex("x", parents=[cdist1], factory=lambda mu: stats.norm(loc=mu, scale=0.5))
    assert cdist2.parents == [cdist1]


# ===========================================================================
# ConditionalDistributionIndex — repr
# ===========================================================================


def test_cdist_repr():
    """Repr includes class name and index name."""
    cdist = ConditionalDistributionIndex("temp", parents=[_weather], factory=_dist_given_weather)
    r = repr(cdist)
    assert "ConditionalDistributionIndex" in r
    assert "temp" in r


# ===========================================================================
# ConditionalDistributionIndex — sample_for
# ===========================================================================


def test_cdist_sample_for_single_sample():
    """sample_for returns a float ndarray of size 1 by default."""
    cdist = ConditionalDistributionIndex("temp", parents=[_weather], factory=_dist_given_weather)
    rng = np.random.default_rng(42)
    result = cdist.sample_for(rng=rng, weather="good")
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert isinstance(result[0], np.floating)
    assert 15.0 < result[0] < 35.0


def test_cdist_sample_for_size_n():
    """sample_for returns an ndarray of shape (N,) for size N."""
    cdist = ConditionalDistributionIndex("temp", parents=[_weather], factory=_dist_given_weather)
    rng = np.random.default_rng(42)
    result = cdist.sample_for(rng=rng, size=5, weather="good")
    assert result.shape == (5,)
    assert isinstance(result, np.ndarray)


def test_cdist_sample_for_reproducible():
    """Same rng seed produces same samples."""
    cdist = ConditionalDistributionIndex("temp", parents=[_weather], factory=_dist_given_weather)
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    s1 = cdist.sample_for(rng=rng1, size=10, weather="good")
    s2 = cdist.sample_for(rng=rng2, size=10, weather="good")
    assert np.allclose(s1, s2)


# ===========================================================================
# ConditionalCategoricalIndex — sample_for
# ===========================================================================


def test_ccat_sample_for_single_sample():
    """sample_for returns an object-dtype ndarray of shape (1,) by default."""
    ccat = ConditionalCategoricalIndex(
        "weather_cond",
        parents=[_season],
        support=["sunny", "rainy"],
        factory=_weather_given_season,
    )
    rng = np.random.default_rng(42)
    result = ccat.sample_for(rng=rng, season="summer")
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] in ("sunny", "rainy")


def test_ccat_sample_for_size_n():
    """sample_for returns an ndarray of shape (N,) for size N."""
    ccat = ConditionalCategoricalIndex(
        "weather_cond",
        parents=[_season],
        support=["sunny", "rainy"],
        factory=_weather_given_season,
    )
    rng = np.random.default_rng(42)
    result = ccat.sample_for(rng=rng, size=20, season="summer")
    assert result.shape == (20,)
    assert all(k in ("sunny", "rainy") for k in result)


def test_ccat_sample_for_reproducible():
    """Same rng seed produces same samples."""
    ccat = ConditionalCategoricalIndex(
        "weather_cond",
        parents=[_season],
        support=["sunny", "rainy"],
        factory=_weather_given_season,
    )
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    s1 = ccat.sample_for(rng=rng1, size=10, season="summer")
    s2 = ccat.sample_for(rng=rng2, size=10, season="summer")
    assert list(s1) == list(s2)


# ===========================================================================
# Public export
# ===========================================================================


def test_importable_from_dt_model():
    """Both conditional index classes are importable from civic_digital_twins.dt_model."""
    from civic_digital_twins.dt_model import ConditionalCategoricalIndex as CCI  # noqa: PLC0415
    from civic_digital_twins.dt_model import ConditionalDistributionIndex as CDI  # noqa: PLC0415

    assert CCI is ConditionalCategoricalIndex
    assert CDI is ConditionalDistributionIndex
