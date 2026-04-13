"""Acceptance tests for typed-axes milestone (#142, #134).

Covers the checklist in docs/design/sessions/typed-axes.md §7:
- Canonical shape contract
- S == T regression (historical #142 failure mode)
- Broadcasting correctness (scalar ↔ timeseries)
- Multi-axis ensemble (factorized weights)
- Backward compatibility (deprecation window)
"""

# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model.model.index import DistributionIndex, Index, TimeseriesIndex
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.ensemble import (
    DistributionEnsemble,
    EnsembleAxisSpec,
    PartitionedEnsemble,
    WeightedScenario,
)
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist_index(name: str, lo: float = 0.0, hi: float = 1.0) -> Index:
    return DistributionIndex(name, stats.uniform, {"loc": lo, "scale": hi - lo})


def _make_model(*indexes) -> Model:
    return Model("test", list(indexes))


# ---------------------------------------------------------------------------
# Canonical shape contract
# ---------------------------------------------------------------------------


def test_parameter_then_ensemble_then_domain_order():
    """result[idx] shape follows (*PARAMETER, *ENSEMBLE) order for scalar nodes.

    1 PARAMETER axis (size 3), 1 ENSEMBLE axis (S=5, from DistributionEnsemble).
    i_p has a concrete default value and is swept via parameters=; i_x is the
    only abstract index (sampled by DistributionEnsemble → ENSEMBLE dim).
    Expected shape: (3, 5) — PARAMETER first, ENSEMBLE second.
    """
    i_x = _dist_index("x")  # abstract: sampled by ensemble → ENSEMBLE dim
    i_p = Index("p", 1.0)  # concrete default, swept via parameters= → PARAMETER dim
    i_result = Index("result", i_p.node * i_x.node)
    model = _make_model(i_x, i_p, i_result)

    pp = np.array([1.0, 2.0, 3.0])
    ens = DistributionEnsemble(model, size=5, rng=np.random.default_rng(0))

    result = Evaluation(model).evaluate(ensemble=ens, parameters={i_p: pp})
    arr = result[i_result]
    # Shape: (N_p=3, S=5) — PARAMETER first, ENSEMBLE second
    assert arr.shape == (3, 5)


def test_no_shape_heuristic_constant_node():
    """A constant node marginalizes to a scalar regardless of ENSEMBLE size."""
    i_c = Index("c", 42.0)
    i_x = _dist_index("x")  # gives DistributionEnsemble something to sample
    model = _make_model(i_c, i_x)

    ens = DistributionEnsemble(model, size=4, rng=np.random.default_rng(0))
    result = Evaluation(model).evaluate(ensemble=ens)

    marginalised = result.marginalize(i_c)
    # Constant node: marginalize over ENSEMBLE should give scalar 42.0
    assert float(marginalised) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# S == T regression (#142)
# ---------------------------------------------------------------------------


def test_s_equals_t_ensemble_contracted_not_time():
    """S == T: ENSEMBLE axis contracted, time axis preserved — shape (T,).

    Uses a timeseries constant (T=5 steps) and ensemble size S=5.
    The historical bug would confuse which axis to contract.
    """
    T = 5
    S = 5
    ts = TimeseriesIndex("ts", np.arange(float(T)))  # shape (T,) = (5,)
    i_x = _dist_index("x")
    model = _make_model(ts, i_x)

    ens = DistributionEnsemble(model, size=S, rng=np.random.default_rng(0))
    result = Evaluation(model).evaluate(ensemble=ens)

    # ts is a constant timeseries — marginalize over ENSEMBLE should preserve (T,)
    marginalised = result.marginalize(ts)
    assert marginalised.shape == (T,)
    assert np.allclose(marginalised, np.arange(float(T)))


def test_deterministic_timeseries_no_ensemble_contraction():
    """Deterministic (ensemble=None) timeseries: no ENSEMBLE axis; shape stays (T,)."""
    T = 24
    ts = TimeseriesIndex("ts", np.ones(T))
    model = _make_model(ts)

    result = Evaluation(model).evaluate(ensemble=None)
    marginalised = result.marginalize(ts)
    assert marginalised.shape == (T,)
    assert np.allclose(marginalised, np.ones(T))


def test_two_ensemble_axes_one_equals_t():
    """Two ENSEMBLE axes where S1 == T: both ENSEMBLE axes contracted, time preserved.

    Setup: timeseries constant (T=24), PartitionedEnsemble with axes of sizes
    S0=10 and S1=24.  marginalize() must contract S0 and S1, not time.
    """
    T = 24
    S0 = 10
    S1 = T  # deliberately equal to T
    ts = TimeseriesIndex("ts", np.arange(float(T)))
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(ts, i_a, i_b)

    ens = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax0", indexes=[i_a], size=S0),
            EnsembleAxisSpec("ax1", indexes=[i_b], size=S1),
        ],
        rng=np.random.default_rng(0),
    )
    result = Evaluation(model).evaluate(ensemble=ens)

    # ts is constant — marginalize over both ENSEMBLE axes → shape (T,)
    marginalised = result.marginalize(ts)
    assert marginalised.shape == (T,)
    assert np.allclose(marginalised, np.arange(float(T)))


# ---------------------------------------------------------------------------
# Broadcasting correctness (scalar ↔ timeseries)
# ---------------------------------------------------------------------------


def test_scalar_ensemble_broadcasts_with_timeseries():
    """Scalar abstract index (S,) broadcasts with timeseries (T,) → (S, T)."""
    T = 6
    S = 4
    ts = TimeseriesIndex("ts", np.ones(T))
    i_x = _dist_index("x")
    i_result = Index("result", i_x.node * ts.node)
    model = _make_model(ts, i_x, i_result)

    ens = DistributionEnsemble(model, size=S, rng=np.random.default_rng(0))
    result = Evaluation(model).evaluate(ensemble=ens)

    arr = result[i_result]
    # Shape should be (S, T) = (4, 6) — ENSEMBLE then DOMAIN
    assert arr.shape == (S, T)


def test_index_sum_axis_minus1_over_timeseries_with_ensemble():
    """Index.sum(axis=-1) on timeseries+ensemble: time reduced, shape (*ENSEMBLE, 1)."""
    T = 6
    S = 4
    ts = TimeseriesIndex("ts", np.arange(1.0, T + 1))
    i_x = _dist_index("x")
    i_prod = Index("prod", i_x.node * ts.node)
    i_sum = Index("sum", i_prod.sum(axis=-1))  # keepdims=True by convention
    model = _make_model(ts, i_x, i_prod, i_sum)

    ens = DistributionEnsemble(model, size=S, rng=np.random.default_rng(0))
    result = Evaluation(model).evaluate(ensemble=ens)

    arr = result[i_sum]
    # sum(axis=-1) reduces T → 1 (keepdims): shape (S, 1)
    assert arr.shape == (S, 1)


# ---------------------------------------------------------------------------
# Known gap — PARAMETER + timeseries, no ensemble (#157)
# ---------------------------------------------------------------------------


def test_parameter_timeseries_no_ensemble_broadcast():
    """PARAMETER sweep × timeseries with no ensemble broadcasts correctly.

    With n_ensemble=0, PARAMETER substitutions get a trailing timeseries
    placeholder so (N, 1) × (T,) → (N, T).
    """
    T = 6
    N = 3  # deliberately different from T
    assert N != T

    ts = TimeseriesIndex("ts", np.ones(T))
    i_p = Index("p", 1.0)
    i_result = Index("result", i_p.node * ts.node)
    model = _make_model(ts, i_p, i_result)

    pp = np.array([1.0, 2.0, 3.0])
    assert pp.size == N

    # No ensemble — pure PARAMETER sweep.  Expected shape: (N, T) = (3, 6).
    result = Evaluation(model).evaluate(ensemble=None, parameters={i_p: pp})
    arr = result[i_result]
    assert arr.shape == (N, T)


# ---------------------------------------------------------------------------
# Regression tests — all-axes invariant bugs (#155, #156)
# ---------------------------------------------------------------------------


def test_grid_ensemble_constant_no_indexerror():
    """grid+ensemble: constant node marginalises correctly without IndexError (#155).

    Bug: with 2 PARAMETER axes and 1 ENSEMBLE axis the old per-axis singleton
    insertion clamped to position 0 for a scalar constant, producing shape (1,)
    instead of (1, 1, 1).  marginalize() then tried arr.shape[2] → IndexError.
    """
    i_x = Index("x", None)
    i_y = Index("y", None)
    i_c = Index("c", 42.0)
    model = _make_model(i_x, i_y, i_c)

    xs = np.array([1.0, 2.0])
    ys = np.array([10.0, 20.0, 30.0])

    result = Evaluation(model).evaluate([(1.0, {})], parameters={i_x: xs, i_y: ys})
    # Must not raise IndexError.  The constant has PARAMETER singleton dims
    # preserved (shape (1, 1)) so we test all values equal 42.0.
    marginalised = result.marginalize(i_c)
    assert np.all(np.isclose(marginalised, 42.0))


def test_grid_ensemble_timeseries_broadcast_no_valueerror():
    """grid+ensemble+timeseries: PARAMETER sub broadcasts with timeseries (#156).

    Bug: with at least one PARAMETER axis the `n_params == 0` guard suppressed
    the trailing-1 appended to ENSEMBLE substitutions, so an ENSEMBLE scalar
    assignment got shape (S,) instead of (S, 1).  A formula that multiplied a
    PARAMETER result (N, S) by a timeseries (T,) then tried to broadcast (N, S)
    against T → ValueError when S != T.
    """
    T = 7
    S = 4
    ts = TimeseriesIndex("ts", np.ones(T))
    i_x = _dist_index("x")
    i_p = Index("p", 1.0)
    i_result = Index("result", i_p.node * i_x.node * ts.node)
    model = _make_model(ts, i_x, i_p, i_result)

    pp = np.array([1.0, 2.0, 3.0])
    ens = DistributionEnsemble(model, size=S, rng=np.random.default_rng(0))

    # Must not raise ValueError during evaluate().
    result = Evaluation(model).evaluate(ensemble=ens, parameters={i_p: pp})
    arr = result[i_result]
    # Shape: (N_p=3, S=4, T=7)
    assert arr.shape == (3, S, T)


def test_grid_ensemble_constant_and_timeseries_both_normalised():
    """grid+ensemble+timeseries: constant AND timeseries nodes both get correct shape.

    Combines #155 and #156 in one model: 1 PARAMETER axis, 1 ENSEMBLE axis,
    1 timeseries constant, 1 scalar constant — both must be injectable as
    singleton dims at all n_full positions.
    """
    T = 5
    S = 3
    N = 2
    ts = TimeseriesIndex("ts", np.arange(float(T)))
    i_c = Index("c", 10.0)
    i_x = _dist_index("x")
    i_p = Index("p", 1.0)
    model = _make_model(ts, i_c, i_x, i_p)

    pp = np.array([1.0, 2.0])
    assert pp.size == N
    ens = DistributionEnsemble(model, size=S, rng=np.random.default_rng(0))

    result = Evaluation(model).evaluate(ensemble=ens, parameters={i_p: pp})
    # Constant scalar: PARAMETER singleton dims are preserved by _squeeze_domain,
    # so shape is (1,) after squeezing the ENSEMBLE dim.  All values equal 10.0.
    assert np.all(np.isclose(result.marginalize(i_c), 10.0))
    # Timeseries: not downstream of any substitution → (1, 1, T) after normalisation.
    # After squeezing ENSEMBLE singleton at pos 1 → (1, T); PARAMETER singleton preserved.
    marginalised_ts = result.marginalize(ts)
    assert marginalised_ts.shape == (1, T)
    assert np.allclose(marginalised_ts, np.arange(float(T))[None, :])


# ---------------------------------------------------------------------------
# Backward compatibility (deprecation window)
# ---------------------------------------------------------------------------


def test_legacy_iterable_emits_deprecation_warning():
    """Passing Iterable[WeightedScenario] emits DeprecationWarning."""
    i_x = _dist_index("x")
    model = _make_model(i_x)
    scenarios: list[WeightedScenario] = [(1.0, {i_x: 0.5})]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Evaluation(model).evaluate(scenarios)

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("WeightedScenario" in str(w.message) for w in deprecations)


def test_legacy_iterable_gives_correct_results():
    """Legacy Iterable[WeightedScenario] adapter yields correct marginalised values."""
    i_x = Index("x", None)
    i_result = Index("result", i_x.node * 2.0)
    model = _make_model(i_x, i_result)

    scenarios: list[WeightedScenario] = [(0.5, {i_x: 1.0}), (0.5, {i_x: 3.0})]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = Evaluation(model).evaluate(scenarios)

    # E[result] = 0.5*(1*2) + 0.5*(3*2) = 0.5*2 + 0.5*6 = 4.0
    assert float(result.marginalize(i_result)) == pytest.approx(4.0)


def test_empty_scenario_list_is_deterministic():
    """Passing [] emits DeprecationWarning and evaluates deterministically."""
    i_c = Index("c", 7.0)
    model = _make_model(i_c)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Evaluation(model).evaluate([])

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations  # at least one deprecation warning
    assert float(result.marginalize(i_c)) == pytest.approx(7.0)
