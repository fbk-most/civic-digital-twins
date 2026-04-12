"""Tests for PartitionedEnsemble — multi-axis batched ensemble."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model.model.axis import ENSEMBLE, Axis
from civic_digital_twins.dt_model.model.index import CategoricalIndex, DistributionIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.ensemble import EnsembleAxisSpec, PartitionedEnsemble
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist_index(name: str, lo: float = 0.0, hi: float = 1.0) -> Index:
    return DistributionIndex(name, stats.uniform, {"loc": lo, "scale": hi - lo})


def _make_model(*indexes) -> Model:
    return Model("test", list(indexes))


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_construction_single_axis():
    """PartitionedEnsemble with one axis wraps all abstract indexes."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)

    ens = PartitionedEnsemble(model, axes=[EnsembleAxisSpec("unc", indexes=[i_a, i_b], size=10)])
    assert len(ens.ensemble_axes) == 1
    assert ens.ensemble_axes[0].name == "unc"
    assert ens.ensemble_axes[0].role == ENSEMBLE
    assert ens.ensemble_weights[0].shape == (10,)
    assert np.isclose(ens.ensemble_weights[0].sum(), 1.0)


def test_construction_two_axes():
    """Two specs produce two ENSEMBLE axes with independent sizes."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)

    ens = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax0", indexes=[i_a], size=30),
            EnsembleAxisSpec("ax1", indexes=[i_b], size=50),
        ],
    )
    assert len(ens.ensemble_axes) == 2
    assert ens.ensemble_axes[0].name == "ax0"
    assert ens.ensemble_axes[1].name == "ax1"
    assert ens.ensemble_weights[0].shape == (30,)
    assert ens.ensemble_weights[1].shape == (50,)


def test_raises_on_uncovered_index_without_default():
    """ValueError when an abstract index is not in any spec and default_axis is None."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)

    with pytest.raises(ValueError, match="not covered"):
        PartitionedEnsemble(model, axes=[EnsembleAxisSpec("ax0", indexes=[i_a], size=10)])


def test_raises_on_non_abstract_index_in_spec():
    """ValueError when a spec lists an index that is not abstract in the model."""
    i_a = _dist_index("a")
    i_outside = _dist_index("outside")
    model = _make_model(i_a)

    with pytest.raises(ValueError, match="not an abstract index"):
        PartitionedEnsemble(model, axes=[EnsembleAxisSpec("ax", indexes=[i_a, i_outside], size=5)])


def test_raises_on_duplicate_index_across_specs():
    """ValueError when the same index appears in two specs."""
    i_a = _dist_index("a")
    model = _make_model(i_a)

    with pytest.raises(ValueError, match="more than one"):
        PartitionedEnsemble(
            model,
            axes=[
                EnsembleAxisSpec("ax0", indexes=[i_a], size=5),
                EnsembleAxisSpec("ax1", indexes=[i_a], size=5),
            ],
        )


def test_raises_on_duplicate_spec_names():
    """ValueError when two specs (or a spec and default_axis) share a name."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)

    with pytest.raises(ValueError, match="Duplicate"):
        PartitionedEnsemble(
            model,
            axes=[
                EnsembleAxisSpec("unc", indexes=[i_a], size=5),
                EnsembleAxisSpec("unc", indexes=[i_b], size=5),
            ],
        )


def test_raises_on_duplicate_name_with_default_axis():
    """ValueError when default_axis name clashes with an explicit spec name."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)

    with pytest.raises(ValueError, match="Duplicate"):
        PartitionedEnsemble(
            model,
            axes=[EnsembleAxisSpec("unc", indexes=[i_a], size=5)],
            default_axis=EnsembleAxisSpec("unc", indexes=[], size=5),
        )


def test_default_axis_absorbs_unassigned():
    """Unassigned abstract indexes are routed to the default_axis."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)

    ens = PartitionedEnsemble(
        model,
        axes=[EnsembleAxisSpec("ax0", indexes=[i_a], size=10)],
        default_axis=EnsembleAxisSpec("default", indexes=[], size=20),
    )
    assert len(ens.ensemble_axes) == 2
    assert ens.ensemble_axes[1].name == "default"
    assert ens.ensemble_weights[1].shape == (20,)


# ---------------------------------------------------------------------------
# assignments() shape contract
# ---------------------------------------------------------------------------


def test_assignments_shape_single_axis():
    """Single-axis: each assignment has shape (S,)."""
    i_a = _dist_index("a")
    model = _make_model(i_a)
    ens = PartitionedEnsemble(
        model,
        axes=[EnsembleAxisSpec("unc", indexes=[i_a], size=7)],
        rng=np.random.default_rng(0),
    )
    asgn = ens.assignments()
    assert i_a in asgn
    assert asgn[i_a].shape == (7,)


def test_assignments_shape_two_axes():
    """Two-axis: index on axis-0 has shape (S0, 1); index on axis-1 has shape (1, S1)."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)
    ens = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax0", indexes=[i_a], size=30),
            EnsembleAxisSpec("ax1", indexes=[i_b], size=50),
        ],
        rng=np.random.default_rng(1),
    )
    asgn = ens.assignments()
    assert asgn[i_a].shape == (30, 1)
    assert asgn[i_b].shape == (1, 50)


# ---------------------------------------------------------------------------
# Evaluation with PartitionedEnsemble
# ---------------------------------------------------------------------------


def test_evaluation_result_shape_two_axes():
    """evaluate() with two ENSEMBLE axes gives result shape (S0, S1)."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    i_result = Index("result", i_a.node + i_b.node)
    model = _make_model(i_a, i_b, i_result)

    ens = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax0", indexes=[i_a], size=10),
            EnsembleAxisSpec("ax1", indexes=[i_b], size=5),
        ],
        rng=np.random.default_rng(0),
    )
    result = Evaluation(model).evaluate(ensemble=ens)
    arr = result[i_result]
    # Shape: (S0=10, S1=5) — no trailing DOMAIN placeholder in non-timeseries models after bug fix #155
    assert arr.shape == (10, 5)


def test_marginalize_contracts_both_ensemble_axes():
    """marginalize() contracts both ENSEMBLE axes; result is a scalar."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    i_result = Index("result", i_a.node + i_b.node)
    model = _make_model(i_a, i_b, i_result)

    ens = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax0", indexes=[i_a], size=200),
            EnsembleAxisSpec("ax1", indexes=[i_b], size=200),
        ],
        rng=np.random.default_rng(42),
    )
    result = Evaluation(model).evaluate(ensemble=ens)
    marginalised = result.marginalize(i_result)
    # Both a and b ~ Uniform(0,1); E[a+b] = 1.0
    assert marginalised.shape == ()
    assert float(marginalised) == pytest.approx(1.0, abs=0.1)


def test_marginalize_order_independence():
    """marginalize() result is the same regardless of ENSEMBLE axis order.

    Reverses the spec order and checks the marginalised value is unchanged.
    """
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    i_result = Index("result", i_a.node + i_b.node)
    model = _make_model(i_a, i_b, i_result)

    rng0 = np.random.default_rng(7)
    ens_fwd = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax0", indexes=[i_a], size=100),
            EnsembleAxisSpec("ax1", indexes=[i_b], size=100),
        ],
        rng=rng0,
    )
    rng1 = np.random.default_rng(7)
    ens_rev = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax1", indexes=[i_b], size=100),
            EnsembleAxisSpec("ax0", indexes=[i_a], size=100),
        ],
        rng=rng1,
    )
    m_fwd = float(Evaluation(model).evaluate(ensemble=ens_fwd).marginalize(i_result))
    m_rev = float(Evaluation(model).evaluate(ensemble=ens_rev).marginalize(i_result))
    assert m_fwd == pytest.approx(m_rev, rel=0.05)


def test_axis_repr():
    """Axis.__repr__ returns a concise human-readable string."""
    ax = Axis("my_axis", ENSEMBLE)
    assert repr(ax) == "Axis('my_axis', role='ENSEMBLE')"


def test_categorical_index_in_partitioned_ensemble():
    """CategoricalIndex is sampled via its sample() in PartitionedEnsemble."""
    i_cat = CategoricalIndex("mode", {"bike": 0.6, "train": 0.4})
    model = _make_model(i_cat)

    ens = PartitionedEnsemble(
        model,
        axes=[EnsembleAxisSpec("unc", indexes=[i_cat], size=10)],
        rng=np.random.default_rng(0),
    )
    asgn = ens.assignments()
    assert i_cat in asgn
    assert asgn[i_cat].shape == (10,)
    assert all(v in ("bike", "train") for v in asgn[i_cat])


def test_partitioned_ensemble_without_rng():
    """PartitionedEnsemble works without an explicit rng (rng=None path)."""
    i_a = _dist_index("a")
    model = _make_model(i_a)

    ens = PartitionedEnsemble(model, axes=[EnsembleAxisSpec("unc", indexes=[i_a], size=5)])
    asgn = ens.assignments()
    assert i_a in asgn
    assert asgn[i_a].shape == (5,)


def test_result_weights_with_two_ensemble_axes():
    """EvaluationResult.weights returns the joint weight array for two ENSEMBLE axes."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)

    ens = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax0", indexes=[i_a], size=3),
            EnsembleAxisSpec("ax1", indexes=[i_b], size=4),
        ],
        rng=np.random.default_rng(0),
    )
    result = Evaluation(model).evaluate(ensemble=ens)
    w = result.weights
    # Joint weight = outer product of (3,) and (4,) uniform weights → (3, 4)
    assert w.shape == (3, 4)
    assert np.isclose(w.sum(), 1.0)


def test_raises_on_non_samplable_index_in_spec():
    """ValueError when assignments() is called with a plain abstract index (no distribution)."""
    I_plain = Index("plain", None)  # abstract but no distribution
    model = _make_model(I_plain)

    ens = PartitionedEnsemble(model, axes=[EnsembleAxisSpec("unc", indexes=[I_plain], size=5)])
    with pytest.raises(ValueError, match="not Distribution-backed"):
        ens.assignments()


def test_factorized_weights_are_uniform():
    """PartitionedEnsemble uses uniform weights per axis."""
    i_a = _dist_index("a")
    i_b = _dist_index("b")
    model = _make_model(i_a, i_b)
    ens = PartitionedEnsemble(
        model,
        axes=[
            EnsembleAxisSpec("ax0", indexes=[i_a], size=4),
            EnsembleAxisSpec("ax1", indexes=[i_b], size=6),
        ],
        rng=np.random.default_rng(0),
    )
    w0, w1 = ens.ensemble_weights
    assert np.allclose(w0, np.full(4, 0.25))
    assert np.allclose(w1, np.full(6, 1.0 / 6))
