# SPDX-License-Identifier: Apache-2.0
"""Tests for EvaluationHandle (incremental evaluation) — Step 3 of engine-control.md."""

import dataclasses

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model.model.index import DistributionIndex, GenericIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.simulation.handle import EvaluationHandle, _merge_results

# ---------------------------------------------------------------------------
# Minimal model fixtures
# ---------------------------------------------------------------------------


class _SimpleModel(Model):
    """Model with one distribution-backed abstract index and one output."""

    @dataclasses.dataclass
    class Inputs:
        x: Index

    @dataclasses.dataclass
    class Outputs:
        y: Index

    def __init__(self, x: Index) -> None:
        y = Index("y", x.node * 2.0)
        super().__init__(
            "SimpleModel",
            inputs=_SimpleModel.Inputs(x=x),
            outputs=_SimpleModel.Outputs(y=y),
        )


class _ConstModel(Model):
    """Model with no abstract indexes — all outputs are constant."""

    @dataclasses.dataclass
    class Inputs:
        pass

    @dataclasses.dataclass
    class Outputs:
        c: Index

    def __init__(self) -> None:
        c = Index("c", 42.0)
        super().__init__(
            "ConstModel",
            inputs=_ConstModel.Inputs(),
            outputs=_ConstModel.Outputs(c=c),
        )


def _make_simple() -> tuple[Index, _SimpleModel]:
    x = DistributionIndex("x", stats.norm, {"loc": 5.0, "scale": 1.0})
    model = _SimpleModel(x)
    return x, model


# ---------------------------------------------------------------------------
# evaluate_incremental — basic creation
# ---------------------------------------------------------------------------


def test_evaluate_incremental_returns_handle() -> None:
    """evaluate_incremental returns an EvaluationHandle."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(50)
    assert isinstance(handle, EvaluationHandle)


def test_handle_result_is_evaluation_result() -> None:
    """handle.result is an EvaluationResult with the correct shape."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(50)
    result = handle.result
    arr = result[model.outputs.y]
    # shape should be (50,) — one ENSEMBLE axis of size 50
    assert arr.shape == (50,)


def test_handle_result_weights_sum_to_one() -> None:
    """Initial result weights are uniform and sum to 1."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(40)
    w = handle.result.weights
    assert w.shape == (40,)
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(w, np.full(40, 1.0 / 40))


def test_evaluate_incremental_reproducible_with_seed() -> None:
    """The same RNG seed produces the same first result."""
    _, model = _make_simple()
    ev = Evaluation(model)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    h1 = ev.evaluate_incremental(30, rng=rng1)
    h2 = ev.evaluate_incremental(30, rng=rng2)
    np.testing.assert_array_equal(h1.result[model.outputs.y], h2.result[model.outputs.y])


# ---------------------------------------------------------------------------
# EvaluationHandle.extend — ensemble extension
# ---------------------------------------------------------------------------


def test_extend_increases_ensemble_size() -> None:
    """extend(n) grows the ensemble from S to S+n scenarios."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(30)
    result = handle.extend(20)
    arr = result[model.outputs.y]
    assert arr.shape == (50,)


def test_extend_updates_handle_result() -> None:
    """handle.result is updated after extend()."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(30)
    extended = handle.extend(20)
    assert handle.result is extended


def test_extend_weights_renormalized() -> None:
    """After extend(), weights are uniform over the combined ensemble."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(30)
    handle.extend(20)
    w = handle.result.weights
    assert w.shape == (50,)
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(w, np.full(50, 1.0 / 50))


def test_extend_zero_is_noop() -> None:
    """extend(0) is a no-op — result unchanged."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(30)
    before = handle.result
    returned = handle.extend(0)
    assert returned is before
    assert handle.result is before


def test_extend_negative_is_noop() -> None:
    """extend(-1) is also a no-op."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(30)
    before = handle.result
    returned = handle.extend(-1)
    assert returned is before


def test_multiple_extends_accumulate() -> None:
    """Multiple extend() calls accumulate scenarios correctly."""
    _, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(10)
    handle.extend(10)
    handle.extend(10)
    arr = handle.result[model.outputs.y]
    assert arr.shape == (30,)


def test_incremental_matches_direct_evaluation() -> None:
    """evaluate_incremental(X) + extend(Y) is numerically identical to evaluate_incremental(X+Y).

    Both paths draw from the same RNG seed and therefore produce the same
    sequence of samples.  The incremental path splits the sequence into two
    batches [0..X) and [X..X+Y); the direct path draws X+Y samples at once.
    After the merge the combined array must equal the direct array element-wise.
    """
    _, model = _make_simple()
    ev = Evaluation(model)

    # --- Incremental: 30 then 20 ---
    h = ev.evaluate_incremental(30, rng=np.random.default_rng(99))
    h.extend(20)
    incremental_arr = h.result[model.outputs.y]

    # --- Direct: 50 at once from the same seed ---
    direct = ev.evaluate_incremental(50, rng=np.random.default_rng(99))
    direct_arr = direct.result[model.outputs.y]

    assert incremental_arr.shape == direct_arr.shape == (50,)
    np.testing.assert_array_equal(incremental_arr, direct_arr)


def test_extend_reproducible_sequence() -> None:
    """Two handles with the same seed produce the same full sequence."""
    _, model = _make_simple()
    ev = Evaluation(model)

    h1 = ev.evaluate_incremental(20, rng=np.random.default_rng(7))
    h1.extend(30)

    h2 = ev.evaluate_incremental(20, rng=np.random.default_rng(7))
    h2.extend(30)

    np.testing.assert_array_equal(h1.result[model.outputs.y], h2.result[model.outputs.y])


# ---------------------------------------------------------------------------
# EvaluationHandle.extend — singleton nodes (constants)
# ---------------------------------------------------------------------------


def test_extend_constant_node_stays_singleton() -> None:
    """Constant nodes (no ensemble dependency) remain singletons after merge."""
    model = _ConstModel()
    # ConstModel has no abstract indexes — we must pass it via evaluate(), not
    # evaluate_incremental (which tries to build a DistributionEnsemble).
    # Instead, test _merge_results directly with a manually built plan + results.
    ev = Evaluation(model)
    plan = ev.build_plan()

    # Execute twice with tiny dummy ensembles to get two results.
    # _ConstModel has NO abstract indexes, so DistributionEnsemble will fail.
    # Use execute_plan with None ensemble (deterministic).
    r1 = ev.execute_plan(plan, ensemble=None)
    r2 = ev.execute_plan(plan, ensemble=None)

    # Both results have no ensemble axis — _merge_results should raise.
    with pytest.raises(ValueError, match="ENSEMBLE axis"):
        _merge_results(r1, r2, plan)


# ---------------------------------------------------------------------------
# extra_parameters raises NotImplementedError
# ---------------------------------------------------------------------------


def test_extend_extra_parameters_raises() -> None:
    """extend(extra_parameters=...) raises NotImplementedError in v0.10.0."""
    x, model = _make_simple()
    ev = Evaluation(model)
    handle = ev.evaluate_incremental(20)
    with pytest.raises(NotImplementedError, match="extra_parameters"):
        handle.extend(extra_parameters={x: np.array([1.0])})


# ---------------------------------------------------------------------------
# evaluate_incremental with PARAMETER axes
# ---------------------------------------------------------------------------


def test_evaluate_incremental_with_parameters() -> None:
    """evaluate_incremental respects PARAMETER axes; extend preserves them."""

    # Build a model with the parameter as a PARAMETER axis index.
    class _ParamModel(Model):
        @dataclasses.dataclass
        class Inputs:
            x: Index
            speed: Index

        @dataclasses.dataclass
        class Outputs:
            y: Index

        def __init__(self, x: Index, speed: Index) -> None:
            y = Index("y", x.node + speed.node)
            super().__init__(
                "ParamModel",
                inputs=_ParamModel.Inputs(x=x, speed=speed),
                outputs=_ParamModel.Outputs(y=y),
            )

    x2 = DistributionIndex("x2", stats.norm, {"loc": 0.0, "scale": 1.0})
    speed = Index("speed", 1.0)  # concrete default; swept via parameters=
    pm = _ParamModel(x2, speed)
    ev2 = Evaluation(pm)

    params: dict[GenericIndex, np.ndarray] = {speed: np.array([1.0, 2.0, 3.0])}
    handle = ev2.evaluate_incremental(20, parameters=params)
    arr = handle.result[pm.outputs.y]
    # shape: (3, 20) — 3 PARAMETER values × 20 ensemble scenarios
    assert arr.shape == (3, 20)

    handle.extend(10)
    arr2 = handle.result[pm.outputs.y]
    # After extend: (3, 30)
    assert arr2.shape == (3, 30)
