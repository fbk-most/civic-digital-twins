# SPDX-License-Identifier: Apache-2.0
"""Tests for EvaluationHandle (incremental evaluation)."""

from typing import Any

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import define, inputs, outputs
from civic_digital_twins.dt_model.axes import ENSEMBLE, Axis
from civic_digital_twins.dt_model.engine.numpybackend import executor as _executor
from civic_digital_twins.dt_model.model.index import DistributionIndex, GenericIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation, EvaluationResult
from civic_digital_twins.dt_model.simulation.handle import EvaluationHandle, _merge_results
from civic_digital_twins.dt_model.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# Minimal model fixtures
# ---------------------------------------------------------------------------


@define("SimpleModel")
class _SimpleModel(Model):
    """Model with one distribution-backed abstract index and one output."""

    @inputs
    class Inputs:
        x: Index

    @outputs
    class Outputs:
        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Double the input index."""
        y = Index("y", inputs.x.node * 2.0)
        return _SimpleModel.Outputs(y=y)


@define("ConstModel")
class _ConstModel(Model):
    """Model with no abstract indexes — all outputs are constant."""

    @inputs
    class Inputs:
        pass

    @outputs
    class Outputs:
        c: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Return a constant output."""
        c = Index("c", 42.0)
        return _ConstModel.Outputs(c=c)


def _make_simple() -> tuple[Index, _SimpleModel]:
    x = DistributionIndex("x", stats.norm, {"loc": 5.0, "scale": 1.0})
    model = _SimpleModel(inputs=_SimpleModel.Inputs(x=x))
    return x, model


# ---------------------------------------------------------------------------
# evaluate_incremental — basic creation
# ---------------------------------------------------------------------------


def test_evaluate_incremental_returns_handle() -> None:
    """evaluate_incremental returns an EvaluationHandle."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 50)
    assert isinstance(handle, EvaluationHandle)


def test_handle_result_is_evaluation_result() -> None:
    """handle.result is an EvaluationResult with the correct shape."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 50)
    result = handle.result
    arr = result[model.outputs.y]
    # shape should be (50,) — one ENSEMBLE axis of size 50
    assert arr.shape == (50,)


def test_handle_result_weights_sum_to_one() -> None:
    """Initial result weights are uniform and sum to 1."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 40)
    w = handle.result.weights
    assert w.shape == (40,)
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(w, np.full(40, 1.0 / 40))


def test_evaluate_incremental_reproducible_with_seed() -> None:
    """The same RNG seed produces the same first result."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    h1 = EvaluationHandle.evaluate(ev, 30, rng=rng1)
    h2 = EvaluationHandle.evaluate(ev, 30, rng=rng2)
    np.testing.assert_array_equal(h1.result[model.outputs.y], h2.result[model.outputs.y])


# ---------------------------------------------------------------------------
# EvaluationHandle.extend — ensemble extension
# ---------------------------------------------------------------------------


def test_extend_increases_ensemble_size() -> None:
    """extend(n) grows the ensemble from S to S+n scenarios."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 30)
    result = handle.extend(20)
    arr = result[model.outputs.y]
    assert arr.shape == (50,)


def test_extend_updates_handle_result() -> None:
    """handle.result is updated after extend()."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 30)
    extended = handle.extend(20)
    assert handle.result is extended


def test_extend_weights_renormalized() -> None:
    """After extend(), weights are uniform over the combined ensemble."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 30)
    handle.extend(20)
    w = handle.result.weights
    assert w.shape == (50,)
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(w, np.full(50, 1.0 / 50))


def test_merge_preserves_nonuniform_weights() -> None:
    """_merge_results uses size-proportional mixture, preserving non-uniform weights.

    merged[i] = w1[i] * S1/(S1+S2)  for i in r1
    merged[j] = w2[j] * S2/(S1+S2)  for j in r2
    """
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    plan = ev.build_plan()

    S1, S2 = 4, 2
    ax1 = Axis("ens_a", ENSEMBLE)
    ax2 = Axis("ens_b", ENSEMBLE)
    w1 = np.array([0.4, 0.4, 0.1, 0.1])  # non-uniform, sums to 1
    w2 = np.array([0.7, 0.3])  # non-uniform, sums to 1

    def _make(ax, sz, w):
        values = {idx.node: np.zeros(sz) for idx in plan.nodes_of_interest}
        state = _executor.State(values)
        return EvaluationResult(state, AxisLayout([(ax, sz)]), {}, factorized_weights={ax: w})

    r1 = _make(ax1, S1, w1)
    r2 = _make(ax2, S2, w2)
    merged = _merge_results(r1, r2, plan)

    alpha = S1 / (S1 + S2)
    expected = np.concatenate([w1 * alpha, w2 * (1.0 - alpha)])
    np.testing.assert_allclose(merged.weights, expected)
    np.testing.assert_allclose(merged.weights.sum(), 1.0)


def test_extend_zero_is_noop() -> None:
    """extend(0) is a no-op — result unchanged."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 30)
    before = handle.result
    returned = handle.extend(0)
    assert returned is before
    assert handle.result is before


def test_extend_negative_is_noop() -> None:
    """extend(-1) is also a no-op."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 30)
    before = handle.result
    returned = handle.extend(-1)
    assert returned is before


def test_multiple_extends_accumulate() -> None:
    """Multiple extend() calls accumulate scenarios correctly."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 10)
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
    ev = Evaluation(Scenario(model))

    # --- Incremental: 30 then 20 ---
    h = EvaluationHandle.evaluate(ev, 30, rng=np.random.default_rng(99))
    h.extend(20)
    incremental_arr = h.result[model.outputs.y]

    # --- Direct: 50 at once from the same seed ---
    direct = EvaluationHandle.evaluate(ev, 50, rng=np.random.default_rng(99))
    direct_arr = direct.result[model.outputs.y]

    assert incremental_arr.shape == direct_arr.shape == (50,)
    np.testing.assert_array_equal(incremental_arr, direct_arr)


def test_extend_reproducible_sequence() -> None:
    """Two handles with the same seed produce the same full sequence."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))

    h1 = EvaluationHandle.evaluate(ev, 20, rng=np.random.default_rng(7))
    h1.extend(30)

    h2 = EvaluationHandle.evaluate(ev, 20, rng=np.random.default_rng(7))
    h2.extend(30)

    np.testing.assert_array_equal(h1.result[model.outputs.y], h2.result[model.outputs.y])


# ---------------------------------------------------------------------------
# EvaluationHandle.extend — singleton nodes (constants)
# ---------------------------------------------------------------------------


def test_extend_constant_node_stays_singleton() -> None:
    """Constant nodes (no ensemble dependency) remain singletons after merge."""
    model = _ConstModel()
    # ConstModel has no abstract indexes — we must pass it via evaluate(), not
    # EvaluationHandle.evaluate (which tries to build a DistributionEnsemble).
    # Instead, test _merge_results directly with a manually built plan + results.
    ev = Evaluation(Scenario(model))
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
# extend() with ensemble_size=1 — stochastic singleton regression (#186)
# ---------------------------------------------------------------------------


def test_extend_with_ensemble_size_1_stochastic() -> None:
    """extend(1) on a size-1 handle must not raise for stochastic nodes (issue #186).

    When ensemble_size=1 every stochastic node produces a shape-(1,) array.
    _merge_results must not treat that as a broadcast constant and assert
    equality — the two draws will almost surely differ.
    """
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 1, rng=np.random.default_rng(0))
    # This must not raise AssertionError.
    result = handle.extend(1)
    assert result[model.outputs.y].shape == (2,)
    assert result[model.inputs.x].shape == (2,)


def test_extend_single_sample_preserves_both_draws() -> None:
    """After extend(1) on a size-1 handle, both stochastic draws are in the result."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    rng = np.random.default_rng(42)
    handle = EvaluationHandle.evaluate(ev, 1, rng=rng)
    first_draw = handle.result[model.inputs.x].copy()
    handle.extend(1)
    merged_x = handle.result[model.inputs.x]
    assert merged_x.shape == (2,)
    # The first draw must still be present in the merged array.
    np.testing.assert_array_equal(merged_x[0:1], first_draw)


# ---------------------------------------------------------------------------
# extra_parameters — parameter-grid extension
# ---------------------------------------------------------------------------


def _make_param_handle(
    param_vals: np.ndarray,
    ensemble_size: int,
    seed: int = 0,
) -> tuple[Index, Any, Evaluation, EvaluationHandle]:
    """Build a test model with one distribution index and one PARAMETER sweep."""
    from scipy import stats

    x2 = DistributionIndex("x2", stats.norm, {"loc": 0.0, "scale": 1.0})
    speed = Index("speed", 1.0)

    class _SPM(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index
            speed: Index

        @outputs
        class Outputs:
            y: Index

        def __init__(self, x: Index, s: Index) -> None:
            y = Index("y", x.node + s.node)
            super().__init__("SPM", inputs=_SPM.Inputs(x=x, speed=s), outputs=_SPM.Outputs(y=y))

    model = _SPM(x2, speed)
    ev = Evaluation(Scenario(model))
    params: dict[GenericIndex, np.ndarray] = {speed: param_vals}
    handle = EvaluationHandle.evaluate(ev, ensemble_size, parameters=params, rng=np.random.default_rng(seed))
    return speed, model, ev, handle


def test_extend_extra_parameters_param_only() -> None:
    """extend(extra_parameters=) extends the PARAMETER axis, keeping the same ensemble."""
    speed, model, ev, handle = _make_param_handle(np.array([1.0, 2.0, 3.0]), ensemble_size=20)
    arr = handle.result[model.outputs.y]
    assert arr.shape == (3, 20)

    handle.extend(extra_parameters={speed: np.array([4.0, 5.0])})
    arr2 = handle.result[model.outputs.y]
    assert arr2.shape == (5, 20)


def test_extend_extra_parameters_combined() -> None:
    """extend(N, extra_parameters=) grows both axes simultaneously."""
    speed, model, ev, handle = _make_param_handle(np.array([1.0, 2.0, 3.0]), ensemble_size=20)
    handle.extend(10, extra_parameters={speed: np.array([4.0, 5.0])})
    arr = handle.result[model.outputs.y]
    assert arr.shape == (5, 30)


def test_extend_extra_parameters_reproducible() -> None:
    """extend(extra_parameters=) matches a single evaluate_incremental with all params."""
    from scipy import stats

    x2 = DistributionIndex("x2", stats.norm, {"loc": 0.0, "scale": 1.0})

    speed = Index("speed", 1.0)

    class _SPM(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index
            speed: Index

        @outputs
        class Outputs:
            y: Index

        def __init__(self, x: Index, s: Index) -> None:
            y = Index("y", x.node + s.node)
            super().__init__("SPM", inputs=_SPM.Inputs(x=x, speed=s), outputs=_SPM.Outputs(y=y))

    model = _SPM(x2, speed)
    ev = Evaluation(Scenario(model))
    all_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    seed = 77

    # Incremental path: 3 values + extend with 2 more.
    h_inc = EvaluationHandle.evaluate(ev, 30, parameters={speed: all_vals[:3]}, rng=np.random.default_rng(seed))
    h_inc.extend(extra_parameters={speed: all_vals[3:]})
    inc_arr = h_inc.result[model.outputs.y]

    # Direct path: all 5 values at once, same seed.
    h_dir = EvaluationHandle.evaluate(ev, 30, parameters={speed: all_vals}, rng=np.random.default_rng(seed))
    dir_arr = h_dir.result[model.outputs.y]

    assert inc_arr.shape == dir_arr.shape == (5, 30)
    np.testing.assert_array_equal(inc_arr, dir_arr)


def test_extend_extra_parameters_multiple_params() -> None:
    """extend(extra_parameters=) with multiple keys extends each axis in turn."""
    from scipy import stats

    x2 = DistributionIndex("x2", stats.norm, {"loc": 0.0, "scale": 1.0})
    speed = Index("speed", 1.0)
    temp = Index("temp", 10.0)

    class _TwoParam(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index
            speed: Index
            temp: Index

        @outputs
        class Outputs:
            y: Index

        def __init__(self, x: Index, s: Index, t: Index) -> None:
            y = Index("y", x.node + s.node + t.node)
            super().__init__("TP", inputs=_TwoParam.Inputs(x=x, speed=s, temp=t), outputs=_TwoParam.Outputs(y=y))

    model = _TwoParam(x2, speed, temp)
    ev = Evaluation(Scenario(model))
    params: dict[GenericIndex, np.ndarray] = {speed: np.array([1.0, 2.0]), temp: np.array([10.0, 20.0])}
    handle = EvaluationHandle.evaluate(ev, 15, parameters=params, rng=np.random.default_rng(0))
    assert handle.result[model.outputs.y].shape == (2, 2, 15)

    handle.extend(extra_parameters={speed: np.array([3.0, 4.0]), temp: np.array([30.0])})
    assert handle.result[model.outputs.y].shape == (4, 3, 15)


def test_extend_extra_parameters_unknown_index_raises() -> None:
    """extend(extra_parameters=) raises ValueError for an index not in the original parameters."""
    x, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 20)
    with pytest.raises(ValueError, match="not in the original parameters"):
        handle.extend(extra_parameters={x: np.array([1.0])})


# ---------------------------------------------------------------------------
# evaluate_incremental with PARAMETER axes
# ---------------------------------------------------------------------------


def test_evaluate_incremental_with_parameters() -> None:
    """evaluate_incremental respects PARAMETER axes; extend preserves them."""

    # Build a model with the parameter as a PARAMETER axis index.
    class _ParamModel(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index
            speed: Index

        @outputs
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
    ev2 = Evaluation(Scenario(pm))

    params: dict[GenericIndex, np.ndarray] = {speed: np.array([1.0, 2.0, 3.0])}
    handle = EvaluationHandle.evaluate(ev2, 20, parameters=params)
    arr = handle.result[pm.outputs.y]
    # shape: (3, 20) — 3 PARAMETER values × 20 ensemble scenarios
    assert arr.shape == (3, 20)

    handle.extend(10)
    arr2 = handle.result[pm.outputs.y]
    # After extend: (3, 30)
    assert arr2.shape == (3, 30)


# ---------------------------------------------------------------------------
# _merge_results error paths
# ---------------------------------------------------------------------------


def _make_fake_result(
    plan,
    ens_axes: tuple[Axis, ...],
    ens_sizes: tuple[int, ...],
) -> EvaluationResult:
    """Build a minimal EvaluationResult with the given ENSEMBLE axes (no real evaluation)."""
    layout = AxisLayout.build(ensemble=list(zip(ens_axes, ens_sizes)))
    factorized_weights: dict[Axis, np.ndarray] = {ax: np.full(sz, 1.0 / sz) for ax, sz in zip(ens_axes, ens_sizes)}
    # Populate values for every node of interest with a zero array of the right shape.
    values: dict = {}
    for idx in plan.nodes_of_interest:
        values[idx.node] = np.zeros(ens_sizes)
    state = _executor.State(values)
    return EvaluationResult(state, layout, {}, factorized_weights=factorized_weights)


def test_merge_results_multi_axis_no_name_raises() -> None:
    """_merge_results raises ValueError for multi-axis results when merge_axis_name is absent."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    plan = ev.build_plan()
    ax1 = Axis("ens1", ENSEMBLE)
    ax2 = Axis("ens2", ENSEMBLE)
    r_multi = _make_fake_result(plan, (ax1, ax2), (2, 3))
    r_multi2 = _make_fake_result(plan, (Axis("ens1", ENSEMBLE), Axis("ens2", ENSEMBLE)), (4, 3))
    with pytest.raises(ValueError, match="multiple ENSEMBLE axes"):
        _merge_results(r_multi, r_multi2, plan)


def test_merge_results_multi_axis_concat() -> None:
    """_merge_results concatenates correctly along the named ENSEMBLE axis."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    plan = ev.build_plan()

    r1 = _make_fake_result(plan, (Axis("ens1", ENSEMBLE), Axis("ens2", ENSEMBLE)), (2, 3))
    r2 = _make_fake_result(plan, (Axis("ens1", ENSEMBLE), Axis("ens2", ENSEMBLE)), (4, 3))
    merged = _merge_results(r1, r2, plan, merge_axis_name="ens1")

    merged_ens1 = next(ax for ax in merged.layout.axes if ax.name == "ens1")
    merged_ens2 = next(ax for ax in merged.layout.axes if ax.name == "ens2")
    assert merged.layout.size_of(merged_ens1) == 6
    assert merged.layout.size_of(merged_ens2) == 3

    for noi in plan.nodes_of_interest:
        assert np.asarray(merged._state.values[noi.node]).shape == (6, 3)

    # Growing axis: proportional mixture; fixed axis: unchanged weights from r1.
    alpha = 2 / 6
    expected_ens1_w = np.concatenate([np.full(2, 1.0 / 2) * alpha, np.full(4, 1.0 / 4) * (1 - alpha)])
    np.testing.assert_allclose(merged._factorized_weights[merged_ens1], expected_ens1_w)
    np.testing.assert_allclose(merged._factorized_weights[merged_ens2], np.full(3, 1.0 / 3))


def test_merge_results_multi_axis_fixed_mismatch_raises() -> None:
    """_merge_results raises ValueError when the fixed ENSEMBLE axis sizes differ."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    plan = ev.build_plan()

    r1 = _make_fake_result(plan, (Axis("ens1", ENSEMBLE), Axis("ens2", ENSEMBLE)), (2, 3))
    r2 = _make_fake_result(plan, (Axis("ens1", ENSEMBLE), Axis("ens2", ENSEMBLE)), (4, 5))
    with pytest.raises(ValueError, match="fixed ENSEMBLE axis"):
        _merge_results(r1, r2, plan, merge_axis_name="ens1")


def test_merge_results_parameter_layout_mismatch_raises() -> None:
    """_merge_results raises ValueError when PARAMETER axis layouts differ."""

    class _PM(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index
            p: Index

        @outputs
        class Outputs:
            y: Index

        def __init__(self, x: Index, p: Index) -> None:
            y = Index("y", x.node + p.node)
            super().__init__("PM", inputs=_PM.Inputs(x=x, p=p), outputs=_PM.Outputs(y=y))

    x2 = DistributionIndex("x2", stats.norm, {"loc": 0.0, "scale": 1.0})
    p2 = Index("p2", 1.0)
    pm = _PM(x2, p2)
    scenario = Scenario(pm)
    ev = Evaluation(scenario)
    plan = ev.build_plan()

    ens = DistributionEnsemble(scenario, size=10, rng=np.random.default_rng(0))
    params_2: dict[GenericIndex, np.ndarray] = {p2: np.array([1.0, 2.0])}
    params_3: dict[GenericIndex, np.ndarray] = {p2: np.array([1.0, 2.0, 3.0])}
    r1 = ev.execute_plan(plan, ens, parameters=params_2)
    r2 = ev.execute_plan(plan, ens, parameters=params_3)
    with pytest.raises(ValueError, match="PARAMETER axis layouts"):
        _merge_results(r1, r2, plan)


def test_evaluation_handle_result_raises_when_none() -> None:
    """EvaluationHandle.result raises RuntimeError when _result is None."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    plan = ev.build_plan()
    # Construct a handle with result=None directly — the async path does this.
    handle = EvaluationHandle(
        evaluation=ev,
        plan=plan,
        result=None,
        rng=np.random.default_rng(),
        parameters={},
        functions=None,
        backend=_executor.NumpyBackend,
    )
    with pytest.raises(RuntimeError, match="not yet available"):
        _ = handle.result


# ---------------------------------------------------------------------------
# FrozenEnsemble / BatchDrawable behaviour
# ---------------------------------------------------------------------------


def test_frozen_ensemble_draw_batch_raises() -> None:
    """FrozenEnsemble.draw_batch raises TypeError with a descriptive message."""
    from civic_digital_twins.dt_model.simulation.ensemble import FrozenEnsemble  # noqa: PLC0415

    ax = Axis("_ensemble", ENSEMBLE)
    fe = FrozenEnsemble((ax,), (np.array([0.5, 0.5]),), {})
    with pytest.raises(TypeError, match="cannot draw new samples"):
        fe.draw_batch(5, np.random.default_rng())


def test_distribution_ensemble_draw_batch_axis_raises() -> None:
    """DistributionEnsemble.draw_batch raises ValueError when axis= is not None."""
    x, model = _make_simple()
    de = DistributionEnsemble(Scenario(model), 10, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="single ENSEMBLE axis"):
        de.draw_batch(5, np.random.default_rng(), axis="unc")


def test_frozen_ensemble_concat_along() -> None:
    """FrozenEnsemble.concat_along appends samples along the named axis correctly."""
    from civic_digital_twins.dt_model.simulation.ensemble import FrozenEnsemble  # noqa: PLC0415

    ax1 = Axis("unc", ENSEMBLE)
    ax2 = Axis("default", ENSEMBLE)
    idx_unc = Index("x_unc", 1.0)
    idx_def = Index("x_def", 2.0)
    # idx_unc has shape (2, 1): assigned to ax1 (axis 0, size 2), singleton on ax2.
    # idx_def has shape (1, 3): singleton on ax1, assigned to ax2 (axis 1, size 3).
    fe = FrozenEnsemble(
        (ax1, ax2),
        (np.array([0.5, 0.5]), np.array([1 / 3, 1 / 3, 1 / 3])),
        {idx_unc: np.ones((2, 1)), idx_def: np.ones((1, 3))},
    )
    # other provides fresh ax1 samples (size 1); idx_unc shape (1,).
    other = FrozenEnsemble(
        (Axis("unc", ENSEMBLE),),
        (np.array([1.0]),),
        {idx_unc: np.array([99.0])},
    )
    merged = fe.concat_along("unc", other)
    # unc axis grows from 2 → 3; default axis stays at 3.
    assert merged.ensemble_weights[0].size == 3
    # idx_unc was (2,1) → concat along ax0 with (1,1) → (3,1)
    assert list(merged._cached_assignments[idx_unc].shape) == [3, 1]
    # idx_def is singleton at ax0 (shape[0]==1) → carried forward unchanged.
    assert list(merged._cached_assignments[idx_def].shape) == [1, 3]


def test_frozen_ensemble_with_replaced_axis() -> None:
    """FrozenEnsemble.with_replaced_axis replaces one axis with fresh samples."""
    from civic_digital_twins.dt_model.simulation.ensemble import FrozenEnsemble  # noqa: PLC0415

    ax1 = Axis("unc", ENSEMBLE)
    ax2 = Axis("default", ENSEMBLE)
    idx_unc = Index("x_unc", 1.0)
    idx_def = Index("x_def", 2.0)
    # idx_unc has shape (2, 1): assigned to ax1 (axis 0, size 2), singleton on ax2.
    # idx_def has shape (1, 3): singleton on ax1, assigned to ax2 (axis 1, size 3).
    fe = FrozenEnsemble(
        (ax1, ax2),
        (np.array([0.5, 0.5]), np.array([1 / 3, 1 / 3, 1 / 3])),
        {idx_unc: np.ones((2, 1)), idx_def: np.ones((1, 3))},
    )
    other = FrozenEnsemble(
        (Axis("unc", ENSEMBLE),),
        (np.array([1.0]),),
        {idx_unc: np.array([42.0])},
    )
    replaced = fe.with_replaced_axis("unc", other)
    # unc axis replaced by new size=1; default axis stays at 3.
    assert replaced.ensemble_weights[0].shape == (1,)
    # idx_unc: other has shape (1,) → reshaped to (1, 1) for the 2-axis result.
    assert replaced._cached_assignments[idx_unc].shape == (1, 1)
    # idx_def: singleton at ax0, carried forward unchanged.
    assert replaced._cached_assignments[idx_def].shape == (1, 3)


# ---------------------------------------------------------------------------
# _merge_results error paths for missing named axis
# ---------------------------------------------------------------------------


def test_merge_results_missing_named_axis_in_r1() -> None:
    """_merge_results raises ValueError when merge_axis_name is not in r1."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    plan = ev.build_plan()

    r1 = _make_fake_result(plan, (Axis("ens_a", ENSEMBLE), Axis("ens_b", ENSEMBLE)), (2, 3))
    r2 = _make_fake_result(plan, (Axis("ens_a", ENSEMBLE), Axis("ens_b", ENSEMBLE)), (4, 3))
    with pytest.raises(ValueError, match="no ENSEMBLE axis named 'nonexistent' in r1"):
        _merge_results(r1, r2, plan, merge_axis_name="nonexistent")


def test_merge_results_missing_named_axis_in_r2() -> None:
    """_merge_results raises ValueError when merge_axis_name is not in r2."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    plan = ev.build_plan()

    r1 = _make_fake_result(plan, (Axis("ens_a", ENSEMBLE), Axis("ens_b", ENSEMBLE)), (2, 3))
    # r2 only has "ens_a", so "ens_b" is absent in r2.
    r2 = _make_fake_result(plan, (Axis("ens_a", ENSEMBLE),), (4,))
    with pytest.raises(ValueError, match="no ENSEMBLE axis named 'ens_b' in r2"):
        _merge_results(r1, r2, plan, merge_axis_name="ens_b")


# ---------------------------------------------------------------------------
# _merge_results_param_extend error paths
# ---------------------------------------------------------------------------


def test_merge_results_param_extend_multi_ensemble_raises() -> None:
    """_merge_results_param_extend raises ValueError when a result has multiple ENSEMBLE axes."""
    from civic_digital_twins.dt_model.simulation.handle import _merge_results_param_extend  # noqa: PLC0415

    speed, model, ev, handle = _make_param_handle(np.array([1.0, 2.0]), ensemble_size=10)
    plan = ev.build_plan()
    r1 = handle.result

    # Build a result with two ENSEMBLE axes.
    ax_a = Axis("ens_a", ENSEMBLE)
    ax_b = Axis("ens_b", ENSEMBLE)
    r2_multi = _make_fake_result(plan, (ax_a, ax_b), (2, 3))
    with pytest.raises(ValueError, match="exactly one ENSEMBLE axis"):
        _merge_results_param_extend(r2_multi, r1, plan, speed)


def test_merge_results_param_extend_ensemble_size_mismatch_raises() -> None:
    """_merge_results_param_extend raises ValueError when ENSEMBLE sizes differ."""
    from civic_digital_twins.dt_model.simulation.handle import _merge_results_param_extend  # noqa: PLC0415

    speed, model, ev, handle = _make_param_handle(np.array([1.0, 2.0]), ensemble_size=10)
    plan = ev.build_plan()
    r1 = handle.result

    # Build r2 with a different ensemble size (10+1=11 vs 10).
    r2 = ev.execute_plan(
        plan,
        DistributionEnsemble(Scenario(model), 11, rng=np.random.default_rng(99)),
        parameters={speed: np.array([1.0, 2.0])},
    )
    with pytest.raises(ValueError, match="identical ENSEMBLE sizes"):
        _merge_results_param_extend(r1, r2, plan, speed)


def test_merge_results_param_extend_missing_param_axis_raises() -> None:
    """_merge_results_param_extend raises ValueError when r1 has no PARAMETER axis for param_idx."""
    from civic_digital_twins.dt_model.simulation.handle import _merge_results_param_extend  # noqa: PLC0415

    x, model = _make_simple()
    scenario = Scenario(model)
    ev = Evaluation(scenario)
    plan = ev.build_plan()

    # Both results have only one ENSEMBLE axis and no PARAMETER axis for speed.
    ens = DistributionEnsemble(scenario, 5, rng=np.random.default_rng(0))
    r1 = ev.execute_plan(plan, ens)
    r2 = ev.execute_plan(plan, ens)

    # speed is not a PARAMETER axis in r1 → raises.
    speed = Index("speed", 1.0)
    with pytest.raises(ValueError, match="no PARAMETER axis named"):
        _merge_results_param_extend(r1, r2, plan, speed)


# ---------------------------------------------------------------------------
# EvaluationHandle.extend() validation paths
# ---------------------------------------------------------------------------


def test_extend_ensemble_size_and_extra_ensemble_mutually_exclusive() -> None:
    """extend() raises ValueError when both ensemble_size > 0 and extra_ensemble are supplied."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    handle = EvaluationHandle.evaluate(ev, 20)
    with pytest.raises(ValueError, match="mutually exclusive"):
        handle.extend(10, extra_ensemble={"_ensemble": 5})


def test_extend_extra_parameters_without_stored_ensemble() -> None:
    """extend(extra_parameters=) works without a stored ensemble by reconstructing from result state."""
    speed, model, ev, _ = _make_param_handle(np.array([1.0, 2.0]), ensemble_size=10)
    plan = ev.build_plan()
    ens = DistributionEnsemble(Scenario(model), 10, rng=np.random.default_rng(0))
    r = ev.execute_plan(plan, ens, parameters={speed: np.array([1.0, 2.0])})
    handle = EvaluationHandle(
        evaluation=ev,
        plan=plan,
        result=r,
        rng=np.random.default_rng(),
        parameters={speed: np.array([1.0, 2.0])},
        ensemble=None,  # no stored ensemble — reconstructed from result state
        functions=None,
        backend=_executor.NumpyBackend,
    )
    result = handle.extend(extra_parameters={speed: np.array([3.0])})
    assert result[model.outputs.y].shape == (3, 10)


# ---------------------------------------------------------------------------
# _merge_results — growing axis at different positions raises
# ---------------------------------------------------------------------------


def test_merge_results_growing_axis_at_different_position_raises() -> None:
    """_merge_results raises ValueError when the named growing axis is at different positions."""
    _, model = _make_simple()
    ev = Evaluation(Scenario(model))
    plan = ev.build_plan()

    # r1: (ens1 at dim 0, ens2 at dim 1)  r2: (ens2 at dim 0, ens1 at dim 1)
    ax_e1_r1 = Axis("ens1", ENSEMBLE)
    ax_e2_r1 = Axis("ens2", ENSEMBLE)
    ax_e1_r2 = Axis("ens1", ENSEMBLE)
    ax_e2_r2 = Axis("ens2", ENSEMBLE)

    values_r1: dict = {}
    values_r2: dict = {}
    for idx in plan.nodes_of_interest:
        values_r1[idx.node] = np.zeros((2, 3))
        values_r2[idx.node] = np.zeros((3, 2))

    from civic_digital_twins.dt_model.engine.numpybackend import executor as _ex  # noqa: PLC0415

    r1 = EvaluationResult(
        _ex.State(values_r1),
        AxisLayout.build(ensemble=[(ax_e1_r1, 2), (ax_e2_r1, 3)]),
        {},
        factorized_weights={ax_e1_r1: np.full(2, 0.5), ax_e2_r1: np.full(3, 1 / 3)},
    )
    r2 = EvaluationResult(
        _ex.State(values_r2),
        AxisLayout.build(ensemble=[(ax_e2_r2, 3), (ax_e1_r2, 2)]),
        {},
        factorized_weights={ax_e2_r2: np.full(3, 1 / 3), ax_e1_r2: np.full(2, 0.5)},
    )
    # ens1 is at dim 0 in r1 but dim 1 in r2.
    with pytest.raises(ValueError, match="dim 0 in r1 but dim 1 in r2"):
        _merge_results(r1, r2, plan, merge_axis_name="ens1")


# ---------------------------------------------------------------------------
# _merge_results_param_extend — additional error paths
# ---------------------------------------------------------------------------


def test_merge_results_param_extend_param_axis_missing_in_r2() -> None:
    """_merge_results_param_extend raises ValueError when r2 lacks the growing PARAMETER axis."""
    from civic_digital_twins.dt_model.axes import PARAMETER  # noqa: PLC0415
    from civic_digital_twins.dt_model.engine.numpybackend import executor as _ex  # noqa: PLC0415
    from civic_digital_twins.dt_model.simulation.handle import _merge_results_param_extend  # noqa: PLC0415

    speed, model, ev, handle = _make_param_handle(np.array([1.0, 2.0]), ensemble_size=5)
    plan = ev.build_plan()
    r1 = handle.result

    # Build r2 with the ENSEMBLE axis at the same position as r1 but WITHOUT the speed PARAMETER axis.
    # r1 has PARAMETER(speed) at dim 0, ENSEMBLE at dim 1 → r2 needs ENSEMBLE at dim 1 too.
    ax_fake_param = Axis("speed2_fake", PARAMETER)
    ax_ens = Axis("_ensemble", ENSEMBLE)
    values: dict = {}
    for idx in plan.nodes_of_interest:
        values[idx.node] = np.zeros((2, 5))
    r2 = EvaluationResult(
        _ex.State(values),
        AxisLayout.build(parameters=[(ax_fake_param, 2)], ensemble=[(ax_ens, 5)]),  # ENSEMBLE at dim 1, matching r1
        {},
        factorized_weights={ax_fake_param: np.full(2, 0.5), ax_ens: np.full(5, 0.2)},
    )
    # r2 has no PARAMETER axis named "speed" → raises
    with pytest.raises(ValueError, match="no PARAMETER axis named"):
        _merge_results_param_extend(r1, r2, plan, speed)


def test_merge_results_param_extend_ensemble_pos_mismatch_raises() -> None:
    """_merge_results_param_extend raises ValueError when ENSEMBLE axis positions differ."""
    from civic_digital_twins.dt_model.engine.numpybackend import executor as _ex  # noqa: PLC0415
    from civic_digital_twins.dt_model.simulation.handle import _merge_results_param_extend  # noqa: PLC0415

    speed, model, ev, handle = _make_param_handle(np.array([1.0, 2.0]), ensemble_size=5)
    plan = ev.build_plan()
    r1 = handle.result
    # r1: PARAMETER(speed) at dim 0, ENSEMBLE at dim 1 → ens_pos=1

    # Build r2 with ENSEMBLE at dim 0.  Under the canonical role ordering
    # (PARAMETER before ENSEMBLE, enforced by AxisLayout) a position mismatch
    # arises from a differing PARAMETER count: r2 has no PARAMETER axis, so
    # its single ENSEMBLE axis sits at dim 0 versus r1's dim 1.
    ax_ens = Axis("_ensemble", ENSEMBLE)
    values: dict = {}
    for idx in plan.nodes_of_interest:
        values[idx.node] = np.zeros((5,))  # shape (5_ens,)
    r2 = EvaluationResult(
        _ex.State(values),
        AxisLayout.build(ensemble=[(ax_ens, 5)]),  # ENSEMBLE at dim 0 (r1 has it at dim 1)
        {},
        factorized_weights={ax_ens: np.full(5, 0.2)},
    )
    with pytest.raises(ValueError, match="ENSEMBLE axis position mismatch"):
        _merge_results_param_extend(r1, r2, plan, speed)


def test_merge_results_param_extend_fixed_param_layout_differs_raises() -> None:
    """_merge_results_param_extend raises ValueError when fixed PARAMETER axis layouts differ."""
    from civic_digital_twins.dt_model.simulation.handle import _merge_results_param_extend  # noqa: PLC0415

    # Build a model with two parameter indexes: speed and temp.
    x2 = DistributionIndex("x2", stats.norm, {"loc": 0.0, "scale": 1.0})
    speed2 = Index("speed2", 1.0)
    temp2 = Index("temp2", 10.0)

    class _TwoP(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index
            speed: Index
            temp: Index

        @outputs
        class Outputs:
            y: Index

        def __init__(self, x: Index, s: Index, t: Index) -> None:
            y = Index("y", x.node + s.node + t.node)
            super().__init__("TP2", inputs=_TwoP.Inputs(x=x, speed=s, temp=t), outputs=_TwoP.Outputs(y=y))

    model2 = _TwoP(x2, speed2, temp2)
    scenario2 = Scenario(model2)
    ev2 = Evaluation(scenario2)
    plan2 = ev2.build_plan()
    ens = DistributionEnsemble(scenario2, 5, rng=np.random.default_rng(0))

    # r1 has speed2(2) × temp2(2) × ens(5)
    r1 = ev2.execute_plan(
        plan2,
        ens,
        parameters={speed2: np.array([1.0, 2.0]), temp2: np.array([10.0, 20.0])},
    )
    # r2 has speed2(3) × temp2(3) × ens(5) — fixed axis temp2 has different size.
    r2 = ev2.execute_plan(
        plan2,
        ens,
        parameters={speed2: np.array([3.0, 4.0, 5.0]), temp2: np.array([30.0, 40.0, 50.0])},
    )
    with pytest.raises(ValueError, match="fixed PARAMETER axis layouts differ"):
        _merge_results_param_extend(r1, r2, plan2, speed2)


# ---------------------------------------------------------------------------
# PartitionedEnsemble-backed EvaluationHandle — extend via extra_ensemble
# ---------------------------------------------------------------------------


def _make_pe_handle() -> tuple[Any, Any, Evaluation, EvaluationHandle]:
    """Build a 2-index model and a PartitionedEnsemble-backed EvaluationHandle."""
    from civic_digital_twins.dt_model.simulation.ensemble import (  # noqa: PLC0415
        EnsembleAxisSpec,
        FrozenEnsemble,
        PartitionedEnsemble,
    )

    a = DistributionIndex("a", stats.norm, {"loc": 0.0, "scale": 1.0})
    b = DistributionIndex("b", stats.norm, {"loc": 1.0, "scale": 0.5})

    class _ABModel(Model, legacy=True):
        @inputs
        class Inputs:
            a: Index
            b: Index

        @outputs
        class Outputs:
            y: Index

        def __init__(self, _a: Index, _b: Index) -> None:
            y = Index("y", _a.node + _b.node)
            super().__init__("ABModel", inputs=_ABModel.Inputs(a=_a, b=_b), outputs=_ABModel.Outputs(y=y))

    model = _ABModel(a, b)
    scenario = Scenario(model)
    ev = Evaluation(scenario)
    plan = ev.build_plan()

    pe = PartitionedEnsemble(
        scenario,
        axes=[EnsembleAxisSpec("unc_a", indexes=[a], size=3)],
        default_axis=EnsembleAxisSpec("unc_b", indexes=[], size=4),
        rng=np.random.default_rng(0),
    )
    assignments = dict(pe.assignments())
    multi_frozen = FrozenEnsemble(
        pe.ensemble_axes,
        pe.ensemble_weights,
        {a: assignments[a], b: assignments[b]},
    )
    result = ev.execute_plan(plan, multi_frozen)

    handle = EvaluationHandle(
        evaluation=ev,
        plan=plan,
        result=result,
        rng=np.random.default_rng(42),
        parameters={},
        ensemble=multi_frozen,
        ensemble_recipe=pe,
        functions=None,
        backend=_executor.NumpyBackend,
    )
    return a, b, ev, handle


def test_extend_extra_ensemble_grows_named_axis() -> None:
    """extend(extra_ensemble=) grows the named axis and updates the result shape."""
    a, b, ev, handle = _make_pe_handle()
    model = ev._scenario.model
    # Initial shape: (3, 4) for axes unc_a × unc_b
    assert handle.result[model.outputs.y].shape == (3, 4)

    handle.extend(extra_ensemble={"unc_a": 2})
    # unc_a grows from 3 → 5; unc_b stays at 4.
    assert handle.result[model.outputs.y].shape == (5, 4)


def test_extend_extra_ensemble_updates_frozen_ensemble() -> None:
    """extend(extra_ensemble=) updates the stored frozen ensemble for unc_a axis."""
    a, b, ev, handle = _make_pe_handle()
    assert handle._ensemble is not None
    initial_size = handle._ensemble.ensemble_weights[0].size
    assert initial_size == 3

    handle.extend(extra_ensemble={"unc_a": 2})
    # unc_a weights should now have size 5.
    assert handle._ensemble is not None
    new_size = handle._ensemble.ensemble_weights[0].size
    assert new_size == 5


def test_extend_ensemble_axis_no_recipe_raises() -> None:
    """_extend_ensemble_axis raises RuntimeError when the handle has no ensemble_recipe."""
    _, model = _make_simple()
    scenario = Scenario(model)
    ev = Evaluation(scenario)
    plan = ev.build_plan()
    ens = DistributionEnsemble(scenario, 10, rng=np.random.default_rng(0))
    result = ev.execute_plan(plan, ens)
    handle = EvaluationHandle(
        evaluation=ev,
        plan=plan,
        result=result,
        rng=np.random.default_rng(1),
        parameters={},
        ensemble_recipe=None,  # no recipe
    )
    with pytest.raises(RuntimeError, match="ensemble_recipe"):
        handle.extend(5)


def test_evaluate_accepts_custom_recipe() -> None:
    """A caller-supplied BatchDrawable recipe is used for both the initial draw and future extends."""
    _, model = _make_simple()
    scenario = Scenario(model)
    ev = Evaluation(scenario)
    recipe = DistributionEnsemble(scenario, 20, rng=np.random.default_rng(0))
    handle = EvaluationHandle.evaluate(ev, 10, ensemble_recipe=recipe)
    assert handle.result[model.outputs.y].shape == (10,)
    # The stored recipe is the one we supplied.
    assert handle._ensemble_recipe is recipe
