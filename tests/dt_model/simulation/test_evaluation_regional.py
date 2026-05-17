# SPDX-License-Identifier: Apache-2.0
"""Tests for build_plan(strategy='regional') and execute_plan with multi-region plans."""

import dataclasses
from collections.abc import Mapping

import numpy as np
import pytest

from civic_digital_twins.dt_model import graph as _graph
from civic_digital_twins.dt_model.engine.numpybackend.executor import NumpyBackend
from civic_digital_twins.dt_model.model.axis import ENSEMBLE, Axis
from civic_digital_twins.dt_model.model.index import CategoricalIndex, DistributionIndex, GenericIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.model.model_variant import ModelVariant
from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble, WeightedScenario
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.simulation.plan import EvaluationPlan, Region, RegionGuard

# ---------------------------------------------------------------------------
# Shared model fixtures (same as test_model_variant_evaluation.py)
# ---------------------------------------------------------------------------

_CAPACITY_VALUE = 100.0


class _BikeModel(Model):
    @dataclasses.dataclass
    class Inputs:
        capacity: Index

    @dataclasses.dataclass
    class Outputs:
        throughput: Index
        emissions: Index

    def __init__(self, capacity: Index) -> None:
        throughput = Index("throughput", capacity.node * 1.0)
        emissions = Index("emissions", 0.0)
        super().__init__(
            "BikeModel",
            inputs=_BikeModel.Inputs(capacity=capacity),
            outputs=_BikeModel.Outputs(throughput=throughput, emissions=emissions),
        )


class _TrainModel(Model):
    @dataclasses.dataclass
    class Inputs:
        capacity: Index

    @dataclasses.dataclass
    class Outputs:
        throughput: Index
        emissions: Index

    def __init__(self, capacity: Index) -> None:
        throughput = Index("throughput", capacity.node * 10.0)
        emissions = Index("emissions", 50.0)
        super().__init__(
            "TrainModel",
            inputs=_TrainModel.Inputs(capacity=capacity),
            outputs=_TrainModel.Outputs(throughput=throughput, emissions=emissions),
        )


def _make_mv(mode: CategoricalIndex) -> ModelVariant:
    """Build a Transport ModelVariant with fixed capacity for both branches."""
    cap_bike = Index("capacity", _CAPACITY_VALUE)
    cap_train = Index("capacity", _CAPACITY_VALUE)
    return ModelVariant(
        "Transport",
        {"bike": _BikeModel(cap_bike), "train": _TrainModel(cap_train)},
        selector=mode,
    )


def _make_presence_mv(mode: CategoricalIndex) -> tuple[Index, ModelVariant]:
    """Build a Transport ModelVariant where both sub-models share a presence axis."""
    presence = Index("presence", None)
    mv = ModelVariant(
        "Transport",
        {"bike": _BikeModel(presence), "train": _TrainModel(presence)},
        selector=mode,
    )
    return presence, mv


# ---------------------------------------------------------------------------
# build_plan(strategy='regional') — structural tests
# ---------------------------------------------------------------------------


def test_regional_plan_has_correct_region_count():
    """Regional plan for a 2-branch variant has exactly 4 regions: shared + 2 branches + merge."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    ev = Evaluation(mv)
    plan = ev.build_plan(strategy="regional")
    # 1 shared + 2 branch + 1 merge = 4
    assert len(plan.regions) == 4


def test_regional_plan_is_evaluation_plan():
    """build_plan(strategy='regional') returns an EvaluationPlan instance."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(mv).build_plan(strategy="regional")
    assert isinstance(plan, EvaluationPlan)


def test_regional_plan_regions_are_region_instances():
    """Every region in a regional plan is a Region instance."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(mv).build_plan(strategy="regional")
    for region in plan.regions:
        assert isinstance(region, Region)


def test_regional_plan_shared_region_has_no_guard():
    """The first region (shared) must have guard=None."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(mv).build_plan(strategy="regional")
    assert plan.regions[0].guard is None


def test_regional_plan_branch_regions_have_guards():
    """Middle regions (branches) must carry a RegionGuard."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(mv).build_plan(strategy="regional")
    # regions[1] and regions[2] are branches
    for region in plan.regions[1:-1]:
        assert isinstance(region.guard, RegionGuard)


def test_regional_plan_branch_keys_match_variant():
    """Branch region guard.branch_key values must match the ModelVariant's branch keys."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(mv).build_plan(strategy="regional")
    branch_keys = {r.guard.branch_key for r in plan.regions[1:-1] if r.guard is not None}
    assert branch_keys == {"bike", "train"}


def test_regional_plan_merge_region_has_no_guard():
    """The last region (merge) must have guard=None."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(mv).build_plan(strategy="regional")
    assert plan.regions[-1].guard is None


def test_regional_plan_correct_dependencies():
    """Check DAG dependencies: shared→branches→merge."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(mv).build_plan(strategy="regional")
    deps = plan.dependencies
    # shared (0): no deps
    assert deps[0] == frozenset()
    # branches (1, 2): each depends on shared (0)
    assert deps[1] == frozenset({0})
    assert deps[2] == frozenset({0})
    # merge (3): depends on shared + both branches
    assert deps[3] == frozenset({0, 1, 2})


# ---------------------------------------------------------------------------
# execute_plan — correctness against monolithic baseline
# ---------------------------------------------------------------------------


def _bike_only_scenarios(mode: CategoricalIndex, n: int) -> list[WeightedScenario]:
    return [(1.0 / n, {mode: np.array(["bike"])}) for _ in range(n)]


def _train_only_scenarios(mode: CategoricalIndex, n: int) -> list[WeightedScenario]:
    return [(1.0 / n, {mode: np.array(["train"])}) for _ in range(n)]


def _mixed_scenarios(mode: CategoricalIndex) -> list[WeightedScenario]:
    return [
        (0.25, {mode: np.array(["bike"])}),
        (0.25, {mode: np.array(["train"])}),
        (0.25, {mode: np.array(["bike"])}),
        (0.25, {mode: np.array(["train"])}),
    ]


def test_regional_bike_only_matches_monolithic():
    """Regional plan: bike-only scenarios match monolithic throughput."""
    mode = CategoricalIndex("mode", {"bike": 1.0})
    mv = _make_mv(mode)
    ev = Evaluation(mv)
    scenarios = _bike_only_scenarios(mode, 4)

    mono = ev.evaluate(scenarios, [mv.outputs.throughput])
    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

        scenarios_list = _bike_only_scenarios(mode, 4)
        adapter = _LegacyEnsembleAdapter(scenarios_list, [mode])

    regional_result = ev.execute_plan(regional_plan, adapter)
    assert float(regional_result.expected_value(mv.outputs.throughput)) == pytest.approx(
        float(mono.expected_value(mv.outputs.throughput))
    )


def test_regional_train_only_matches_monolithic():
    """Regional plan: train-only scenarios match monolithic throughput."""
    mode = CategoricalIndex("mode", {"train": 1.0})
    mv = _make_mv(mode)
    ev = Evaluation(mv)

    from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

    scenarios_list = _train_only_scenarios(mode, 4)
    adapter = _LegacyEnsembleAdapter(scenarios_list, [mode])

    mono_result = ev.evaluate(_train_only_scenarios(mode, 4), [mv.outputs.throughput])
    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, adapter)

    assert float(regional_result.expected_value(mv.outputs.throughput)) == pytest.approx(
        float(mono_result.expected_value(mv.outputs.throughput))
    )


def test_regional_mixed_modes_matches_monolithic():
    """Regional plan: mixed bike/train scenarios produce correctly weighted mean."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    ev = Evaluation(mv)

    from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

    scenarios_list = _mixed_scenarios(mode)
    adapter = _LegacyEnsembleAdapter(scenarios_list, [mode])

    mono_result = ev.evaluate(scenarios_list, [mv.outputs.throughput])
    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, adapter)

    assert float(regional_result.expected_value(mv.outputs.throughput)) == pytest.approx(
        float(mono_result.expected_value(mv.outputs.throughput))
    )


def test_regional_emissions_bike_only():
    """Regional plan: bike-only emissions = 0."""
    mode = CategoricalIndex("mode", {"bike": 1.0})
    mv = _make_mv(mode)
    ev = Evaluation(mv)

    from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

    scenarios_list = _bike_only_scenarios(mode, 4)
    adapter = _LegacyEnsembleAdapter(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.emissions], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, adapter)
    assert float(regional_result.expected_value(mv.outputs.emissions)) == pytest.approx(0.0)


def test_regional_emissions_train_only():
    """Regional plan: train-only emissions = 50."""
    mode = CategoricalIndex("mode", {"train": 1.0})
    mv = _make_mv(mode)
    ev = Evaluation(mv)

    from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

    scenarios_list = _train_only_scenarios(mode, 4)
    adapter = _LegacyEnsembleAdapter(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.emissions], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, adapter)
    assert float(regional_result.expected_value(mv.outputs.emissions)) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# execute_plan with PARAMETER axes + regional plan
# ---------------------------------------------------------------------------


def test_regional_plan_with_parameter_axis_bike_only():
    """Regional plan + PARAMETER axis: bike-only throughput = presence * 1."""
    mode = CategoricalIndex("mode", {"bike": 1.0})
    presence, mv = _make_presence_mv(mode)
    ev = Evaluation(mv)
    xs = np.array([100.0, 200.0, 300.0])

    # Single scenario with mode="bike" (presence is the PARAMETER axis)
    from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

    scenarios_list: list[WeightedScenario] = [(1.0, {mode: np.array(["bike"])})]
    adapter = _LegacyEnsembleAdapter(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    result = ev.execute_plan(regional_plan, adapter, parameters={presence: xs})

    assert np.allclose(result.expected_value(mv.outputs.throughput), xs * 1.0)


def test_regional_plan_with_parameter_axis_train_only():
    """Regional plan + PARAMETER axis: train-only throughput = presence * 10."""
    mode = CategoricalIndex("mode", {"train": 1.0})
    presence, mv = _make_presence_mv(mode)
    ev = Evaluation(mv)
    xs = np.array([100.0, 200.0, 300.0])

    from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

    scenarios_list: list[WeightedScenario] = [(1.0, {mode: np.array(["train"])})]
    adapter = _LegacyEnsembleAdapter(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    result = ev.execute_plan(regional_plan, adapter, parameters={presence: xs})

    assert np.allclose(result.expected_value(mv.outputs.throughput), xs * 10.0)


def test_regional_plan_with_parameter_axis_mixed_matches_monolithic():
    """Regional plan + PARAMETER axis: mixed modes weighted mean = presence * 5.5."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    presence, mv = _make_presence_mv(mode)
    ev = Evaluation(mv)
    xs = np.array([100.0, 200.0, 300.0])

    scenarios_list: list[WeightedScenario] = [
        (0.5, {mode: np.array(["bike"])}),
        (0.5, {mode: np.array(["train"])}),
    ]

    from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

    adapter = _LegacyEnsembleAdapter(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, adapter, parameters={presence: xs})
    mono_result = ev.evaluate(scenarios_list, [mv.outputs.throughput], parameters={presence: xs})

    assert np.allclose(
        regional_result.expected_value(mv.outputs.throughput),
        mono_result.expected_value(mv.outputs.throughput),
    )


# ---------------------------------------------------------------------------
# execute_plan raises for a PARAMETER-varying selector
# ---------------------------------------------------------------------------


def test_regional_raises_for_parameter_varying_selector():
    """Regional execution must raise when the selector varies along a PARAMETER axis.

    A selector derived from a PARAMETER index produces a scenario partition
    that differs per parameter combination.  Silently using only position-0
    of the PARAMETER axis for the mask would give wrong results, so the
    executor rejects the case with NotImplementedError.
    """
    presence = Index("presence", None)
    selector = ModelVariant.guards_to_selector([("train", presence.node > 150.0), ("bike", True)])
    cap = Index("capacity", 100.0)
    mv = ModelVariant(
        "Transport",
        {"bike": _BikeModel(cap), "train": _TrainModel(Index("capacity", 100.0))},
        selector=selector,
    )
    ev = Evaluation(mv)
    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")

    from civic_digital_twins.dt_model.simulation.evaluation import _LegacyEnsembleAdapter

    # One dummy scenario (no ENSEMBLE abstract indexes; presence is PARAMETER-only).
    dummy_ens = _LegacyEnsembleAdapter([(1.0, {})], [])
    xs = np.array([100.0, 200.0, 300.0])

    with pytest.raises(NotImplementedError, match="PARAMETER axes"):
        ev.execute_plan(regional_plan, dummy_ens, parameters={presence: xs})


# ---------------------------------------------------------------------------
# build_plan(strategy='regional') raises ValueError for plain Model
# ---------------------------------------------------------------------------


def test_regional_plan_raises_for_plain_model():
    """build_plan(strategy='regional') must raise ValueError when no variant_selector exists."""
    cap = Index("capacity", 100.0)
    plain_model = _BikeModel(cap)
    ev = Evaluation(plain_model)
    with pytest.raises(ValueError, match="No variant_selector found"):
        ev.build_plan(strategy="regional")


# ---------------------------------------------------------------------------
# Monolithic plan still works (regression guard)
# ---------------------------------------------------------------------------


def test_monolithic_plan_still_works_after_regional_changes():
    """Existing monolithic path is unaffected by regional implementation."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    ev = Evaluation(mv)
    scenarios = _mixed_scenarios(mode)
    result = ev.evaluate(scenarios, [mv.outputs.throughput])
    # 0.5 * 100*1 + 0.5 * 100*10 = 550
    assert float(result.expected_value(mv.outputs.throughput)) == pytest.approx(550.0)


# ---------------------------------------------------------------------------
# Regional execution error paths (coverage)
# ---------------------------------------------------------------------------


def test_regional_execute_plan_no_ensemble_raises():
    """execute_plan with a regional plan and no ensemble raises NotImplementedError."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    ev = Evaluation(mv)
    plan = ev.build_plan(strategy="regional")
    with pytest.raises(NotImplementedError, match="multi-axis or absent ensemble"):
        ev.execute_plan(plan, ensemble=None)


# ---------------------------------------------------------------------------
# Branch-local abstract index (covers ens_node override path in _execute_plan)
# ---------------------------------------------------------------------------


def _make_mv_branch_local() -> tuple[Index, Index, "ModelVariant"]:
    """Build a ModelVariant where each branch has its OWN distinct abstract index.

    bike branch  → ``cap_bike`` (abstract)
    train branch → ``cap_train`` (abstract)

    With a regional plan, each cap index node lives exclusively in its branch
    region, so the branch-local ensemble-slice path in ``_execute_plan`` is
    exercised.
    """
    from scipy import stats as _stats

    cap_bike = DistributionIndex("cap_bike", _stats.norm, {"loc": 100.0, "scale": 5.0})
    cap_train = DistributionIndex("cap_train", _stats.norm, {"loc": 200.0, "scale": 10.0})
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = ModelVariant(
        "Transport",
        {"bike": _BikeModel(cap_bike), "train": _TrainModel(cap_train)},
        selector=mode,
    )
    return cap_bike, cap_train, mv


def test_regional_branch_local_abstract_index_correctness():
    """Regional execution with branch-local abstract indexes matches monolithic.

    Each branch samples its own distinct distribution, so the regional plan
    must correctly slice the ensemble arrays per branch and scatter results
    back to the full scenario array.  The test verifies the regional result
    equals the monolithic baseline element-wise.
    """
    cap_bike, cap_train, mv = _make_mv_branch_local()
    ev = Evaluation(mv)
    rng = np.random.default_rng(42)

    ens_mono = DistributionEnsemble(mv, size=200, rng=np.random.default_rng(42))
    ens_reg = DistributionEnsemble(mv, size=200, rng=rng)

    plan_mono = ev.build_plan(strategy="monolithic")
    plan_reg = ev.build_plan(strategy="regional")

    result_mono = ev.execute_plan(plan_mono, ens_mono)
    result_reg = ev.execute_plan(plan_reg, ens_reg)

    np.testing.assert_array_equal(
        result_mono[mv.outputs.throughput],
        result_reg[mv.outputs.throughput],
    )


# ---------------------------------------------------------------------------
# Selective execution: PositiveOnly vs AllValues
# ---------------------------------------------------------------------------
#
# Scenario:
#   Two runtime variants share a single abstract index ``x``.
#   - "positive" branch  → PositiveOnlyModel: computes strict_sqrt(x),
#     a user-defined function that raises ValueError for any negative input.
#   - "negative" branch  → AllValuesModel: computes x**2, safe everywhere.
#
# The ensemble is deliberately correlated:
#   - scenarios  0–49: sign="positive", x ∈ (0.1, 2.0)  (all positive)
#   - scenarios 50–99: sign="negative", x ∈ (−2.0, −0.1)  (all negative)
#
# Monolithic evaluates ALL branches for ALL scenarios:
#   strict_sqrt receives x[0..99] which includes the 50 negative values → raises.
#
# Regional evaluates each branch only for its matching scenarios:
#   strict_sqrt receives x[0..49] (all positive) → succeeds.
#   x**2       receives x[50..99] (all negative) → succeeds (squares are positive).


def _make_sign_variant() -> tuple[CategoricalIndex, Index, ModelVariant]:
    """Build the PositiveOnly/AllValues ModelVariant used by the selective-execution tests."""
    sign = CategoricalIndex("sign", {"positive": 0.5, "negative": 0.5})
    x = Index("x", None)  # abstract — values injected by the correlated ensemble

    class _PositiveOnlyModel(Model):
        @dataclasses.dataclass
        class Inputs:
            x: Index

        @dataclasses.dataclass
        class Outputs:
            result: Index

        def __init__(self, x: Index) -> None:
            result = Index("result", _graph.function_call("strict_sqrt", x.node))
            super().__init__(
                "PositiveOnly",
                inputs=_PositiveOnlyModel.Inputs(x=x),
                outputs=_PositiveOnlyModel.Outputs(result=result),
            )

    class _AllValuesModel(Model):
        @dataclasses.dataclass
        class Inputs:
            x: Index

        @dataclasses.dataclass
        class Outputs:
            result: Index

        def __init__(self, x: Index) -> None:
            result = Index("result", x.node**2)
            super().__init__(
                "AllValues",
                inputs=_AllValuesModel.Inputs(x=x),
                outputs=_AllValuesModel.Outputs(result=result),
            )

    mv = ModelVariant(
        "SqrtModel",
        {"positive": _PositiveOnlyModel(x), "negative": _AllValuesModel(x)},
        selector=sign,
    )
    return sign, x, mv


class _CorrelatedEnsemble:
    """Single-axis ensemble with hand-crafted correlated sign / x values.

    Scenarios 0–49: sign="positive", x ∈ (0.1, 2.0).
    Scenarios 50–99: sign="negative", x ∈ (−2.0, −0.1).
    """

    N_PER_BRANCH = 50

    def __init__(self, sign_idx: GenericIndex, x_idx: GenericIndex, rng: np.random.Generator) -> None:
        n = self.N_PER_BRANCH
        self._sign_arr = np.array(["positive"] * n + ["negative"] * n, dtype=object)
        self._x_arr = np.concatenate(
            [
                rng.uniform(0.1, 2.0, n),  # positive x for "positive" branch
                rng.uniform(-2.0, -0.1, n),  # negative x for "negative" branch
            ]
        )
        self._sign_idx = sign_idx
        self._x_idx = x_idx
        self._axis = Axis("_ensemble", ENSEMBLE)

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        return (self._axis,)

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        n_total = 2 * self.N_PER_BRANCH
        return (np.full(n_total, 1.0 / n_total),)

    def assignments(self) -> Mapping[GenericIndex, np.ndarray]:
        return {self._sign_idx: self._sign_arr, self._x_idx: self._x_arr}


def _strict_sqrt(arr: np.ndarray) -> np.ndarray:
    """Square root that raises ValueError for any negative input."""
    n_neg = int(np.sum(arr < 0))
    if n_neg > 0:
        raise ValueError(f"strict_sqrt: received {n_neg} negative value(s) (min={arr.min():.3f})")
    return np.sqrt(arr)


def test_monolithic_evaluates_all_branches_strict_sqrt_raises() -> None:
    """Monolithic execution evaluates every branch for every scenario.

    The "positive" branch calls strict_sqrt(x).  In monolithic mode x
    contains the full 100-element array (50 positive + 50 negative), so
    strict_sqrt raises as soon as it encounters a negative value.
    """
    sign, x, mv = _make_sign_variant()
    ens = _CorrelatedEnsemble(sign, x, np.random.default_rng(0))
    functions = {"strict_sqrt": NumpyBackend.adapt(_strict_sqrt)}
    ev = Evaluation(mv)
    plan = ev.build_plan(strategy="monolithic")

    with pytest.raises(ValueError, match="strict_sqrt.*negative"):
        ev.execute_plan(plan, ens, functions=functions)


def test_regional_evaluates_branch_only_for_matching_scenarios() -> None:
    """Regional execution evaluates each branch only for its scenario subset.

    The "positive" branch receives only x[0..49] (all > 0) → strict_sqrt
    succeeds.  The "negative" branch receives only x[50..99] (all < 0) →
    x**2 succeeds.  Every result is finite and numerically correct.
    """
    sign, x, mv = _make_sign_variant()
    rng = np.random.default_rng(1)
    ens = _CorrelatedEnsemble(sign, x, rng)
    functions = {"strict_sqrt": NumpyBackend.adapt(_strict_sqrt)}
    ev = Evaluation(mv)
    plan = ev.build_plan(strategy="regional")

    result = ev.execute_plan(plan, ens, functions=functions)
    arr = result[mv.outputs.result].ravel()  # shape (100,)

    assert np.all(np.isfinite(arr)), f"Expected all finite; got NaN/inf at {np.where(~np.isfinite(arr))[0]}"

    n = _CorrelatedEnsemble.N_PER_BRANCH
    x_arr = ens._x_arr
    # positive scenarios → sqrt(x)
    np.testing.assert_allclose(arr[:n], np.sqrt(x_arr[:n]), rtol=1e-12)
    # negative scenarios → x**2 (positive because squaring)
    np.testing.assert_allclose(arr[n:], x_arr[n:] ** 2, rtol=1e-12)
