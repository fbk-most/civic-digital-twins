# SPDX-License-Identifier: Apache-2.0
"""Tests for build_plan(strategy='regional') and execute_plan with multi-region plans."""

from collections.abc import Mapping

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import define, inputs, outputs
from civic_digital_twins.dt_model import graph as _graph
from civic_digital_twins.dt_model.axes import ENSEMBLE, Axis
from civic_digital_twins.dt_model.engine.numpybackend.executor import NumpyBackend
from civic_digital_twins.dt_model.model.index import CategoricalIndex, DistributionIndex, GenericIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.model.model_variant import ModelVariant
from civic_digital_twins.dt_model.simulation.ensemble import (
    CrossProductEnsemble,
    DistributionEnsemble,
    EnsembleAxisSpec,
    FrozenEnsemble,
    PartitionedEnsemble,
    WeightedScenario,
)
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.simulation.plan import EvaluationPlan, Region, RegionGuard
from civic_digital_twins.dt_model.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# Shared model fixtures (same as test_model_variant_evaluation.py)
# ---------------------------------------------------------------------------

_CAPACITY_VALUE = 100.0


@define("BikeModel")
class _BikeModel(Model):
    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        throughput: Index
        emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute throughput/emissions for the bike variant."""
        throughput = Index("throughput", inputs.capacity.node * 1.0)
        emissions = Index("emissions", 0.0)
        return _BikeModel.Outputs(throughput=throughput, emissions=emissions)


@define("TrainModel")
class _TrainModel(Model):
    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        throughput: Index
        emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute throughput/emissions for the train variant."""
        throughput = Index("throughput", inputs.capacity.node * 10.0)
        emissions = Index("emissions", 50.0)
        return _TrainModel.Outputs(throughput=throughput, emissions=emissions)


def _make_mv(mode: CategoricalIndex) -> ModelVariant:
    """Build a Transport ModelVariant with fixed capacity for both branches."""
    cap_bike = Index("capacity", _CAPACITY_VALUE)
    cap_train = Index("capacity", _CAPACITY_VALUE)
    return ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train)),
        },
        selector=mode,
    )


def _make_presence_mv(mode: CategoricalIndex) -> tuple[Index, ModelVariant]:
    """Build a Transport ModelVariant where both sub-models share a presence axis."""
    presence = Index("presence", None)
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=presence)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=presence)),
        },
        selector=mode,
    )
    return presence, mv


def _frozen_from_weighted_scenarios(
    scenarios: list[WeightedScenario],
    abstract_indexes: list[GenericIndex],
) -> FrozenEnsemble:
    """Materialise an explicit list of weighted scenarios into a FrozenEnsemble.

    Replaces the legacy ``Iterable[WeightedScenario]`` ensemble adaptation
    (removed in 0.11.0) for tests that need a small, deterministic ensemble
    built from explicit scenario assignments.
    """
    axis = Axis("_ensemble", ENSEMBLE)
    weights = np.array([w for w, _ in scenarios])
    assignments: dict[GenericIndex, np.ndarray] = {}
    for idx in abstract_indexes:
        values = [a[idx] for _, a in scenarios]
        normalized = [v.flat[0] if isinstance(v, np.ndarray) and v.size == 1 else v for v in values]
        assignments[idx] = np.asarray(normalized)
    return FrozenEnsemble((axis,), (weights,), assignments)


# ---------------------------------------------------------------------------
# build_plan(strategy='regional') — structural tests
# ---------------------------------------------------------------------------


def test_regional_plan_has_correct_region_count():
    """Regional plan for a 2-branch variant has exactly 4 regions: shared + 2 branches + merge."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan(strategy="regional")
    # 1 shared + 2 branch + 1 merge = 4
    assert len(plan.regions) == 4


def test_regional_plan_is_evaluation_plan():
    """build_plan(strategy='regional') returns an EvaluationPlan instance."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(Scenario(mv)).build_plan(strategy="regional")
    assert isinstance(plan, EvaluationPlan)


def test_regional_plan_regions_are_region_instances():
    """Every region in a regional plan is a Region instance."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(Scenario(mv)).build_plan(strategy="regional")
    for region in plan.regions:
        assert isinstance(region, Region)


def test_regional_plan_shared_region_has_no_guard():
    """The first region (shared) must be unconditional (no guards)."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(Scenario(mv)).build_plan(strategy="regional")
    assert plan.regions[0].guards == ()


def test_regional_plan_branch_regions_have_guards():
    """Middle regions (branches) must carry exactly one RegionGuard."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(Scenario(mv)).build_plan(strategy="regional")
    # regions[1] and regions[2] are branches
    for region in plan.regions[1:-1]:
        assert len(region.guards) == 1
        assert isinstance(region.guards[0], RegionGuard)


def test_regional_plan_branch_keys_match_variant():
    """Branch region guard branch_key values must match the ModelVariant's branch keys."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(Scenario(mv)).build_plan(strategy="regional")
    branch_keys = {region.guards[0].branch_key for region in plan.regions[1:-1]}
    assert branch_keys == {"bike", "train"}


def test_regional_plan_merge_region_has_no_guard():
    """The last region (merge) must be unconditional (no guards)."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(Scenario(mv)).build_plan(strategy="regional")
    assert plan.regions[-1].guards == ()


def test_regional_plan_correct_dependencies():
    """Check DAG dependencies: shared→branches→merge."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    plan = Evaluation(Scenario(mv)).build_plan(strategy="regional")
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


def test_regional_emissions_bike_only():
    """Regional plan: bike-only emissions = 0."""
    mode = CategoricalIndex("mode", {"bike": 1.0})
    mv = _make_mv(mode)
    ev = Evaluation(Scenario(mv))

    scenarios_list = _bike_only_scenarios(mode, 4)
    ensemble = _frozen_from_weighted_scenarios(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.emissions], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, ensemble)
    assert float(regional_result.expected_value(mv.outputs.emissions)) == pytest.approx(0.0)


def test_regional_emissions_train_only():
    """Regional plan: train-only emissions = 50."""
    mode = CategoricalIndex("mode", {"train": 1.0})
    mv = _make_mv(mode)
    ev = Evaluation(Scenario(mv))

    scenarios_list = _train_only_scenarios(mode, 4)
    ensemble = _frozen_from_weighted_scenarios(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.emissions], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, ensemble)
    assert float(regional_result.expected_value(mv.outputs.emissions)) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# execute_plan with PARAMETER axes + regional plan
# ---------------------------------------------------------------------------


def test_regional_plan_with_parameter_axis_bike_only():
    """Regional plan + PARAMETER axis: bike-only throughput = presence * 1."""
    mode = CategoricalIndex("mode", {"bike": 1.0})
    presence, mv = _make_presence_mv(mode)
    ev = Evaluation(Scenario(mv))
    xs = np.array([100.0, 200.0, 300.0])

    # Single scenario with mode="bike" (presence is the PARAMETER axis)
    scenarios_list: list[WeightedScenario] = [(1.0, {mode: np.array(["bike"])})]
    ensemble = _frozen_from_weighted_scenarios(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    result = ev.execute_plan(regional_plan, ensemble, parameters={presence: xs})

    assert np.allclose(result.expected_value(mv.outputs.throughput), xs * 1.0)


def test_regional_plan_with_parameter_axis_train_only():
    """Regional plan + PARAMETER axis: train-only throughput = presence * 10."""
    mode = CategoricalIndex("mode", {"train": 1.0})
    presence, mv = _make_presence_mv(mode)
    ev = Evaluation(Scenario(mv))
    xs = np.array([100.0, 200.0, 300.0])

    scenarios_list: list[WeightedScenario] = [(1.0, {mode: np.array(["train"])})]
    ensemble = _frozen_from_weighted_scenarios(scenarios_list, [mode])

    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    result = ev.execute_plan(regional_plan, ensemble, parameters={presence: xs})

    assert np.allclose(result.expected_value(mv.outputs.throughput), xs * 10.0)


# ---------------------------------------------------------------------------
# execute_plan raises for a PARAMETER-varying selector
# ---------------------------------------------------------------------------


def test_regional_parameter_varying_selector_matches_monolithic():
    """Regional execution supports selectors that vary along a PARAMETER axis."""
    presence = Index("presence", None)
    selector = ModelVariant.guards_to_selector([("train", presence.node > 150.0), ("bike", True)])
    cap = Index("capacity", 100.0)
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=Index("capacity", 100.0))),
        },
        selector=selector,
    )
    ev = Evaluation(Scenario(mv))
    xs = np.array([100.0, 200.0, 300.0])

    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, ensemble=None, parameters={presence: xs})
    monolithic_result = ev.evaluate(ensemble=None, nodes_of_interest=[mv.outputs.throughput], parameters={presence: xs})

    np.testing.assert_allclose(regional_result[mv.outputs.throughput], monolithic_result[mv.outputs.throughput])
    np.testing.assert_allclose(regional_result[mv.outputs.throughput].ravel(), np.array([100.0, 1000.0, 1000.0]))


# ---------------------------------------------------------------------------
# execute_plan with no leading axes (deterministic, leading_shape = ())
# ---------------------------------------------------------------------------


def test_regional_deterministic_no_leading_axes():
    """Regional execution with no PARAMETER and no ENSEMBLE axes (leading_shape = ()).

    Pinning the CategoricalIndex selector to a concrete branch via a Scenario
    override eliminates all leading axes.  The guard mask is a scalar boolean
    and scatter is a no-op.
    """
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    scenario = Scenario(mv, overrides={mode: "train"})
    ev = Evaluation(scenario)

    regional_plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    regional_result = ev.execute_plan(regional_plan, ensemble=None)
    monolithic_result = ev.evaluate(ensemble=None, nodes_of_interest=[mv.outputs.throughput])

    # Selector is pinned to "train"; train throughput = capacity * 10.
    np.testing.assert_allclose(regional_result[mv.outputs.throughput], monolithic_result[mv.outputs.throughput])
    assert float(np.asarray(regional_result[mv.outputs.throughput])) == pytest.approx(_CAPACITY_VALUE * 10.0)


# ---------------------------------------------------------------------------
# execute_plan with multi-axis / cross-product ensembles + regional plan
# ---------------------------------------------------------------------------


def _frozen_from_axis_ensemble(ensemble: PartitionedEnsemble) -> FrozenEnsemble:
    """Materialise a multi-axis ensemble once so monolithic and regional share samples."""
    return FrozenEnsemble(
        ensemble.ensemble_axes,
        ensemble.ensemble_weights,
        dict(ensemble.assignments()),
    )


def test_regional_plan_with_cross_product_ensemble_matches_monolithic():
    """Regional execution works with CrossProductEnsemble."""
    from scipy import stats as _stats

    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    presence = DistributionIndex("presence", _stats.uniform, {"loc": 100.0, "scale": 10.0})
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=presence)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=presence)),
        },
        selector=mode,
    )
    scenario = Scenario(mv)
    ensemble = CrossProductEnsemble(scenario, n_samples_per_combo=4, rng=np.random.default_rng(7))
    ev = Evaluation(scenario)

    monolithic = ev.execute_plan(ev.build_plan([mv.outputs.throughput]), ensemble)
    regional = ev.execute_plan(ev.build_plan([mv.outputs.throughput], strategy="regional"), ensemble)

    np.testing.assert_allclose(regional[mv.outputs.throughput], monolithic[mv.outputs.throughput])


def test_regional_plan_with_partitioned_ensemble_matches_monolithic():
    """Regional execution masks over all ENSEMBLE axes of a PartitionedEnsemble."""
    from scipy import stats as _stats

    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    presence = DistributionIndex("presence", _stats.uniform, {"loc": 100.0, "scale": 10.0})
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=presence)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=presence)),
        },
        selector=mode,
    )
    scenario = Scenario(mv)
    recipe = PartitionedEnsemble(
        scenario,
        axes=[
            EnsembleAxisSpec("mode_axis", [mode], size=6),
            EnsembleAxisSpec("presence_axis", [presence], size=5),
        ],
        rng=np.random.default_rng(11),
    )
    ensemble = _frozen_from_axis_ensemble(recipe)
    ev = Evaluation(scenario)

    monolithic = ev.execute_plan(ev.build_plan([mv.outputs.throughput]), ensemble)
    regional = ev.execute_plan(ev.build_plan([mv.outputs.throughput], strategy="regional"), ensemble)

    np.testing.assert_allclose(regional[mv.outputs.throughput], monolithic[mv.outputs.throughput])


# ---------------------------------------------------------------------------
# build_plan(strategy='regional') raises ValueError for plain Model
# ---------------------------------------------------------------------------


def test_regional_plan_raises_for_plain_model():
    """build_plan(strategy='regional') must raise ValueError when no variant_selector exists."""
    cap = Index("capacity", 100.0)
    plain_model = _BikeModel(inputs=_BikeModel.Inputs(capacity=cap))
    ev = Evaluation(Scenario(plain_model))
    with pytest.raises(ValueError, match="No variant_selector found"):
        ev.build_plan(strategy="regional")


# ---------------------------------------------------------------------------
# Regional execution error paths (coverage)
# ---------------------------------------------------------------------------


def test_regional_execute_plan_uncovered_selector_raises():
    """Regional execution without an ensemble still requires selector coverage."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv(mode)
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan(strategy="regional")
    with pytest.raises(ValueError, match="abstract indexes"):
        ev.execute_plan(plan, ensemble=None)


def test_regional_leading_mask_raises_for_array_selector_no_leading():
    """_leading_mask raises NotImplementedError for non-singleton DOMAIN with n_full==0.

    Covers the ``n_full == 0`` branch of ``_leading_mask`` where the mask has a
    dimension > 1 (the non-singleton DOMAIN raise).

    A ConstTimeseriesIndex with two time steps is used directly as the ModelVariant
    selector node.  With no ensemble and no parameters (n_full=0) the full (2,) constant
    array ends up as the selector value, making mask.shape=(2,).
    """
    from civic_digital_twins.dt_model.model.index import ConstTimeseriesIndex

    const_sel = ConstTimeseriesIndex("mode_sel", np.array(["bike", "train"]))
    cap_bike = Index("capacity", _CAPACITY_VALUE)
    cap_train = Index("capacity", _CAPACITY_VALUE)
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train)),
        },
        selector=const_sel.node,
    )
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan(strategy="regional")
    with pytest.raises(NotImplementedError, match="non-singleton DOMAIN"):
        ev.execute_plan(plan, ensemble=None)


# Note: in _leading_mask, the n_full==0 singleton-mask reshape ((1,) -> ()) is a
# defensive normalisation marked ``# pragma: no cover``.  It is only reachable
# by a selector that evaluates to a (1,) array rather than a 0-d scalar, which
# no supported index/selector construction produces (scalar selectors already
# yield mask.shape == ()).  The non-singleton DOMAIN raise on the same path is
# covered by test_regional_leading_mask_raises_for_array_selector_no_leading.


# ---------------------------------------------------------------------------
# Timeseries + regional: _normalise_leading domain-only reshape path and
# _scatter_leading extra_ts trailing-reshape path
# ---------------------------------------------------------------------------


def _make_ts_mv() -> tuple[CategoricalIndex, "ModelVariant"]:
    """Build a ModelVariant whose branches each have a timeseries AND a scalar output.

    * ``ts_out``  — formula derived from a ``TimeseriesIndex``; carries a DOMAIN axis.
    * ``throughput`` — formula derived only from ``capacity``; no DOMAIN axis.

    Having both kinds of outputs in the same branch region ensures:
    - ``_normalise_leading`` hits the domain-only reshape path (DOMAIN value with
      no explicit leading axes) during gather.
    - ``_scatter_leading`` hits the extra_ts trailing-reshape path for the scalar
      (non-DOMAIN) output.
    """
    from civic_digital_twins.dt_model.model.index import TimeseriesIndex

    T = 3
    ts = TimeseriesIndex("ts_load", np.arange(float(T)))
    cap_bike = Index("capacity", _CAPACITY_VALUE)
    cap_train = Index("capacity", _CAPACITY_VALUE * 2.0)
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})

    class _TSTransportModel(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index
            ts_load: GenericIndex

        @outputs
        class Outputs:
            throughput: Index
            ts_out: Index

        def __init__(self, cap: Index, ts_load: GenericIndex) -> None:
            thr = Index("throughput", cap.node * 1.0)
            ts_o = Index("ts_out", ts_load.node * 1.0)
            super().__init__(
                "_TSTransport",
                inputs=_TSTransportModel.Inputs(capacity=cap, ts_load=ts_load),
                outputs=_TSTransportModel.Outputs(throughput=thr, ts_out=ts_o),
            )

    mv = ModelVariant(
        "Transport",
        {"bike": _TSTransportModel(cap_bike, ts), "train": _TSTransportModel(cap_train, ts)},
        selector=mode,
    )
    return mode, mv


def test_regional_timeseries_branch_scatter_normalise():
    """Regional execution with timeseries covers _normalise_leading and the extra_ts scatter path.

    The branch regions contain both a timeseries formula node (ts_out, DOMAIN axis) and a
    scalar formula node (throughput, no DOMAIN axis).  With an ensemble present (n_full=1,
    extra_ts=1) the gather loop triggers the domain-only reshape path in _normalise_leading,
    and the scatter loop triggers the extra_ts reshape path for the scalar throughput node.
    Result correctness is verified by comparing regional to monolithic.
    """
    mode, mv = _make_ts_mv()
    scenario = Scenario(mv)
    rng = np.random.default_rng(0)
    ens = DistributionEnsemble(scenario, size=40, rng=rng)

    ev = Evaluation(scenario)
    plan_reg = ev.build_plan(strategy="regional")
    plan_mono = ev.build_plan(strategy="monolithic")

    rng2 = np.random.default_rng(0)
    ens2 = DistributionEnsemble(scenario, size=40, rng=rng2)

    result_reg = ev.execute_plan(plan_reg, ens)
    result_mono = ev.execute_plan(plan_mono, ens2)

    np.testing.assert_allclose(
        result_reg[mv.outputs.throughput],
        result_mono[mv.outputs.throughput],
    )


# ---------------------------------------------------------------------------
# _leading_mask trailing-DOMAIN path (n_full>0 branch): ConstTimeseriesIndex
# selector with ensemble (n_full=1) so sel broadcasts to shape (S, T).
# ---------------------------------------------------------------------------


def _make_const_ts_selector_mv(keys: list[str]) -> "ModelVariant":
    """Build a ModelVariant whose selector is a ConstTimeseriesIndex holding *keys*.

    ConstTimeseriesIndex creates a timeseries_constant node — no Scenario needed.
    With an ensemble (n_full=1) the selector broadcasts to shape (S, len(keys)).
    Branch keys are "bike" / "train".
    """
    from civic_digital_twins.dt_model.model.index import ConstTimeseriesIndex

    sel = ConstTimeseriesIndex("mode_sel", np.array(keys))
    cap_bike = Index("capacity", _CAPACITY_VALUE)
    cap_train = Index("capacity", _CAPACITY_VALUE)
    return ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train)),
        },
        selector=sel.node,
    )


def test_regional_leading_mask_raises_for_timeseries_selector_with_ensemble():
    """_leading_mask raises NotImplementedError for non-singleton DOMAIN with n_full>0.

    Covers the ``n_full > 0`` branch of ``_leading_mask`` where a trailing
    (DOMAIN) dimension is > 1 (the non-singleton DOMAIN raise).

    selector = ConstTimeseriesIndex(["bike","train"]) → shape (2,);
    with ensemble (n_full=1) → (S, 2); trailing dim 2 > 1 → NotImplementedError.
    """
    mv = _make_const_ts_selector_mv(["bike", "train"])
    scenario = Scenario(mv)
    ev = Evaluation(scenario)
    ens = DistributionEnsemble(scenario, size=10)
    plan = ev.build_plan(strategy="regional")
    with pytest.raises(NotImplementedError, match="non-singleton DOMAIN"):
        ev.execute_plan(plan, ens)


def test_regional_leading_mask_singleton_timeseries_selector_with_ensemble():
    """_leading_mask reshapes a singleton trailing (1,) dim when n_full>0.

    Covers the ``n_full > 0`` branch of ``_leading_mask`` where the trailing
    (DOMAIN) dimension is exactly 1 and is reshaped away.

    selector = ConstTimeseriesIndex(["bike"]) → shape (1,);
    with ensemble (n_full=1) → (S, 1); trailing dim=1 → reshape to (S,).
    All ensemble members fall in the "bike" branch; execution succeeds.

    Uses graph-backed emissions (capacity.node * 0.0) so scatter runs and applies
    the trailing (1,) reshape; avoids concrete scalar defaults that bypass scatter.
    """
    from civic_digital_twins.dt_model.model.index import ConstTimeseriesIndex

    sel = ConstTimeseriesIndex("mode_sel", np.array(["bike"]))
    cap_bike = Index("capacity", _CAPACITY_VALUE)
    cap_train = Index("capacity", _CAPACITY_VALUE)

    class _GraphBikeModel(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index

        @outputs
        class Outputs:
            throughput: Index
            emissions: Index

        def __init__(self, capacity: Index) -> None:
            throughput = Index("throughput", capacity.node * 1.0)
            emissions = Index("emissions", capacity.node * 0.0)
            super().__init__(
                "BikeModel",
                inputs=_GraphBikeModel.Inputs(capacity=capacity),
                outputs=_GraphBikeModel.Outputs(throughput=throughput, emissions=emissions),
            )

    class _GraphTrainModel(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index

        @outputs
        class Outputs:
            throughput: Index
            emissions: Index

        def __init__(self, capacity: Index) -> None:
            throughput = Index("throughput", capacity.node * 10.0)
            emissions = Index("emissions", capacity.node * 50.0)
            super().__init__(
                "TrainModel",
                inputs=_GraphTrainModel.Inputs(capacity=capacity),
                outputs=_GraphTrainModel.Outputs(throughput=throughput, emissions=emissions),
            )

    mv = ModelVariant(
        "Transport",
        {"bike": _GraphBikeModel(cap_bike), "train": _GraphTrainModel(cap_train)},
        selector=sel.node,
    )
    scenario = Scenario(mv)
    ev = Evaluation(scenario)
    ens = DistributionEnsemble(scenario, size=10)
    plan = ev.build_plan(strategy="regional")
    result = ev.execute_plan(plan, ens)
    # Verify both outputs are present. extra_ts=1 (selector is timeseries_constant)
    # adds a trailing (1,) to nodes without DOMAIN axes, so shape is (S, 1).
    assert result[mv.outputs.throughput].shape == (10, 1)
    assert result[mv.outputs.emissions].shape == (10, 1)


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
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train)),
        },
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
    _, _, mv = _make_mv_branch_local()
    scenario = Scenario(mv)
    ev = Evaluation(scenario)
    rng = np.random.default_rng(42)

    ens_mono = DistributionEnsemble(scenario, size=200, rng=np.random.default_rng(42))
    ens_reg = DistributionEnsemble(scenario, size=200, rng=rng)

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

    class _PositiveOnlyModel(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            result: Index

        def __init__(self, x: Index) -> None:
            result = Index("result", _graph.function_call("strict_sqrt", x.node))
            super().__init__(
                "PositiveOnly",
                inputs=_PositiveOnlyModel.Inputs(x=x),
                outputs=_PositiveOnlyModel.Outputs(result=result),
            )

    class _AllValuesModel(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @outputs
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
    ev = Evaluation(Scenario(mv))
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
    ev = Evaluation(Scenario(mv))
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


# ---------------------------------------------------------------------------
# Nested ModelVariant — two-level nesting tests  (issue #177)
# ---------------------------------------------------------------------------
#
# Outer: Mode.{bike, car}
# Inner (car branch only): Policy.{strict, loose}
#
# Throughput formulas:
#   bike                     → capacity * 1.0
#   car + strict policy      → capacity * 2.0
#   car + loose  policy      → capacity * 3.0
#
# With capacity = 100 and equal probability:
#   E[throughput] = (1/3)*100 + (1/3)*200 + (1/3)*300 = 200
# ---------------------------------------------------------------------------


def _make_nested_mv() -> tuple[CategoricalIndex, CategoricalIndex, ModelVariant]:
    """Build Mode.{bike,car} × Policy.{strict,loose} nested ModelVariant.

    Inner variant lives inside the "car" branch.  Throughput formulas:
      bike           → capacity * 1.0
      car + strict   → capacity * 2.0
      car + loose    → capacity * 3.0
    """
    mode = CategoricalIndex("mode", {"bike": 1 / 3, "car": 2 / 3})
    policy = CategoricalIndex("policy", {"strict": 0.5, "loose": 0.5})
    cap = Index("capacity", _CAPACITY_VALUE)

    bike_model = _BikeModel(inputs=_BikeModel.Inputs(capacity=cap))  # throughput = cap * 1.0

    class _StrictCarModel(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index

        @outputs
        class Outputs:
            throughput: Index
            emissions: Index

        def __init__(self, capacity: Index) -> None:
            throughput = Index("throughput", capacity.node * 2.0)
            emissions = Index("emissions", 10.0)
            super().__init__(
                "StrictCar",
                inputs=_StrictCarModel.Inputs(capacity=capacity),
                outputs=_StrictCarModel.Outputs(throughput=throughput, emissions=emissions),
            )

    class _LooseCarModel(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index

        @outputs
        class Outputs:
            throughput: Index
            emissions: Index

        def __init__(self, capacity: Index) -> None:
            throughput = Index("throughput", capacity.node * 3.0)
            emissions = Index("emissions", 20.0)
            super().__init__(
                "LooseCar",
                inputs=_LooseCarModel.Inputs(capacity=capacity),
                outputs=_LooseCarModel.Outputs(throughput=throughput, emissions=emissions),
            )

    inner_mv = ModelVariant(
        "Policy",
        {"strict": _StrictCarModel(cap), "loose": _LooseCarModel(cap)},
        selector=policy,
    )
    outer_mv = ModelVariant(
        "Transport",
        {"bike": bike_model, "car": inner_mv},
        selector=mode,
    )
    return mode, policy, outer_mv


# --- structural tests -------------------------------------------------------


def test_nested_regional_plan_region_count() -> None:
    """Two-level nesting produces exactly 7 regions."""
    mode, policy, mv = _make_nested_mv()
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan(strategy="regional")
    # outer-shared + bike + car-shared + car-strict + car-loose + car-merge + outer-merge
    assert len(plan.regions) == 7


def test_nested_regional_plan_guard_structure() -> None:
    """Verify the guards tuple for each of the 7 regions.

    Expected layout (topological order):
      0  outer shared           guards=()
      1  outer bike branch      guards=(mode=="bike",)
      2  car-context shared     guards=(mode=="car",)
      3  car + strict branch    guards=(mode=="car", policy=="strict")
      4  car + loose branch     guards=(mode=="car", policy=="loose")
      5  car-context merge      guards=(mode=="car",)
      6  outer merge            guards=()
    """
    mode, policy, mv = _make_nested_mv()
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan(strategy="regional")

    def _guard_sig(region: Region) -> tuple[str, ...]:
        return tuple(g.branch_key for g in region.guards)

    sigs = [_guard_sig(r) for r in plan.regions]

    assert sigs[0] == ()  # outer shared
    assert sigs[6] == ()  # outer merge

    # exactly two regions with one car guard and no further inner guard
    car_only = [s for s in sigs if s == ("car",)]
    assert len(car_only) == 2  # car-shared and car-merge

    # one bike branch
    assert ("bike",) in sigs

    # exactly one of each inner branch
    assert ("car", "strict") in sigs
    assert ("car", "loose") in sigs


def test_nested_regional_plan_correct_dependencies() -> None:
    """DAG structure: each region depends only on its necessary predecessors."""
    mode, policy, mv = _make_nested_mv()
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan(strategy="regional")
    deps = plan.dependencies

    # Identify region indices by guard signature
    def _sig(i: int) -> tuple[str, ...]:
        return tuple(g.branch_key for g in plan.regions[i].guards)

    outer_shared = next(i for i in range(7) if _sig(i) == () and i == 0)
    outer_merge = next(i for i in range(7) if _sig(i) == () and i > 0)
    bike_idx = next(i for i in range(7) if _sig(i) == ("bike",))
    car_shared = next(i for i in range(7) if _sig(i) == ("car",) and i < outer_merge)
    car_merge = next(i for i in range(7) if _sig(i) == ("car",) and i > car_shared)
    strict_idx = next(i for i in range(7) if _sig(i) == ("car", "strict"))
    loose_idx = next(i for i in range(7) if _sig(i) == ("car", "loose"))

    # Outer shared has no predecessors
    assert deps[outer_shared] == frozenset()

    # Bike and car-shared depend on outer shared
    assert outer_shared in deps[bike_idx]
    assert outer_shared in deps[car_shared]

    # Inner branches depend on (at least) car-shared
    assert car_shared in deps[strict_idx]
    assert car_shared in deps[loose_idx]

    # Car-merge depends on (at least) car-shared, strict, loose
    assert car_shared in deps[car_merge]
    assert strict_idx in deps[car_merge]
    assert loose_idx in deps[car_merge]

    # Outer merge depends on (at least) outer-shared, bike, and car-merge
    assert outer_shared in deps[outer_merge]
    assert bike_idx in deps[outer_merge]
    assert car_merge in deps[outer_merge]


# --- correctness tests -------------------------------------------------------


def _nested_equal_weight_ensemble(mode: CategoricalIndex, policy: CategoricalIndex) -> list[WeightedScenario]:
    """Three equal-weight scenarios: bike, car+strict, car+loose."""
    return [
        (1 / 3, {mode: np.array(["bike"]), policy: np.array(["strict"])}),
        (1 / 3, {mode: np.array(["car"]), policy: np.array(["strict"])}),
        (1 / 3, {mode: np.array(["car"]), policy: np.array(["loose"])}),
    ]


def test_nested_regional_matches_monolithic() -> None:
    """Two-level nested regional plan produces the same result as monolithic."""
    mode, policy, mv = _make_nested_mv()
    ev = Evaluation(Scenario(mv))
    scenarios = _nested_equal_weight_ensemble(mode, policy)
    ensemble = _frozen_from_weighted_scenarios(scenarios, [mode, policy])

    mono = ev.execute_plan(ev.build_plan([mv.outputs.throughput]), ensemble)
    regional = ev.execute_plan(ev.build_plan([mv.outputs.throughput], strategy="regional"), ensemble)

    np.testing.assert_allclose(
        regional[mv.outputs.throughput],
        mono[mv.outputs.throughput],
    )


def test_nested_regional_expected_value() -> None:
    """Two-level nested: expected throughput = (100 + 200 + 300) / 3 = 200."""
    mode, policy, mv = _make_nested_mv()
    ev = Evaluation(Scenario(mv))
    scenarios = _nested_equal_weight_ensemble(mode, policy)
    ensemble = _frozen_from_weighted_scenarios(scenarios, [mode, policy])

    plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    result = ev.execute_plan(plan, ensemble)

    assert float(result.expected_value(mv.outputs.throughput)) == pytest.approx(200.0)


# --- selective execution at two levels --------------------------------------
#
# The inner "strict" branch calls strict_sqrt(x) which raises on negative x.
# The inner "loose" branch computes x**2, safe everywhere.
# The outer "bike" branch computes x * 1.0, safe everywhere.
#
# Correlated ensemble:
#   0..49  : mode="bike",  policy can be anything  → bike branch only
#   50..74 : mode="car",   policy="strict", x > 0  → strict branch (strict_sqrt safe)
#   75..99 : mode="car",   policy="loose",  x < 0  → loose branch (x**2 safe)
#
# Monolithic: strict_sqrt sees all 100 x values including negatives → raises.
# Regional:   strict_sqrt sees only x[50..74] (all positive) → succeeds.


def _make_nested_sign_variant() -> tuple[CategoricalIndex, CategoricalIndex, Index, ModelVariant]:
    """Build Mode.{bike, car} × Policy.{strict, loose} with strict_sqrt in strict branch."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "car": 0.5})
    policy = CategoricalIndex("policy", {"strict": 0.5, "loose": 0.5})
    x = Index("x", None)

    class _BikePassthroughModel(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            result: Index

        def __init__(self, x: Index) -> None:
            result = Index("result", x.node * 1.0)
            super().__init__(
                "BikePassthrough",
                inputs=_BikePassthroughModel.Inputs(x=x),
                outputs=_BikePassthroughModel.Outputs(result=result),
            )

    class _StrictSqrtModel(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            result: Index

        def __init__(self, x: Index) -> None:
            result = Index("result", _graph.function_call("strict_sqrt", x.node))
            super().__init__(
                "StrictSqrt",
                inputs=_StrictSqrtModel.Inputs(x=x),
                outputs=_StrictSqrtModel.Outputs(result=result),
            )

    class _SquareModel(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            result: Index

        def __init__(self, x: Index) -> None:
            result = Index("result", x.node**2)
            super().__init__(
                "Square",
                inputs=_SquareModel.Inputs(x=x),
                outputs=_SquareModel.Outputs(result=result),
            )

    inner_mv = ModelVariant(
        "Policy",
        {"strict": _StrictSqrtModel(x), "loose": _SquareModel(x)},
        selector=policy,
    )
    outer_mv = ModelVariant(
        "Transport",
        {"bike": _BikePassthroughModel(x), "car": inner_mv},
        selector=mode,
    )
    return mode, policy, x, outer_mv


class _NestedCorrelatedEnsemble:
    """Single-axis correlated ensemble for the nested selective-execution test.

    Scenarios 0..49:  mode="bike",  policy="strict", x ∈ (0.1, 2.0)
    Scenarios 50..74: mode="car",   policy="strict", x ∈ (0.1, 2.0)  (positive)
    Scenarios 75..99: mode="car",   policy="loose",  x ∈ (−2.0, −0.1) (negative)
    """

    N_BIKE = 50
    N_CAR_STRICT = 25
    N_CAR_LOOSE = 25

    def __init__(
        self,
        mode_idx: GenericIndex,
        policy_idx: GenericIndex,
        x_idx: GenericIndex,
        rng: np.random.Generator,
    ) -> None:
        n_bike, n_strict, n_loose = self.N_BIKE, self.N_CAR_STRICT, self.N_CAR_LOOSE
        n_total = n_bike + n_strict + n_loose
        self._mode = np.array(["bike"] * n_bike + ["car"] * (n_strict + n_loose), dtype=object)
        self._policy = np.array(["strict"] * n_bike + ["strict"] * n_strict + ["loose"] * n_loose, dtype=object)
        self._x = np.concatenate(
            [
                rng.uniform(0.1, 2.0, n_bike),
                rng.uniform(0.1, 2.0, n_strict),
                rng.uniform(-2.0, -0.1, n_loose),
            ]
        )
        self._mode_idx = mode_idx
        self._policy_idx = policy_idx
        self._x_idx = x_idx
        self._axis = Axis("_ensemble", ENSEMBLE)
        self._n_total = n_total

    @property
    def ensemble_axes(self) -> tuple[Axis, ...]:
        return (self._axis,)

    @property
    def ensemble_weights(self) -> tuple[np.ndarray, ...]:
        return (np.full(self._n_total, 1.0 / self._n_total),)

    def assignments(self) -> Mapping[GenericIndex, np.ndarray]:
        return {self._mode_idx: self._mode, self._policy_idx: self._policy, self._x_idx: self._x}


def test_nested_monolithic_strict_sqrt_raises() -> None:
    """Monolithic nested execution fails when strict_sqrt sees negative x."""
    mode, policy, x, mv = _make_nested_sign_variant()
    ens = _NestedCorrelatedEnsemble(mode, policy, x, np.random.default_rng(42))
    functions = {"strict_sqrt": NumpyBackend.adapt(_strict_sqrt)}
    ev = Evaluation(Scenario(mv))

    with pytest.raises(ValueError, match="strict_sqrt.*negative"):
        ev.execute_plan(ev.build_plan(strategy="monolithic"), ens, functions=functions)


def test_nested_regional_selective_execution() -> None:
    """Two-level nested regional execution: strict_sqrt only sees positive x.

    The "strict" branch receives only the car+strict scenarios (x > 0) so
    strict_sqrt never encounters a negative value.  Results are finite and
    match expected values per-scenario.
    """
    mode, policy, x, mv = _make_nested_sign_variant()
    rng = np.random.default_rng(7)
    ens = _NestedCorrelatedEnsemble(mode, policy, x, rng)
    functions = {"strict_sqrt": NumpyBackend.adapt(_strict_sqrt)}
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan(strategy="regional")

    result = ev.execute_plan(plan, ens, functions=functions)
    arr = result[mv.outputs.result].ravel()

    assert np.all(np.isfinite(arr)), f"Expected all finite; NaN/inf at {np.where(~np.isfinite(arr))[0]}"

    n_b = _NestedCorrelatedEnsemble.N_BIKE
    n_s = _NestedCorrelatedEnsemble.N_CAR_STRICT
    x_arr = ens._x

    # bike scenarios → x * 1.0
    np.testing.assert_allclose(arr[:n_b], x_arr[:n_b] * 1.0, rtol=1e-12)
    # car+strict → sqrt(x)
    np.testing.assert_allclose(arr[n_b : n_b + n_s], np.sqrt(x_arr[n_b : n_b + n_s]), rtol=1e-12)
    # car+loose → x**2
    np.testing.assert_allclose(arr[n_b + n_s :], x_arr[n_b + n_s :] ** 2, rtol=1e-12)


# ---------------------------------------------------------------------------
# EvaluationPlan.scoped_abstract_indexes(scenario) — branch-scope API
#
# Pins the contract for #137: group scenario-abstract indexes by the guard
# chain of the region whose .nodes contain each index's .node.  Keys are
# tuple[RegionGuard, ...] (empty tuple = unconditional).
# ---------------------------------------------------------------------------


def _make_branch_abstract_mv() -> tuple[CategoricalIndex, DistributionIndex, DistributionIndex, ModelVariant]:
    """2-branch ModelVariant with a per-branch abstract Index in each variant.

    Concrete capacity is shared by both branches; the only branch-specific
    nodes are the per-branch ``weather`` placeholders.  Used by both
    ``scoped_abstract_indexes`` tests and ``DistributionEnsemble`` per-scope
    sampling tests.
    """
    from scipy import stats as _stats

    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    cap = Index("capacity", _CAPACITY_VALUE)
    weather_bike = DistributionIndex("weather_bike", _stats.uniform, {"loc": 0.0, "scale": 1.0})
    weather_train = DistributionIndex("weather_train", _stats.uniform, {"loc": 0.0, "scale": 1.0})

    class _BikeBranch(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index
            weather: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, capacity: Index, weather: Index) -> None:
            throughput = Index("throughput", capacity.node * weather.node)
            super().__init__(
                "BikeBranch",
                inputs=_BikeBranch.Inputs(capacity=capacity, weather=weather),
                outputs=_BikeBranch.Outputs(throughput=throughput),
            )

    class _TrainBranch(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index
            weather: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, capacity: Index, weather: Index) -> None:
            throughput = Index("throughput", capacity.node * weather.node)
            super().__init__(
                "TrainBranch",
                inputs=_TrainBranch.Inputs(capacity=capacity, weather=weather),
                outputs=_TrainBranch.Outputs(throughput=throughput),
            )

    mv = ModelVariant(
        "Transport",
        {"bike": _BikeBranch(cap, weather_bike), "train": _TrainBranch(cap, weather_train)},
        selector=mode,
    )
    return mode, weather_bike, weather_train, mv


def _make_nested_branch_abstract_mv() -> tuple[
    CategoricalIndex, CategoricalIndex, DistributionIndex, DistributionIndex, DistributionIndex, ModelVariant
]:
    """Nested ModelVariant with an abstract Index in each scoped sub-region.

    Expected scope buckets (5 distinct keys, after collapsing same-guards
    regions and dropping empty buckets):

      ()                                            : mode
      ((mode, 'bike'),)                             : bike_w
      ((mode, 'car'),)                              : policy
      ((mode, 'car'), (policy, 'strict'))           : strict_w
      ((mode, 'car'), (policy, 'loose'))            : loose_w
    """
    from scipy import stats as _stats

    mode = CategoricalIndex("mode", {"bike": 1 / 3, "car": 2 / 3})
    policy = CategoricalIndex("policy", {"strict": 0.5, "loose": 0.5})
    cap = Index("capacity", _CAPACITY_VALUE)

    bike_w = DistributionIndex("bike_w", _stats.uniform, {"loc": 0.0, "scale": 1.0})
    strict_w = DistributionIndex("strict_w", _stats.uniform, {"loc": 0.0, "scale": 1.0})
    loose_w = DistributionIndex("loose_w", _stats.uniform, {"loc": 0.0, "scale": 1.0})

    class _BikeOuter(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index
            bike_w: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, capacity: Index, bike_w: Index) -> None:
            throughput = Index("throughput", capacity.node * bike_w.node)
            super().__init__(
                "BikeOuter",
                inputs=_BikeOuter.Inputs(capacity=capacity, bike_w=bike_w),
                outputs=_BikeOuter.Outputs(throughput=throughput),
            )

    class _StrictInner(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index
            strict_w: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, capacity: Index, strict_w: Index) -> None:
            throughput = Index("throughput", capacity.node * 2.0 * strict_w.node)
            super().__init__(
                "StrictInner",
                inputs=_StrictInner.Inputs(capacity=capacity, strict_w=strict_w),
                outputs=_StrictInner.Outputs(throughput=throughput),
            )

    class _LooseInner(Model, legacy=True):
        @inputs
        class Inputs:
            capacity: Index
            loose_w: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, capacity: Index, loose_w: Index) -> None:
            throughput = Index("throughput", capacity.node * 3.0 * loose_w.node)
            super().__init__(
                "LooseInner",
                inputs=_LooseInner.Inputs(capacity=capacity, loose_w=loose_w),
                outputs=_LooseInner.Outputs(throughput=throughput),
            )

    inner_mv = ModelVariant(
        "Policy",
        {"strict": _StrictInner(cap, strict_w), "loose": _LooseInner(cap, loose_w)},
        selector=policy,
    )
    outer_mv = ModelVariant(
        "Transport",
        {"bike": _BikeOuter(cap, bike_w), "car": inner_mv},
        selector=mode,
    )
    return mode, policy, bike_w, strict_w, loose_w, outer_mv


def test_scoped_abstract_indexes_monolithic():
    """Monolithic plan: a single () key holds every scenario-abstract index.

    Establishes the base case: a no-variant plan has one unconditional
    region and one bucket with the full abstract-index list.
    """
    x = Index("x", None)

    class _M(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            y: Index

        def __init__(self, x: Index) -> None:
            y = Index("y", x.node)
            super().__init__(
                "_M",
                inputs=_M.Inputs(x=x),
                outputs=_M.Outputs(y=y),
            )

    m = _M(x)
    scenario = Scenario(m)
    plan = Evaluation(scenario).build_plan()

    scoped = plan.scoped_abstract_indexes(scenario)

    assert scoped == {(): frozenset({x})}


def test_scoped_abstract_indexes_regional_single_level():
    """Single-level regional: 3 keys, branch abstracts route to their branch.

    Pins: branch-keyed entries use a *singleton tuple* guard; shared and
    merge regions (both with guards=()) collapse into one () key; empty
    buckets (if any) are not emitted.
    """
    mode, weather_bike, weather_train, mv = _make_branch_abstract_mv()
    scenario = Scenario(mv)
    plan = Evaluation(scenario).build_plan(strategy="regional")

    scoped = plan.scoped_abstract_indexes(scenario)

    bike_guard = (RegionGuard(mode.node, "bike"),)
    train_guard = (RegionGuard(mode.node, "train"),)

    assert set(scoped.keys()) == {(), bike_guard, train_guard}
    assert scoped[()] == frozenset({mode})
    assert scoped[bike_guard] == frozenset({weather_bike})
    assert scoped[train_guard] == frozenset({weather_train})


def test_scoped_abstract_indexes_regional_nested():
    """Nested regional: keys are *tuples* of guards (length 0/1/2).

    Pins the load-bearing design point distinguishing this proposal from
    the issue sketch: a region can carry a multi-element guards chain, so
    the dict key must be ``tuple[RegionGuard, ...]``, not ``str``.

    Also pins that the *inner* selector (``policy``) lands at the
    ``(mode=='car',)`` key, not at the outer-shared ``()`` key — i.e.
    selector placement tracks the deepest scope where its node is needed.
    """
    mode, policy, bike_w, strict_w, loose_w, mv = _make_nested_branch_abstract_mv()
    scenario = Scenario(mv)
    plan = Evaluation(scenario).build_plan(strategy="regional")

    scoped = plan.scoped_abstract_indexes(scenario)

    # Recover the actual RegionGuard instances from the plan so the test
    # is independent of how those objects are constructed internally.
    guard_by_branch_key: dict[str, RegionGuard] = {g.branch_key: g for region in plan.regions for g in region.guards}
    bike_guard = (guard_by_branch_key["bike"],)
    car_guard = (guard_by_branch_key["car"],)
    strict_guard = (guard_by_branch_key["strict"],)
    loose_guard = (guard_by_branch_key["loose"],)

    expected_keys = {
        (),
        bike_guard,
        car_guard,
        car_guard + strict_guard,
        car_guard + loose_guard,
    }
    assert set(scoped.keys()) == expected_keys

    assert scoped[()] == frozenset({mode})
    assert scoped[car_guard] == frozenset({policy})
    assert scoped[bike_guard] == frozenset({bike_w})
    assert scoped[car_guard + strict_guard] == frozenset({strict_w})
    assert scoped[car_guard + loose_guard] == frozenset({loose_w})


def test_scoped_abstract_indexes_raises_on_overlap():
    """Raises when an abstract index's node appears in multiple regions.

    build_plan currently guarantees disjointness, but the per-scope
    allocation in DistributionEnsemble (issue #173) relies on this
    invariant: an index's node must live in exactly one region for the
    per-scope sampling to be well-defined.  We construct a synthetic
    plan with overlapping regions to verify the defensive check fires.
    """
    x = Index("x", None)
    y = Index("y", None)

    class _M(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index
            y: Index

        @outputs
        class Outputs:
            z: Index

        def __init__(self, x: Index, y: Index) -> None:
            z = Index("z", x.node + y.node)
            super().__init__(
                "_M",
                inputs=_M.Inputs(x=x, y=y),
                outputs=_M.Outputs(z=z),
            )

    m = _M(x, y)
    scenario = Scenario(m)
    # Synthesise an EvaluationPlan where x.node appears in two regions.
    plan = EvaluationPlan(
        model=m,
        nodes_of_interest=(x, y),
        regions=(
            Region(nodes=(x.node,), domain_axes=(), guards=()),
            Region(nodes=(x.node, y.node), domain_axes=(), guards=()),
        ),
        dependencies=(frozenset(), frozenset({0})),
    )

    with pytest.raises(ValueError, match="appear in multiple regions"):
        plan.scoped_abstract_indexes(scenario)


# ---------------------------------------------------------------------------
# DistributionEnsemble per-scope sampling (issue #173)
#
# When constructed with plan=, assignments() performs per-scope sampling:
# indexes scoped to a particular region are sampled only at the scenario
# positions where that region is active; unsampled slots are filled with
# a semantically valid default (mean() for DistributionIndex, argmax
# for CategoricalIndex).
# ---------------------------------------------------------------------------


def test_scoped_sampling_flat_branch_sample_count():
    """Flat 2-branch MV: per-branch DistributionIndex sampled at branch positions.

    p_bike = p_train = 0.5, size = 10_000.  Allow a 5-sigma binomial
    tolerance (std = 50) on each branch's real-sample count.  Real
    samples are detected by ``~np.isnan(arr)`` (the sentinel for
    unsampled DistributionIndex slots is ``np.nan``).
    """
    mode, weather_bike, weather_train, mv = _make_branch_abstract_mv()
    scenario = Scenario(mv)
    plan = Evaluation(scenario).build_plan(strategy="regional")
    rng = np.random.default_rng(0)
    ens = DistributionEnsemble(scenario, size=10_000, rng=rng, plan=plan)
    a = ens.assignments()

    # Shared: sampled at full size.
    assert a[mode].shape == (10_000,)

    # Per-branch: sentinel is np.nan; real samples are non-NaN.
    bike_real = int(np.sum(~np.isnan(a[weather_bike])))
    train_real = int(np.sum(~np.isnan(a[weather_train])))
    assert 4750 <= bike_real <= 5250
    assert 4750 <= train_real <= 5250
    assert bike_real + train_real == 10_000  # every position is exactly one branch


def test_scoped_sampling_nested_intersection():
    """Nested MV: inner-strict DistributionIndex sampled at car AND strict positions.

    Verifies the multi-pass / outer-to-inner invariant: positions for
    the (mode=='car', policy=='strict') bucket are intersected across
    all ancestor guards, not just the innermost.
    """
    mode, policy, bike_w, strict_w, loose_w, mv = _make_nested_branch_abstract_mv()
    scenario = Scenario(mv)
    plan = Evaluation(scenario).build_plan(strategy="regional")
    rng = np.random.default_rng(0)
    ens = DistributionEnsemble(scenario, size=30_000, rng=rng, plan=plan)
    a = ens.assignments()

    car_mask = a[mode] == "car"
    strict_mask = car_mask & (a[policy] == "strict")
    loose_mask = car_mask & (a[policy] == "loose")
    bike_mask = a[mode] == "bike"

    # Real-sample counts (non-NaN) must equal the position counts for
    # the corresponding active scopes.
    assert int(np.sum(~np.isnan(a[strict_w]))) == int(np.sum(strict_mask))
    assert int(np.sum(~np.isnan(a[loose_w]))) == int(np.sum(loose_mask))
    # bike_w is at the (mode=='bike',) bucket; its real-sample count
    # equals the count of bike positions.
    assert int(np.sum(~np.isnan(a[bike_w]))) == int(np.sum(bike_mask))


def test_scoped_sampling_categorical_sentinel():
    """Per-branch CategoricalIndex: non-active positions get the None sentinel.

    For the nested fixture, policy lives at the (mode=='car',) bucket.
    Non-car positions must be ``None``; car positions are real samples
    from {"strict", "loose"}.  Filtering with ``arr != None`` recovers
    the real samples.
    """
    mode, policy, _bike_w, _strict_w, _loose_w, mv = _make_nested_branch_abstract_mv()
    scenario = Scenario(mv)
    plan = Evaluation(scenario).build_plan(strategy="regional")
    rng = np.random.default_rng(0)
    ens = DistributionEnsemble(scenario, size=2000, rng=rng, plan=plan)
    a = ens.assignments()

    car_mask = a[mode] == "car"
    non_car = ~car_mask
    # All non-car positions: policy sentinel = None
    policy_non_car = a[policy][non_car]
    assert np.all(np.asarray(policy_non_car == None, dtype=bool))  # noqa: E711
    # Car positions: real samples from {"strict", "loose"}
    policy_car = np.asarray(a[policy][car_mask], dtype=object)
    assert set(policy_car.tolist()) <= {"strict", "loose"}


def test_scoped_sampling_draw_batch_propagates_plan():
    """draw_batch must propagate self._plan so the plan-aware path is used."""
    mode, weather_bike, _weather_train, mv = _make_branch_abstract_mv()
    scenario = Scenario(mv)
    plan = Evaluation(scenario).build_plan(strategy="regional")
    rng = np.random.default_rng(0)
    ens = DistributionEnsemble(scenario, size=1000, rng=rng, plan=plan)

    rng2 = np.random.default_rng(1)
    batch = ens.draw_batch(500, rng2)
    a = batch.assignments()

    # The batch has its own size = 500; per-scope sampling is active.
    assert a[mode].shape == (500,)
    bike_real = int(np.sum(~np.isnan(a[weather_bike])))
    # 50% of 500 = 250, allow binomial wiggle
    assert 200 <= bike_real <= 300


def test_scoped_sampling_end_to_end_matches_no_plan_mean() -> None:
    """Per-scope and no-plan DistributionEnsemble produce statistically equivalent throughput means.

    The per-scope path only samples per-branch DistributionIndex at
    branch positions and fills the rest with sentinels (np.nan).  The
    executor's region masking ensures the sentinels are never read, so
    the per-scope throughput mean converges to the same analytical value
    as the no-plan mean.

    Both paths run with the same seed.  The means will not be *equal*
    (the draw order and count differ), but they should agree within
    statistical tolerance and both should match the analytical
    expected value.
    """
    _, _, _, mv = _make_branch_abstract_mv()
    scenario = Scenario(mv)
    ev = Evaluation(scenario)
    n = 50_000

    rng_a = np.random.default_rng(42)
    ens_no = DistributionEnsemble(scenario, size=n, rng=rng_a)
    throughput_no = np.asarray(ev.execute_plan(ev.build_plan(), ens_no)[mv.outputs.throughput]).ravel()

    rng_b = np.random.default_rng(42)
    plan_reg = ev.build_plan(strategy="regional")
    ens_with = DistributionEnsemble(scenario, size=n, rng=rng_b, plan=plan_reg)
    throughput_with = np.asarray(ev.execute_plan(plan_reg, ens_with)[mv.outputs.throughput]).ravel()

    # Analytical expected throughput: cap * E[weather] = 100 * 0.5 = 50.
    expected = _CAPACITY_VALUE * 0.5

    # Per-position std is cap * sqrt(1/12) ~= 28.87.  Std of the mean
    # over n=50000 positions: ~0.129.  A tolerance of 1.0 is ~7.7
    # sigma — comfortably above any plausible Monte Carlo noise.
    assert abs(np.mean(throughput_no) - expected) < 1.0
    assert abs(np.mean(throughput_with) - expected) < 1.0

    # The two means should agree; tolerance ~5.5 sigma on the
    # difference-of-means standard error (~0.18).
    assert abs(np.mean(throughput_no) - np.mean(throughput_with)) < 1.5


def test_scoped_sampling_exclude_skips_bucket() -> None:
    """An abstract index listed in ``exclude`` is dropped from the per-scope result.

    The bucket containing only excluded indexes is skipped (the
    ``if not active_indexes: continue`` branch in ``_scoped_assignments``).
    Other buckets in the same plan are still sampled.
    """
    mode, weather_bike, weather_train, mv = _make_branch_abstract_mv()
    scenario = Scenario(mv)
    plan = Evaluation(scenario).build_plan(strategy="regional")
    ens = DistributionEnsemble(
        scenario,
        size=100,
        rng=np.random.default_rng(0),
        plan=plan,
        exclude=frozenset({weather_bike}),
    )
    a = ens.assignments()
    # mode is in the shared bucket — sampled at all positions.
    assert a[mode].shape == (100,)
    # weather_train is in its own bucket — sampled at train positions.
    assert a[weather_train].shape == (100,)
    assert 30 <= int(np.sum(~np.isnan(a[weather_train]))) <= 70
    # weather_bike was excluded; its bucket is empty after filtering.
    assert weather_bike not in a


def test_scoped_sampling_no_rng() -> None:
    """Per-scope sampling works without an rng (scipy's default is used).

    Exercises the ``self._rng is None`` branch in ``_sample_with_default``.
    """
    mode, weather_bike, _, mv = _make_branch_abstract_mv()
    scenario = Scenario(mv)
    plan = Evaluation(scenario).build_plan(strategy="regional")
    ens = DistributionEnsemble(scenario, size=200, plan=plan)
    a = ens.assignments()
    assert a[mode].shape == (200,)
    # Real samples for weather_bike have mean ~0.5 (uniform[0, 1]).
    bike_real = a[weather_bike][~np.isnan(a[weather_bike])]
    assert 0.35 < float(np.mean(bike_real)) < 0.65


def test_scoped_abstract_indexes_unions_same_guards_regions() -> None:
    """Two regions with the same guards tuple union their abstract indexes.

    build_plan currently never produces duplicate guard tuples, but the
    union branch is the correct fallback if it ever does — abstract
    indexes that live in regions with identical guards are all active
    whenever the guards are satisfied.
    """
    x = Index("x", None)
    y = Index("y", None)

    class _M(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index
            y: Index

        @outputs
        class Outputs:
            z: Index

        def __init__(self, x: Index, y: Index) -> None:
            z = Index("z", x.node + y.node)
            super().__init__(
                "_M",
                inputs=_M.Inputs(x=x, y=y),
                outputs=_M.Outputs(z=z),
            )

    m = _M(x, y)
    scenario = Scenario(m)
    # x and y are disjoint, so the overlap check passes.  The two
    # regions share guards=() and the union branch fires for the second.
    plan = EvaluationPlan(
        model=m,
        nodes_of_interest=(x, y),
        regions=(
            Region(nodes=(x.node,), domain_axes=(), guards=()),
            Region(nodes=(y.node,), domain_axes=(), guards=()),
        ),
        dependencies=(frozenset(), frozenset({0})),
    )
    assert plan.scoped_abstract_indexes(scenario) == {(): frozenset({x, y})}


def test_scoped_sampling_pinned_selector_raises() -> None:
    """A guard whose selector is not in ``scenario.abstract_indexes()`` raises.

    Per-scope sampling requires every guard's selector to be an
    abstract index.  When the selector is pinned (via Scenario
    overrides or ``parameter_axes=``), it is no longer in
    ``scenario.abstract_indexes()`` and the per-scope path cannot
    resolve the guard.  ``_active_positions`` raises a clear
    ``ValueError`` so the user gets a fast, actionable failure
    rather than silently wrong per-branch statistics.
    """
    mode, _weather_bike, _weather_train, mv = _make_branch_abstract_mv()
    plan = Evaluation(Scenario(mv)).build_plan(strategy="regional")
    pinned_scenario = Scenario(mv, overrides={mode: "bike"})
    with pytest.raises(ValueError, match="not in scenario.abstract_indexes"):
        DistributionEnsemble(pinned_scenario, size=100, rng=np.random.default_rng(0), plan=plan).assignments()


def test_scoped_sampling_defensive_wrong_order_plan_raises() -> None:
    """A custom plan with regions in the wrong order triggers the defensive RuntimeError.

    The build_plan invariant guarantees that the bucket order from
    ``scoped_abstract_indexes`` is outer-to-inner — the shared
    pre-selector region is processed first, so its bucket is populated
    before any per-branch bucket uses it as a guard.  We construct a
    custom plan that violates this invariant to verify the defensive
    check fires.
    """
    mode, weather_bike, _weather_train, mv = _make_branch_abstract_mv()
    scenario = Scenario(mv)
    # Wrong order: per-branch region first (depends on mode), shared
    # region second (where mode is actually sampled).
    bike_guard = RegionGuard(selector_node=mode.node, branch_key="bike")
    plan = EvaluationPlan(
        model=mv,
        nodes_of_interest=(mv.outputs.throughput,),
        regions=(
            Region(nodes=(weather_bike.node,), domain_axes=(), guards=(bike_guard,)),
            Region(nodes=(mode.node,), domain_axes=(), guards=()),
        ),
        dependencies=(frozenset(), frozenset({0})),
    )
    with pytest.raises(RuntimeError, match="has not been sampled yet"):
        DistributionEnsemble(scenario, size=100, rng=np.random.default_rng(0), plan=plan).assignments()


def test_scoped_sampling_empty_active_positions() -> None:
    """A guard whose branch_key never matches produces all-placeholder output.

    Exercises the ``n_branch == 0`` branch in ``_sample_with_default``:
    when no positions satisfy the guard, the per-scope path fills the
    full output array with the placeholder (np.nan for
    DistributionIndex).
    """
    mode, weather_bike, _weather_train, mv = _make_branch_abstract_mv()
    scenario = Scenario(mv)
    # mode's outcomes are {bike, train}, so a guard with branch_key="x"
    # never matches.  The bucket for weather_bike is then empty.
    unmatched_guard = RegionGuard(selector_node=mode.node, branch_key="x")
    plan = EvaluationPlan(
        model=mv,
        nodes_of_interest=(mv.outputs.throughput,),
        regions=(
            Region(nodes=(mode.node,), domain_axes=(), guards=()),
            Region(nodes=(weather_bike.node,), domain_axes=(), guards=(unmatched_guard,)),
        ),
        dependencies=(frozenset(), frozenset({0})),
    )
    ens = DistributionEnsemble(scenario, size=100, rng=np.random.default_rng(0), plan=plan)
    a = ens.assignments()
    # mode is in the shared bucket, sampled at every position.
    assert a[mode].shape == (100,)
    # weather_bike: no positions match, so all positions are sentinels.
    assert a[weather_bike].shape == (100,)
    assert np.all(np.isnan(a[weather_bike]))


# ---------------------------------------------------------------------------
# Three-level nested ModelVariant  (issue #177 — recursive partitioning)
# ---------------------------------------------------------------------------
#
# Outer: Mode.{bike(0.2), car(0.8)}
#   car → Policy.{strict(0.5), loose(0.5)}
#     strict → Fuel.{gas(0.5), elec(0.5)}
#
# Throughput (capacity = 100):
#   bike              → 100 * 1.0 = 100
#   car + strict + gas  → 100 * 2.2 = 220
#   car + strict + elec → 100 * 2.4 = 240
#   car + loose + *     → 100 * 3.0 = 300  (fuel irrelevant in loose branch)
#
# E[throughput] = 0.2*100 + 0.2*220 + 0.2*240 + 0.2*300 + 0.2*300 = 232.0
# (all 5 reachable leaf combos have equal probability 0.2)
# ---------------------------------------------------------------------------

_CAP_TRIPLE = 100.0


def _make_triple_nested_mv() -> tuple[CategoricalIndex, CategoricalIndex, CategoricalIndex, ModelVariant]:
    """Build Mode.{bike,car} × Policy.{strict,loose} × Fuel.{gas,elec}."""
    mode = CategoricalIndex("mode", {"bike": 0.2, "car": 0.8})
    policy = CategoricalIndex("policy", {"strict": 0.5, "loose": 0.5})
    fuel = CategoricalIndex("fuel", {"gas": 0.5, "elec": 0.5})
    cap = Index("cap", _CAP_TRIPLE)

    class _LeafModel(Model, legacy=True):
        @inputs
        class Inputs:
            cap: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, c: Index, mult: float) -> None:
            super().__init__(
                f"Leaf{mult}",
                inputs=_LeafModel.Inputs(cap=c),
                outputs=_LeafModel.Outputs(throughput=Index("throughput", c.node * mult)),
            )

    fuel_mv = ModelVariant("Fuel", {"gas": _LeafModel(cap, 2.2), "elec": _LeafModel(cap, 2.4)}, selector=fuel)
    policy_mv = ModelVariant("Policy", {"strict": fuel_mv, "loose": _LeafModel(cap, 3.0)}, selector=policy)
    outer_mv = ModelVariant("Mode", {"bike": _LeafModel(cap, 1.0), "car": policy_mv}, selector=mode)
    return mode, policy, fuel, outer_mv


def test_triple_nested_guard_chain_depths() -> None:
    """Recursive partitioning builds correct guard chain depths for each branch.

    gas/elec leaves: depth 3 (mode=car → policy=strict → fuel=gas/elec)
    loose branch:    depth 2 (mode=car → policy=loose)
    bike branch:     depth 1 (mode=bike)
    """
    *_, mv = _make_triple_nested_mv()
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan([mv.outputs.throughput], strategy="regional")

    gas_regions = [r for r in plan.regions if any(g.branch_key == "gas" for g in r.guards)]
    elec_regions = [r for r in plan.regions if any(g.branch_key == "elec" for g in r.guards)]
    loose_regions = [r for r in plan.regions if any(g.branch_key == "loose" for g in r.guards)]
    bike_regions = [r for r in plan.regions if any(g.branch_key == "bike" for g in r.guards)]

    assert gas_regions, "no gas region in plan"
    assert elec_regions, "no elec region in plan"
    assert loose_regions, "no loose region in plan"
    assert bike_regions, "no bike region in plan"

    assert tuple(g.branch_key for g in gas_regions[0].guards) == ("car", "strict", "gas")
    assert tuple(g.branch_key for g in elec_regions[0].guards) == ("car", "strict", "elec")
    assert tuple(g.branch_key for g in loose_regions[0].guards) == ("car", "loose")
    assert tuple(g.branch_key for g in bike_regions[0].guards) == ("bike",)


def test_triple_nested_regional_matches_expected_value() -> None:
    """Three-level regional plan produces the analytically correct expected throughput."""
    *_, mv = _make_triple_nested_mv()
    scenario = Scenario(mv)
    ev = Evaluation(scenario)
    plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    ens = CrossProductEnsemble(scenario, rng=np.random.default_rng(0))
    result = ev.execute_plan(plan, ens)
    ev_val = float(result.expected_value(mv.outputs.throughput))
    assert abs(ev_val - 232.0) < 1e-10, f"expected 232.0, got {ev_val}"


# ---------------------------------------------------------------------------
# Per-scope sampling (plan= on DistributionEnsemble) with ModelVariant
# ---------------------------------------------------------------------------
#
# Tests the integration path that inspect G Part 5 did not exercise:
# DistributionEnsemble(scenario, N, plan=plan) where plan is built with
# strategy="regional" from a ModelVariant with branch-local abstract indexes.
#
# Model:
#   mode    = CategoricalIndex(bike:0.5, car:0.5)    — shared
#   wind    = DistributionIndex(N(0, 2))             — bike-local
#   traffic = DistributionIndex(N(10, 1))            — car-local
#   cap     = 100.0
#
#   BikeModel:  throughput = cap * (1 + 0.01 * wind)
#   CarModel:   throughput = cap * (1 + 0.01 * traffic)
#
# Analytical: E[throughput] = 0.5*100*(1+0) + 0.5*100*(1+0.1) = 105.0
# ---------------------------------------------------------------------------

_SCOPED_MODE = CategoricalIndex("scoped_mode", {"bike": 0.5, "car": 0.5})
_SCOPED_WIND = DistributionIndex("wind", stats.norm, {"loc": 0.0, "scale": 2.0})
_SCOPED_TRAFFIC = DistributionIndex("traffic", stats.norm, {"loc": 10.0, "scale": 1.0})
_SCOPED_CAP = Index("cap", 100.0)


def _make_scoped_mv() -> tuple["ModelVariant", Scenario, Evaluation, "EvaluationPlan"]:
    """Build the shared scoped-sampling fixture."""

    class _BikeLeaf(Model, legacy=True):
        @inputs
        class Inputs:
            cap: Index
            wind: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, cap: Index, wind: Index) -> None:
            super().__init__(
                "BikeLeaf",
                inputs=self.Inputs(cap=cap, wind=wind),
                outputs=self.Outputs(throughput=Index("throughput", cap.node * (1.0 + wind.node * 0.01))),
            )

    class _CarLeaf(Model, legacy=True):
        @inputs
        class Inputs:
            cap: Index
            traffic: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, cap: Index, traffic: Index) -> None:
            super().__init__(
                "CarLeaf",
                inputs=self.Inputs(cap=cap, traffic=traffic),
                outputs=self.Outputs(throughput=Index("throughput", cap.node * (1.0 + traffic.node * 0.01))),
            )

    mv = ModelVariant(
        "ScopedTransport",
        {"bike": _BikeLeaf(_SCOPED_CAP, _SCOPED_WIND), "car": _CarLeaf(_SCOPED_CAP, _SCOPED_TRAFFIC)},
        selector=_SCOPED_MODE,
    )
    scenario = Scenario(mv)
    ev = Evaluation(scenario)
    plan = ev.build_plan([mv.outputs.throughput], strategy="regional")
    return mv, scenario, ev, plan


def test_scoped_ensemble_sentinels_at_inactive_branch_positions() -> None:
    """Per-scope sampling places NaN exactly at positions where a branch is inactive.

    With plan= set, DistributionEnsemble samples wind only at bike-mode positions
    and traffic only at car-mode positions; the complementary positions receive NaN.
    """
    _, scenario, _, plan = _make_scoped_mv()
    N = 400
    ens = DistributionEnsemble(scenario, N, rng=np.random.default_rng(7), plan=plan)
    a = ens.assignments()

    mode_arr = a[_SCOPED_MODE]
    wind_arr = np.asarray(a[_SCOPED_WIND], dtype=float)
    traffic_arr = np.asarray(a[_SCOPED_TRAFFIC], dtype=float)

    bike_mask = mode_arr == "bike"
    car_mask = mode_arr == "car"

    # wind: sampled at bike positions, NaN elsewhere
    assert np.all(~np.isnan(wind_arr[bike_mask])), "wind has NaN at active bike positions"
    assert np.all(np.isnan(wind_arr[car_mask])), "wind is not NaN at inactive car positions"
    # traffic: sampled at car positions, NaN elsewhere
    assert np.all(~np.isnan(traffic_arr[car_mask])), "traffic has NaN at active car positions"
    assert np.all(np.isnan(traffic_arr[bike_mask])), "traffic is not NaN at inactive bike positions"


def test_scoped_ensemble_regional_execution_correct() -> None:
    """Regional execution with a scoped ensemble is NaN-free and matches E[throughput]=105.

    Verifies that the sentinel NaN values in branch-local indexes never propagate
    into the output: the regional executor's branch mask ensures inactive positions
    are never read by the wrong branch formula.
    """
    mv, scenario, ev, plan = _make_scoped_mv()
    N = 500
    ens = DistributionEnsemble(scenario, N, rng=np.random.default_rng(42), plan=plan)
    result = ev.execute_plan(plan, ens)
    arr = np.asarray(result[mv.outputs.throughput]).ravel()

    assert np.all(np.isfinite(arr)), f"output contains NaN/inf at positions {np.where(~np.isfinite(arr))[0]}"

    ev_val = float(result.expected_value(mv.outputs.throughput))
    # E[throughput] = 0.5*100*(1+0.01*0) + 0.5*100*(1+0.01*10) = 105.0
    # MC std-error ≈ 5/sqrt(500) ≈ 0.22; ±2 is effectively deterministic
    assert abs(ev_val - 105.0) < 2.0, f"expected ≈ 105.0, got {ev_val}"


# ---------------------------------------------------------------------------
# list[str] selector override + regional plan: zero-scenario branch
# ---------------------------------------------------------------------------


def test_regional_list_str_override_restricts_active_branches() -> None:
    """A list[str] override on the selector restricts the active branches.

    The regional plan is built for all three branches (bike, car, train), but the
    scenario restricts the selector support to {bike, car}.  The train branch
    receives zero scenarios; the executor must handle this gracefully and produce
    the correct expected value for the two active branches.

    E[throughput | bike/car only] = 0.5 * 100 + 0.5 * 1000 = 550
    """
    mode = CategoricalIndex("mode", {"bike": 1 / 3, "car": 1 / 3, "train": 1 / 3})
    cap = Index("cap", 1.0)

    class _Leaf(Model, legacy=True):
        @inputs
        class Inputs:
            cap: Index

        @outputs
        class Outputs:
            throughput: Index

        def __init__(self, c: Index, mult: float) -> None:
            super().__init__(
                f"Leaf{mult}",
                inputs=self.Inputs(cap=c),
                outputs=self.Outputs(throughput=Index("throughput", c.node * mult)),
            )

    mv = ModelVariant(
        "Transport",
        {"bike": _Leaf(cap, 100.0), "car": _Leaf(cap, 1000.0), "train": _Leaf(cap, 500.0)},
        selector=mode,
    )
    scenario = Scenario(mv, overrides={mode: ["bike", "car"]})
    ev = Evaluation(Scenario(mv))
    plan = ev.build_plan([mv.outputs.throughput], strategy="regional")

    ens = CrossProductEnsemble(scenario)
    assert "train" not in set(ens.assignments()[mode].tolist()), "train should be absent from ensemble"

    result = ev.execute_plan(plan, ens)
    ev_val = float(result.expected_value(mv.outputs.throughput))
    assert abs(ev_val - 550.0) < 1e-6, f"expected 550.0, got {ev_val}"
