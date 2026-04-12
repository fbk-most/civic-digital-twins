"""End-to-end evaluation tests for ModelVariant with CategoricalIndex selector."""

# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest

from civic_digital_twins.dt_model.model.index import CategoricalIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.model.model_variant import ModelVariant
from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble, WeightedScenario
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation

# ---------------------------------------------------------------------------
# Simple models with known output values
# ---------------------------------------------------------------------------


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


# Shared capacity value wired into both models.
_CAPACITY_VALUE = 100.0


def _make_mv_with_mode(mode: CategoricalIndex) -> ModelVariant:
    cap_bike = Index("capacity", _CAPACITY_VALUE)
    cap_train = Index("capacity", _CAPACITY_VALUE)
    return ModelVariant(
        "Transport",
        {"bike": _BikeModel(cap_bike), "train": _TrainModel(cap_train)},
        selector=mode,
    )


# ===========================================================================
# DistributionEnsemble handles CategoricalIndex
# ===========================================================================


def test_distribution_ensemble_accepts_categorical_index():
    """DistributionEnsemble does not raise when model has only a CategoricalIndex."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv_with_mode(mode)
    ens = DistributionEnsemble(mv, size=10, rng=np.random.default_rng(0))
    scenarios = list(ens)
    assert len(scenarios) == 10


def test_distribution_ensemble_assigns_mode_in_every_scenario():
    """Every scenario's assignments dict contains the CategoricalIndex."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv_with_mode(mode)
    ens = DistributionEnsemble(mv, size=20, rng=np.random.default_rng(1))
    for _weight, assignments in ens:
        assert mode in assignments


def test_distribution_ensemble_mode_values_are_valid_keys():
    """Mode values assigned by the ensemble are always in the support."""
    mode = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    mv = _make_mv_with_mode(mode)
    ens = DistributionEnsemble(mv, size=50, rng=np.random.default_rng(2))
    for _weight, assignments in ens:
        val = assignments[mode]
        # val is a 1-element array like np.array(["bike"])
        assert val.item() in mode.support


# ===========================================================================
# Evaluation dispatch — correct output per variant
# ===========================================================================


def test_evaluation_bike_scenario_gives_bike_outputs():
    """A scenario that selects 'bike' gets BikeModel throughput (capacity * 1)."""
    mode = CategoricalIndex("mode", {"bike": 1.0})  # always bike
    mv = _make_mv_with_mode(mode)
    ens = DistributionEnsemble(mv, size=4, rng=np.random.default_rng(0))
    ev = Evaluation(mv)
    result = ev.evaluate(ens, [mv.outputs.throughput, mv.outputs.emissions])
    # All scenarios use bike: throughput = capacity * 1 = 100
    throughput = result.marginalize(mv.outputs.throughput)
    assert float(throughput) == pytest.approx(_CAPACITY_VALUE * 1.0)


def test_evaluation_train_scenario_gives_train_outputs():
    """A scenario that selects 'train' gets TrainModel throughput (capacity * 10)."""
    mode = CategoricalIndex("mode", {"train": 1.0})  # always train
    mv = _make_mv_with_mode(mode)
    ens = DistributionEnsemble(mv, size=4, rng=np.random.default_rng(0))
    ev = Evaluation(mv)
    result = ev.evaluate(ens, [mv.outputs.throughput, mv.outputs.emissions])
    throughput = result.marginalize(mv.outputs.throughput)
    assert float(throughput) == pytest.approx(_CAPACITY_VALUE * 10.0)


def test_evaluation_mixed_modes_weighted_average():
    """Mixed bike/train scenarios give a correctly weighted average throughput."""
    # Degenerate ensemble: 2 bike and 2 train out of 4; force via rng that picks
    # 50/50 mode.  Rather than relying on randomness, build a manual ensemble.
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = _make_mv_with_mode(mode)

    # Build a deterministic 4-scenario ensemble: 2 bike, 2 train.
    bike_val = np.array(["bike"])
    train_val = np.array(["train"])
    manual_scenarios: list[WeightedScenario] = [
        (0.25, {mode: bike_val}),
        (0.25, {mode: train_val}),
        (0.25, {mode: bike_val}),
        (0.25, {mode: train_val}),
    ]

    ev = Evaluation(mv)
    result = ev.evaluate(manual_scenarios, [mv.outputs.throughput])
    # Expected: 0.5 * (100 * 1) + 0.5 * (100 * 10) = 50 + 500 = 550
    throughput = result.marginalize(mv.outputs.throughput)
    assert float(throughput) == pytest.approx(550.0)


def test_evaluation_emissions_bike_only():
    """BikeModel always has emissions=0."""
    mode = CategoricalIndex("mode", {"bike": 1.0})
    mv = _make_mv_with_mode(mode)
    ens = DistributionEnsemble(mv, size=5, rng=np.random.default_rng(0))
    ev = Evaluation(mv)
    result = ev.evaluate(ens, [mv.outputs.emissions])
    assert float(result.marginalize(mv.outputs.emissions)) == pytest.approx(0.0)


def test_evaluation_emissions_train_only():
    """TrainModel always has emissions=50."""
    mode = CategoricalIndex("mode", {"train": 1.0})
    mv = _make_mv_with_mode(mode)
    ens = DistributionEnsemble(mv, size=5, rng=np.random.default_rng(0))
    ev = Evaluation(mv)
    result = ev.evaluate(ens, [mv.outputs.emissions])
    assert float(result.marginalize(mv.outputs.emissions)) == pytest.approx(50.0)


# ===========================================================================
# selector_index in EvaluationResult
# ===========================================================================


def test_selector_index_accessible_in_result():
    """result[mv._selector_index] returns a (S,) string array of variant keys."""
    mode = CategoricalIndex("mode", {"bike": 1.0})
    mv = _make_mv_with_mode(mode)
    ens = DistributionEnsemble(mv, size=3, rng=np.random.default_rng(0))
    ev = Evaluation(mv)
    result = ev.evaluate(ens, [mv._selector_index, mv.outputs.throughput])
    arr = result[mv._selector_index]
    # Should be shape (S,) with all entries "bike"
    # (No trailing DOMAIN placeholder in non-timeseries models after bug fix #155.)
    assert arr.shape == (3,)
    assert all(v == "bike" for v in arr.ravel())


# ===========================================================================
# Grid mode (axes) — runtime variant with numeric axis
# ===========================================================================


def _make_presence_mv(mode: CategoricalIndex) -> tuple[Index, ModelVariant]:
    """ModelVariant where both sub-models scale a shared presence axis."""
    presence = Index("presence", None)
    mv = ModelVariant(
        "Transport",
        {"bike": _BikeModel(presence), "train": _TrainModel(presence)},
        selector=mode,
    )
    return presence, mv


def test_grid_mode_categorical_selector_all_bike():
    """CategoricalIndex selector (all-bike) + numeric axis: throughput = presence * 1."""
    mode = CategoricalIndex("mode", {"bike": 1.0})
    presence, mv = _make_presence_mv(mode)
    xs = np.array([100.0, 200.0, 300.0])

    # mode is a non-axis abstract; presence is the axis → single scenario needs mode assignment.
    manual_scenarios: list[WeightedScenario] = [(1.0, {mode: np.array(["bike"])})]
    result = Evaluation(mv).evaluate(manual_scenarios, [mv.outputs.throughput], parameters={presence: xs})

    assert np.allclose(result.marginalize(mv.outputs.throughput), xs * 1.0)


def test_grid_mode_categorical_selector_all_train():
    """CategoricalIndex selector (all-train) + numeric axis: throughput = presence * 10."""
    mode = CategoricalIndex("mode", {"train": 1.0})
    presence, mv = _make_presence_mv(mode)
    xs = np.array([100.0, 200.0, 300.0])

    manual_scenarios: list[WeightedScenario] = [(1.0, {mode: np.array(["train"])})]
    result = Evaluation(mv).evaluate(manual_scenarios, [mv.outputs.throughput], parameters={presence: xs})

    assert np.allclose(result.marginalize(mv.outputs.throughput), xs * 10.0)


def test_grid_mode_categorical_selector_mixed_scenarios():
    """CategoricalIndex selector, two equal-weight scenarios + numeric axis.

    Weighted mean throughput = presence * (0.5*1 + 0.5*10) = presence * 5.5.
    """
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    presence, mv = _make_presence_mv(mode)
    xs = np.array([100.0, 200.0, 300.0])

    manual_scenarios: list[WeightedScenario] = [
        (0.5, {mode: np.array(["bike"])}),
        (0.5, {mode: np.array(["train"])}),
    ]
    result = Evaluation(mv).evaluate(manual_scenarios, [mv.outputs.throughput], parameters={presence: xs})

    assert np.allclose(result.marginalize(mv.outputs.throughput), xs * 5.5)


def test_grid_mode_node_selector_from_axis():
    """graph.Node selector derived from numeric axis.

    Selector: presence > 150 → 'train', else 'bike'.
    Single dummy scenario (no non-axis abstract indexes).
    """
    presence = Index("presence", None)
    selector = ModelVariant.guards_to_selector(
        [
            ("train", presence.node > 150.0),
            ("bike", True),
        ]
    )
    mv = ModelVariant(
        "Transport",
        {"bike": _BikeModel(presence), "train": _TrainModel(presence)},
        selector=selector,
    )
    xs = np.array([100.0, 200.0, 300.0])

    # presence is the only abstract index and it is the axis → no assignments needed.
    result = Evaluation(mv).evaluate([(1.0, {})], [mv.outputs.throughput], parameters={presence: xs})

    # presence=100 → bike → 100*1=100
    # presence=200 → train → 200*10=2000
    # presence=300 → train → 300*10=3000
    assert np.allclose(result.marginalize(mv.outputs.throughput), [100.0, 2000.0, 3000.0])
