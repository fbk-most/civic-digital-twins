"""Runnable snippets from docs/design/dd-cdt-model.md."""
# SPDX-License-Identifier: Apache-2.0

import sys
import warnings
from pathlib import Path

# Add examples/ to sys.path so overtourism_molveno can be imported.
_examples_dir = Path(__file__).parent.parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from civic_digital_twins.dt_model import ConstIndex, Model  # noqa: E402

# ---------------------------------------------------------------------------
# Block 00: dd-cdt-model.md — Index Types: Index modes
# ---------------------------------------------------------------------------


def _demo_00_index_modes() -> None:
    """Block 00: Index modes."""
    from scipy import stats

    from civic_digital_twins.dt_model import ConstIndex, DistributionIndex, Index

    # Distribution-backed (abstract — must be resolved in each scenario)
    # Pass any scipy-compatible distribution callable and a params dict:
    cap_dist = DistributionIndex("capacity", stats.uniform, {"loc": 400.0, "scale": 200.0})

    mu = DistributionIndex("mu", stats.norm, {"loc": 0.5, "scale": 0.1})

    # Constant
    cap = ConstIndex("capacity", 500.0)

    # Formula referencing other indexes
    load = Index("load", mu * cap)

    # Explicit placeholder (resolved by the caller)
    demand = Index("demand", None)

    assert cap_dist.value is not None
    assert mu.value is not None
    assert cap.value == 500.0
    assert demand.value is None
    _ = load


# ---------------------------------------------------------------------------
# Block 01: dd-cdt-model.md — Index Types: CategoricalIndex
# ---------------------------------------------------------------------------


def _demo_01_categorical_index() -> None:
    """Block 01: CategoricalIndex placeholder."""
    from civic_digital_twins.dt_model import CategoricalIndex

    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})

    assert mode.value is None
    assert mode.support == ["bike", "train"]


# ---------------------------------------------------------------------------
# Block 03: dd-cdt-model.md — ConditionalCategoricalIndex
# ---------------------------------------------------------------------------


def _demo_03_conditional_categorical_index() -> None:
    """Block 03: ConditionalCategoricalIndex."""
    from civic_digital_twins.dt_model import CategoricalIndex, ConditionalCategoricalIndex

    cv_season = CategoricalIndex("season", {"low": 0.6, "high": 0.4})

    def weekend_probs(season):
        return {"yes": 0.4, "no": 0.6} if season == "low" else {"yes": 0.3, "no": 0.7}

    cv_weekend = ConditionalCategoricalIndex("weekend", [cv_season], ["yes", "no"], weekend_probs)

    assert cv_weekend.value is None  # abstract


# ---------------------------------------------------------------------------
# Block 04: dd-cdt-model.md — ConditionalDistributionIndex
# ---------------------------------------------------------------------------


def _demo_04_conditional_distribution_index() -> None:
    """Block 04: ConditionalDistributionIndex."""
    from scipy import stats

    from civic_digital_twins.dt_model import CategoricalIndex, ConditionalDistributionIndex

    cv_weather = CategoricalIndex("weather", {"good": 0.5, "bad": 0.5})

    def load_dist(weather):
        if weather == "good":
            return stats.uniform(loc=100.0, scale=200.0)
        return stats.uniform(loc=50.0, scale=100.0)

    pv_load = ConditionalDistributionIndex("load", [cv_weather], load_dist)

    assert pv_load.value is None  # abstract — resolved per-scenario by the ensemble


# ---------------------------------------------------------------------------
# Block 02: dd-cdt-model.md — TimeseriesIndex
# ---------------------------------------------------------------------------


def _demo_02_timeseries_index() -> None:
    """Block 02: TimeseriesIndex."""
    import numpy as np

    from civic_digital_twins.dt_model import TimeseriesIndex

    # Fixed time series
    flow = TimeseriesIndex("flow", np.array([10.0, 20.0, 30.0]))

    # Placeholder (externally supplied)
    demand_ts = TimeseriesIndex("demand_ts")

    assert flow.value is not None
    assert demand_ts.value is None


# ---------------------------------------------------------------------------
# Block 05: dd-cdt-model.md — Model: Recommended API abstract_indexes / is_instantiated
# ---------------------------------------------------------------------------


def _demo_05_recommended_api() -> None:
    """Block 05: Recommended API — abstract_indexes / is_instantiated."""
    from scipy import stats

    from civic_digital_twins.dt_model import DistributionIndex, Index, Model, define, inputs, outputs

    @define("Demo")
    class DemoModel(Model):

        @inputs
        class Inputs:
            x: DistributionIndex
            y: DistributionIndex

        @outputs
        class Outputs:
            z: Index

        def compute(self, inputs: Inputs) -> Outputs:
            z = Index("z", inputs.x + inputs.y)
            return DemoModel.Outputs(z=z)

    x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
    y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
    m = DemoModel(inputs=DemoModel.Inputs(x=x, y=y))
    assert len(m.abstract_indexes()) == 2
    assert any(idx is x for idx in m.abstract_indexes())
    assert any(idx is y for idx in m.abstract_indexes())
    assert m.is_instantiated() is False


# ---------------------------------------------------------------------------
# Block 06: dd-cdt-model.md — Scenario: what-if overrides
# ---------------------------------------------------------------------------


def _demo_06_scenario_overrides() -> None:
    """Block 06: Scenario what-if overrides."""
    from civic_digital_twins.dt_model import Index, Model, Scenario, define, inputs, outputs

    @define("Parking")
    class ParkingModel(Model):

        @inputs
        class Inputs:
            cost: Index

        @outputs
        class Outputs:
            cost: Index

        def compute(self, inputs: Inputs) -> Outputs:
            return ParkingModel.Outputs(cost=inputs.cost)

    cost = Index("cost", 8.0)
    model = ParkingModel(inputs=ParkingModel.Inputs(cost=cost))
    base = Scenario(model)  # uses the model's own value
    expensive = Scenario(model, overrides={cost: 12.0})  # what-if: cost = 12.0

    assert base.overrides == {}
    assert expensive.overrides == {cost: 12.0}


# ---------------------------------------------------------------------------
# Block 08: dd-cdt-model.md — Contract Warnings: filterwarnings
# ---------------------------------------------------------------------------


def _demo_08_contract_warnings() -> None:
    """Block 08: Contract warnings — filterwarnings."""
    import warnings

    from civic_digital_twins.dt_model import ModelContractWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ModelContractWarning)
        # All ModelContractWarning subclasses are now raised as hard errors.


# ---------------------------------------------------------------------------
# Block 12: dd-cdt-model.md — DistributionEnsemble
# ---------------------------------------------------------------------------


def _demo_12_distribution_ensemble() -> None:
    """Block 12: DistributionEnsemble."""
    from scipy import stats

    from civic_digital_twins.dt_model import (
        DistributionEnsemble, DistributionIndex, Index, Model, Scenario, define, inputs, outputs,
    )

    @define("Demo12")
    class DemoModel(Model):

        @inputs
        class Inputs:
            x: DistributionIndex
            y: DistributionIndex

        @outputs
        class Outputs:
            z: Index

        def compute(self, inputs: Inputs) -> Outputs:
            z = Index("z", inputs.x + inputs.y)
            return DemoModel.Outputs(z=z)

    x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
    y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
    model = DemoModel(inputs=DemoModel.Inputs(x=x, y=y))

    scenario = Scenario(model)
    ensemble = DistributionEnsemble(scenario, size=100)

    assert len(ensemble.ensemble_weights[0]) == 100
    weight = ensemble.ensemble_weights[0][0]
    assert abs(weight - 0.01) < 1e-12
    assignments = ensemble.assignments()
    assert x in assignments
    assert y in assignments


# ---------------------------------------------------------------------------
# Blocks 14 + 15: dd-cdt-model.md — Grid mode marginalize + End-to-End (1-D)
# ---------------------------------------------------------------------------


def _demo_14_15_end_to_end() -> None:
    """Blocks 14+15: Grid-mode marginalize + End-to-End Example (1-D mode)."""
    from scipy import stats

    from civic_digital_twins.dt_model import (
        DistributionEnsemble, DistributionIndex, Evaluation, Index, Model, Scenario, define, inputs, outputs,
    )

    @define("Demo")
    class DemoModel(Model):

        @inputs
        class Inputs:
            x: DistributionIndex
            y: DistributionIndex

        @outputs
        class Outputs:
            z: Index

        def compute(self, inputs: Inputs) -> Outputs:
            z = Index("z", inputs.x + inputs.y)
            return DemoModel.Outputs(z=z)

    # Define the model
    x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
    y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
    model = DemoModel(inputs=DemoModel.Inputs(x=x, y=y))
    scenario = Scenario(model)

    # Build an ensemble of 200 scenarios
    ensemble = DistributionEnsemble(scenario, size=200)

    # Evaluate
    result = Evaluation(scenario).evaluate(ensemble=ensemble)

    # Weighted mean of z across all scenarios
    print(result.expected_value(model.outputs.z))  # ≈ 10.0

    # shape (N₀, N₁, …, S) → (N₀, N₁, …)
    idx = model.outputs.z
    marginalised = result.expected_value(idx)
    assert 7.0 < marginalised < 13.0, f"Expected ~10, got {marginalised}"


# ---------------------------------------------------------------------------
# Block 17: dd-cdt-model.md — Constraint dataclass
# ---------------------------------------------------------------------------


def _demo_17_constraint() -> None:
    """Block 17: Constraint dataclass definition."""
    from dataclasses import dataclass

    from civic_digital_twins.dt_model import Index

    @dataclass(eq=False)
    class Constraint:
        name: str
        usage: Index  # formula-mode index for usage
        capacity: Index  # constant or distribution-backed capacity

    usage = Index("usage_demo", 1.0)
    cap = ConstIndex("cap_demo", 100.0)
    c = Constraint("demo", usage, cap)
    assert c.name == "demo"


# ---------------------------------------------------------------------------
# Blocks 18 + 19: dd-cdt-model.md — CrossProductEnsemble + Grid Evaluation
# ---------------------------------------------------------------------------


def _demo_18_19_overtourism() -> None:
    """Blocks 18+19: CrossProductEnsemble + Grid Evaluation."""
    import numpy as np
    from overtourism_molveno.molveno_model import Constraint
    from scipy import stats

    from civic_digital_twins.dt_model import (
        CategoricalIndex,
        ConditionalDistributionIndex,
        ConstIndex,
        CrossProductEnsemble,
        Distribution,
        Evaluation,
        GenericIndex,
        Index,
        Model,
        Scenario,
        inputs,
        outputs,
    )

    CV_weather = CategoricalIndex(
        "weather",
        {"good": 0.5, "unsettled": 0.3, "bad": 0.2},
    )

    def tourist_dist(w):
        del w
        return stats.uniform(loc=4000.0, scale=2000.0)

    def excursionist_dist(w):
        del w
        return stats.uniform(loc=2500.0, scale=1000.0)

    PV_tourists = ConditionalDistributionIndex("tourists", [CV_weather], tourist_dist)
    PV_excursionists = ConditionalDistributionIndex("excursionists", [CV_weather], excursionist_dist)

    usage_idx = Index("usage", PV_tourists + PV_excursionists)
    capacity_idx = ConstIndex("capacity_idx", 100_000.0)
    c_beach = Constraint("beach", usage_idx, capacity_idx)

    class _MinimalModel(Model, legacy=True):
        @inputs
        class Inputs:
            cvs: list[CategoricalIndex]
            pvs: list[ConditionalDistributionIndex]
            capacities: list[GenericIndex]

        @outputs
        class Outputs:
            usage_indexes: list[GenericIndex]

        def __init__(self, cvs, pvs, capacities, constraints):
            super().__init__(
                "demo",
                inputs=self.Inputs(cvs=cvs, pvs=pvs, capacities=capacities),
                outputs=self.Outputs(usage_indexes=[c.usage for c in constraints]),
            )
            self.cvs = cvs
            self.pvs = pvs
            self.constraints = constraints

    model = _MinimalModel(
        cvs=[CV_weather],
        pvs=[PV_tourists, PV_excursionists],
        capacities=[capacity_idx],
        constraints=[c_beach],
    )

    # Block 18 — CrossProductEnsemble
    scenario = Scenario(
        model,
        overrides={CV_weather: ["good", "unsettled", "bad"]},
        parameter_axes=[PV_tourists, PV_excursionists],
    )
    ensemble = CrossProductEnsemble(
        scenario,
        max_categorical_size=20,
    )

    # Block 19 — Grid Evaluation with CrossProductEnsemble
    tt = np.linspace(0, 50_000, 101)  # tourist presence axis
    ee = np.linspace(0, 50_000, 101)  # excursionist presence axis

    result = Evaluation(scenario).evaluate(
        ensemble=ensemble,
        parameters={PV_tourists: tt, PV_excursionists: ee},
    )

    # Compute sustainability field per constraint
    field = np.ones((tt.size, ee.size))
    for c in model.constraints:
        usage = np.broadcast_to(result[c.usage], result.full_shape)
        if isinstance(c.capacity.value, Distribution):
            mask = 1.0 - c.capacity.value.cdf(usage)
        else:
            cap = np.broadcast_to(result[c.capacity], result.full_shape)
            mask = (usage <= cap).astype(float)
        field *= np.tensordot(mask, result.weights, axes=([-1], [0]))

    assert field.shape == (tt.size, ee.size)


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------

_demo_00_index_modes()
_demo_01_categorical_index()
_demo_03_conditional_categorical_index()
_demo_04_conditional_distribution_index()
_demo_02_timeseries_index()
_demo_05_recommended_api()
_demo_06_scenario_overrides()
_demo_08_contract_warnings()
_demo_12_distribution_ensemble()
_demo_14_15_end_to_end()
_demo_17_constraint()
_demo_18_19_overtourism()

if __name__ == "__main__":
    print("doc_model.py: all snippets OK")
