"""Runnable snippets from docs/design/dd-cdt-model.md."""

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

# Add examples/ to sys.path so overtourism_molveno can be imported.
_examples_dir = Path(__file__).parent.parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from civic_digital_twins.dt_model import Evaluation, Model, ModelContractWarning
from civic_digital_twins.dt_model.model.index import (
    ConstIndex,
    DistributionIndex,
    Index,
    TimeseriesIndex,
)
from civic_digital_twins.dt_model.model.index import Distribution
from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble


# ---------------------------------------------------------------------------
# Block 00: dd-cdt-model.md — Index Types: Index modes
# ---------------------------------------------------------------------------


def _demo_00_index_modes() -> None:
    """Block 00: Index modes."""
    from scipy import stats
    from civic_digital_twins.dt_model.model.index import (
        ConstIndex, DistributionIndex, Index,
    )

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
# Block 02: dd-cdt-model.md — TimeseriesIndex
# ---------------------------------------------------------------------------


def _demo_02_timeseries_index() -> None:
    """Block 02: TimeseriesIndex."""
    import numpy as np
    from civic_digital_twins.dt_model.model.index import TimeseriesIndex

    # Fixed time series
    flow = TimeseriesIndex("flow", np.array([10.0, 20.0, 30.0]))

    # Placeholder (externally supplied)
    demand_ts = TimeseriesIndex("demand_ts")

    assert flow.value is not None
    assert demand_ts.value is None


# ---------------------------------------------------------------------------
# Block 05: dd-cdt-model.md — Model: Legacy indexes= API
# ---------------------------------------------------------------------------


def _demo_05_legacy_api() -> None:
    """Block 05: Legacy indexes= API."""
    from scipy import stats
    from civic_digital_twins.dt_model.model.model import Model
    from civic_digital_twins.dt_model.model.index import DistributionIndex, Index

    x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
    y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
    z = Index("z", x + y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = Model("demo", [x, y, z])   # DeprecationWarning

    assert len(model.abstract_indexes()) == 2
    assert model.is_instantiated() is False


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
    from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble
    from civic_digital_twins.dt_model.model.index import DistributionIndex, Index

    x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
    y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
    z = Index("z", x + y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = Model("demo12", [x, y, z])

    ensemble = DistributionEnsemble(model, size=100)
    scenarios = list(ensemble)

    assert len(scenarios) == 100
    weight, assignments = scenarios[0]
    assert abs(weight - 0.01) < 1e-12
    assert x in assignments
    assert y in assignments


# ---------------------------------------------------------------------------
# Blocks 14 + 15: dd-cdt-model.md — Grid mode marginalize + End-to-End (1-D)
# ---------------------------------------------------------------------------


def _demo_14_15_end_to_end() -> None:
    """Blocks 14+15: Grid-mode marginalize + End-to-End Example (1-D mode)."""
    import numpy as np
    from scipy import stats
    from civic_digital_twins.dt_model import Evaluation, Model
    from civic_digital_twins.dt_model.model.index import DistributionIndex, Index
    from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

    # Define the model
    x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
    y = DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 10.0})
    z = Index("z", x + y)
    model = Model("demo", [x, y, z])

    # Build an ensemble of 200 scenarios
    ensemble = DistributionEnsemble(model, size=200)

    # Evaluate
    result = Evaluation(model).evaluate(ensemble)

    # Weighted mean of z across all scenarios
    print(result.marginalize(z))  # ≈ 10.0

    # shape (N₀, N₁, …, S) → (N₀, N₁, …)
    idx = z
    marginalised = result.marginalize(idx)
    assert 7.0 < marginalised < 13.0, f"Expected ~10, got {marginalised}"


# ---------------------------------------------------------------------------
# Block 17: dd-cdt-model.md — Constraint dataclass
# ---------------------------------------------------------------------------


def _demo_17_constraint() -> None:
    """Block 17: Constraint dataclass definition."""
    from dataclasses import dataclass

    @dataclass(eq=False)
    class Constraint:
        name: str
        usage: Index     # formula-mode index for usage
        capacity: Index  # constant or distribution-backed capacity

    usage = Index("usage_demo", 1.0)
    cap = ConstIndex("cap_demo", 100.0)
    c = Constraint("demo", usage, cap)
    assert c.name == "demo"


# ---------------------------------------------------------------------------
# Blocks 18 + 19 + 20: dd-cdt-model.md — OvertourismModel + Grid Evaluation
# ---------------------------------------------------------------------------


def _demo_18_20_overtourism() -> None:
    """Blocks 18+19+20: OvertourismModel attributes + Grid Evaluation."""
    import numpy as np
    from civic_digital_twins.dt_model import Evaluation
    from civic_digital_twins.dt_model.model.index import ConstIndex, Distribution, Index
    from overtourism_molveno.overtourism_metamodel import (
        CategoricalContextVariable,
        Constraint,
        OvertourismEnsemble,
        OvertourismModel,
        PresenceVariable,
    )

    CV_weather = CategoricalContextVariable(
        "weather",
        {"good": 0.5, "unsettled": 0.3, "bad": 0.2},
    )

    PV_tourists = PresenceVariable(
        "tourists", [CV_weather], lambda w: {"mean": 5000.0, "std": 1000.0}
    )
    PV_excursionists = PresenceVariable(
        "excursionists", [CV_weather], lambda w: {"mean": 3000.0, "std": 500.0}
    )

    usage_idx = Index("usage", PV_tourists + PV_excursionists)
    capacity_idx = ConstIndex("capacity_idx", 100_000.0)
    c_beach = Constraint("beach", usage_idx, capacity_idx)

    model = OvertourismModel(
        "demo",
        cvs=[CV_weather],
        pvs=[PV_tourists, PV_excursionists],
        indexes=[],
        capacities=[capacity_idx],
        constraints=[c_beach],
    )

    # Block 18 — OvertourismModel attribute access
    model.cvs          # list[ContextVariable]
    model.pvs          # list[PresenceVariable]
    model.domain_indexes   # list[Index]  (e.g. scaling factors)
    model.capacities   # list[Index]      (capacity indexes)
    model.constraints  # list[Constraint]

    # Block 19 — OvertourismEnsemble
    ensemble = OvertourismEnsemble(
        model,
        {CV_weather: ["good", "unsettled", "bad"]},
        cv_ensemble_size=20,
    )

    # Block 20 — Grid Evaluation with OvertourismEnsemble
    tt = np.linspace(0, 50_000, 101)   # tourist presence axis
    ee = np.linspace(0, 50_000, 101)   # excursionist presence axis

    result = Evaluation(model).evaluate(
        list(ensemble),
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
_demo_02_timeseries_index()
_demo_05_legacy_api()
_demo_08_contract_warnings()
_demo_12_distribution_ensemble()
_demo_14_15_end_to_end()
_demo_17_constraint()
_demo_18_20_overtourism()

if __name__ == "__main__":
    print("doc_model.py: all snippets OK")
