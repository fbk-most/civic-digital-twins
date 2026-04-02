"""Runnable snippets from docs/getting-started.md."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import DistributionEnsemble, Evaluation, Model
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import DistributionIndex, Index, TimeseriesIndex

# ---------------------------------------------------------------------------
# getting-started.md §1 — Define the model (legacy flat-list API)
# ---------------------------------------------------------------------------

fuel_efficiency = DistributionIndex("fuel_efficiency_km_l", stats.uniform, {"loc": 10.0, "scale": 5.0})
distance = DistributionIndex("distance_km", stats.uniform, {"loc": 50.0, "scale": 30.0})

litres = Index("litres", distance / fuel_efficiency)
co2_per_litre = Index("co2_per_litre", 2.31)
co2 = Index("co2_kg", litres * co2_per_litre)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    model = Model("co2_model", [fuel_efficiency, distance, litres, co2_per_litre, co2])

assert len(model.abstract_indexes()) == 2   # fuel_efficiency, distance
assert model.is_instantiated() is False


# ---------------------------------------------------------------------------
# getting-started.md §1 — Define the model (dataclass-based API, v0.8.0+)
# ---------------------------------------------------------------------------


class Co2Model(Model):
    """CO2 model using the recommended dataclass-based API."""

    @dataclass
    class Inputs:
        """Uncertain parameters."""

        fuel_efficiency: DistributionIndex
        distance: DistributionIndex

    @dataclass
    class Outputs:
        """KPI outputs."""

        litres: Index
        co2_per_litre: Index
        co2: Index

    def __init__(self) -> None:
        Inputs = Co2Model.Inputs
        Outputs = Co2Model.Outputs

        inputs = Inputs(
            fuel_efficiency=DistributionIndex("fuel_efficiency_km_l2", stats.uniform, {"loc": 10.0, "scale": 5.0}),
            distance=DistributionIndex("distance_km2", stats.uniform, {"loc": 50.0, "scale": 30.0}),
        )

        litres = Index("litres2", inputs.distance / inputs.fuel_efficiency)
        co2_per_litre = Index("co2_per_litre2", 2.31)
        co2 = Index("co2_kg2", litres * co2_per_litre)

        super().__init__(
            "co2_model2",
            inputs=inputs,
            outputs=Outputs(
                litres=litres,
                co2_per_litre=co2_per_litre,
                co2=co2,
            ),
        )


co2_model2 = Co2Model()

assert len(co2_model2.abstract_indexes()) == 2   # fuel_efficiency, distance
assert co2_model2.is_instantiated() is False
assert co2_model2.outputs.co2 is not None
assert co2_model2.outputs.litres is not None
# indexes derived automatically from inputs + outputs — no flat list needed
assert len(co2_model2.indexes) == 5


# ---------------------------------------------------------------------------
# getting-started.md §2 — Build an ensemble
# ---------------------------------------------------------------------------

# Use the dataclass-based model for the rest of the walkthrough
ensemble = DistributionEnsemble(co2_model2, size=1000)


# ---------------------------------------------------------------------------
# getting-started.md §3 — Evaluate
# ---------------------------------------------------------------------------

result = Evaluation(co2_model2).evaluate(ensemble)

co2_out = co2_model2.outputs.co2
co2_samples = result[co2_out]            # np.ndarray, shape (1000, 1)
co2_mean = result.marginalize(co2_out)   # scalar

assert co2_samples.shape == (1000, 1)
# E[CO2] = E[distance / fuel_efficiency * 2.31]
# distance ~ U(50,80), fuel_efficiency ~ U(10,15)  → reasonable range
assert 0 < co2_mean < 200, f"Unexpected CO2 mean: {co2_mean:.1f}"


# ---------------------------------------------------------------------------
# getting-started.md §4 — Timeseries and user-defined functions
# ---------------------------------------------------------------------------

# 24-hour demand time series (one value per hour)
demand_ts = TimeseriesIndex("demand", np.array([10.0, 12.0, 15.0, 14.0] * 6))

# A custom smoothing function applied as a graph node
smoothed = TimeseriesIndex(
    "smoothed_demand",
    graph.function_call("smooth", demand_ts.node),
)

# Build a small model that uses the timeseries indexes
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    ts_model = Model("ts_model", [demand_ts, smoothed])

# Single scenario with no abstract indexes
ts_result = Evaluation(ts_model).evaluate(
    [(1.0, {})],
    functions={
        "smooth": executor.LambdaAdapter(
            lambda ts: np.convolve(ts, np.ones(3) / 3, mode="same")
        )
    },
)

smoothed_values = ts_result[smoothed]
assert smoothed_values.shape[-1] == 24   # 24 time steps


if __name__ == "__main__":
    print(f"doc_getting_started.py: all snippets OK  (E[CO2] ≈ {co2_mean:.1f} kg)")
