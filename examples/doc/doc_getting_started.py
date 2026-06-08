# SPDX-License-Identifier: Apache-2.0
"""Runnable snippets from docs/getting-started.md."""

import warnings

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    DistributionEnsemble,
    DistributionIndex,
    Evaluation,
    Index,
    Model,
    NumpyBackend,
    Scenario,
    TimeseriesIndex,
    define,
    graph,
    inputs,
    outputs,
)

# ---------------------------------------------------------------------------
# getting-started.md §1 — Define the model (@define + compute())
# ---------------------------------------------------------------------------


@define("CO2 Model")
class Co2Model(Model):

    @inputs
    class Inputs:
        fuel_efficiency: DistributionIndex
        distance: DistributionIndex

    @outputs
    class Outputs:
        litres: Index
        co2_per_litre: Index
        co2: Index

    def compute(self, inputs: Inputs) -> Outputs:
        litres = Index("litres", inputs.distance / inputs.fuel_efficiency)
        co2_per_litre = Index("co2_per_litre", 2.31)
        co2 = Index("co2_kg", litres * co2_per_litre)
        return Co2Model.Outputs(litres=litres, co2_per_litre=co2_per_litre, co2=co2)


co2_model = Co2Model(inputs=Co2Model.Inputs(
    fuel_efficiency=DistributionIndex("fuel_efficiency_km_l", stats.uniform, {"loc": 10.0, "scale": 5.0}),
    distance=DistributionIndex("distance_km", stats.uniform, {"loc": 50.0, "scale": 30.0}),
))
co2 = co2_model.outputs.co2   # access via contractual output

assert len(co2_model.abstract_indexes()) == 2   # fuel_efficiency, distance
assert co2_model.is_instantiated() is False
assert co2_model.outputs.co2 is not None
assert len(co2_model.indexes) == 5              # indexes collected automatically


# ---------------------------------------------------------------------------
# getting-started.md §1 — abstract_indexes / is_instantiated (Block 01)
# ---------------------------------------------------------------------------

co2_model.abstract_indexes()   # → [fuel_efficiency, distance]
co2_model.is_instantiated()    # → False


# ---------------------------------------------------------------------------
# getting-started.md §2 — Build an ensemble
# ---------------------------------------------------------------------------

scenario = Scenario(co2_model)
ensemble = DistributionEnsemble(scenario, size=1000)


# ---------------------------------------------------------------------------
# getting-started.md §3 — Evaluate
# ---------------------------------------------------------------------------

result = Evaluation(scenario).evaluate(ensemble=ensemble)

co2_samples = result[co2]          # np.ndarray, shape (1000,)
co2_mean = result.expected_value(co2) # scalar

assert co2_samples.shape == (1000,)
assert 0 < co2_mean < 200, f"Unexpected CO2 mean: {co2_mean:.1f}"


# ---------------------------------------------------------------------------
# getting-started.md §4 — Timeseries and user-defined functions
# ---------------------------------------------------------------------------

# 24-hour demand time series (one value per hour)
demand_ts = TimeseriesIndex("demand", np.array([10.0, 12.0, 15.0, 14.0] * 6))

# A custom smoothing function applied as a graph node
smoothed = TimeseriesIndex(
    "smoothed_demand",
    graph.function_call("smooth", demand_ts),
)

# Build a small model that uses the timeseries indexes
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    model = Model("ts_model", [demand_ts, smoothed])

# No abstract indexes — omit ensemble (deterministic evaluation)
result = Evaluation(Scenario(model)).evaluate(
    functions={
        "smooth": NumpyBackend.adapt(
            lambda ts: np.convolve(ts, np.ones(3) / 3, mode="same")
        )
    },
)

smoothed_values = result[smoothed]
assert smoothed_values.shape[-1] == 24  # 24 time steps


if __name__ == "__main__":
    print(f"doc_getting_started.py: all snippets OK  (E[CO2] ≈ {co2_mean:.1f} kg)")
