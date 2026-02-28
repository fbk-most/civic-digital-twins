"""Runnable snippets from docs/getting-started.md."""

import numpy as np

from civic_digital_twins.dt_model import DistributionEnsemble, Evaluation, Model
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import Index, TimeseriesIndex, UniformDistIndex

# ---------------------------------------------------------------------------
# getting-started.md §1 — Define the model
# ---------------------------------------------------------------------------

fuel_efficiency = UniformDistIndex("fuel_efficiency_km_l", loc=10.0, scale=5.0)
distance = UniformDistIndex("distance_km", loc=50.0, scale=30.0)

litres = Index("litres", distance / fuel_efficiency)
co2_per_litre = Index("co2_per_litre", 2.31)
co2 = Index("co2_kg", litres * co2_per_litre)

model = Model("co2_model", [fuel_efficiency, distance, litres, co2_per_litre, co2])

assert len(model.abstract_indexes()) == 2   # fuel_efficiency, distance
assert model.is_instantiated() is False


# ---------------------------------------------------------------------------
# getting-started.md §2 — Build an ensemble
# ---------------------------------------------------------------------------

ensemble = DistributionEnsemble(model, size=1000)


# ---------------------------------------------------------------------------
# getting-started.md §3 — Evaluate
# ---------------------------------------------------------------------------

result = Evaluation(model).evaluate(ensemble)

co2_samples = result[co2]            # np.ndarray, shape (1000, 1)
co2_mean = result.marginalize(co2)   # scalar

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
