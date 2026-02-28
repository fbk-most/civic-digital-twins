# Getting Started

This guide walks through the **direct pattern** of the
`civic_digital_twins.dt_model` package: build a model entirely from
built-in index types, sample uncertain parameters with
`DistributionEnsemble`, and evaluate with `Evaluation`.

For the **vertical extension pattern** — which introduces domain-specific
context variables, presence variables, and constraints to compute a
sustainability field on a multi-dimensional grid — see
[`examples/overtourism_molveno/overtourism-getting-started.md`](../examples/overtourism_molveno/overtourism-getting-started.md).

Full working examples are in the
[`examples/`](../examples) directory:
[`examples/mobility_bologna/`](../examples/mobility_bologna)
uses the direct pattern;
[`examples/overtourism_molveno/`](../examples/overtourism_molveno)
uses the vertical extension pattern.

(For reference documentation on the model/simulation layer see
[`docs/design/dd-cdt-model.md`](design/dd-cdt-model.md); for the
engine layer see [`docs/design/dd-cdt-engine.md`](design/dd-cdt-engine.md).)

---

## 1 — Define the model

Define indexes as attributes of a `Model` subclass (or pass a list to
`Model` directly).  Use `UniformDistIndex`, `LognormDistIndex`, or
`TriangDistIndex` for uncertain parameters and plain `Index` for formulas
and constants.

```python
import numpy as np
from civic_digital_twins.dt_model import Model
from civic_digital_twins.dt_model.model.index import Index, UniformDistIndex

# Two uncertain parameters
fuel_efficiency = UniformDistIndex("fuel_efficiency_km_l", loc=10.0, scale=5.0)
distance         = UniformDistIndex("distance_km",          loc=50.0, scale=30.0)

# Derived formula: litres consumed
litres = Index("litres", distance / fuel_efficiency)

# CO2 factor is a known constant (2.31 kg CO2 per litre of petrol)
co2_per_litre = Index("co2_per_litre", 2.31)

# CO2 emitted
co2 = Index("co2_kg", litres * co2_per_litre)

model = Model("co2_model", [fuel_efficiency, distance, litres, co2_per_litre, co2])
```

The model is **abstract** because `fuel_efficiency` and `distance` are
distribution-backed:

```python
print(model.abstract_indexes())   # [fuel_efficiency, distance]
print(model.is_instantiated())    # False
```

## 2 — Build an ensemble

`DistributionEnsemble` draws `size` independent samples from every
distribution-backed abstract index and yields equally-weighted scenarios:

```python
from civic_digital_twins.dt_model import DistributionEnsemble

ensemble = DistributionEnsemble(model, size=1000)
```

## 3 — Evaluate

```python
from civic_digital_twins.dt_model import Evaluation

result = Evaluation(model).evaluate(ensemble)
```

`result` is an `EvaluationResult`.  Use `result[idx]` for the raw array
(shape `(S,)` here, one value per scenario) and `result.marginalize(idx)`
for the weighted expectation:

```python
# Distribution of CO2 across 1000 scenarios
co2_samples = result[co2]          # np.ndarray, shape (1000,)

# Expected (mean) CO2
co2_mean = result.marginalize(co2) # scalar
print(f"Expected CO2: {co2_mean:.1f} kg")
```

## 4 — Timeseries and user-defined functions

For time-indexed quantities use `TimeseriesIndex`.  If a computation
cannot be expressed as a graph formula (e.g. an iterative solver), wrap
it in a `function_call` node and register a `LambdaAdapter` at evaluation
time.

```python
import numpy as np
from civic_digital_twins.dt_model.model.index import TimeseriesIndex, Index
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import executor

# 24-hour demand time series (one value per hour)
demand_ts = TimeseriesIndex("demand", np.array([10.0, 12.0, 15.0, 14.0] * 6))

# A custom smoothing function applied as a graph node
smoothed = TimeseriesIndex(
    "smoothed_demand",
    graph.function_call("smooth", demand_ts.node),
)

# Register the implementation at evaluation time
result = Evaluation(model).evaluate(
    ensemble,
    functions={
        "smooth": executor.LambdaAdapter(
            lambda ts: np.convolve(ts, np.ones(3) / 3, mode="same")
        )
    },
)
```

---

## Next Steps

- Browse the full examples:
  - [`examples/mobility_bologna/`](../examples/mobility_bologna) — direct pattern with timeseries and a custom solver function.
  - [`examples/overtourism_molveno/`](../examples/overtourism_molveno) — vertical extension with four constraints and visualisation.
- Walk through the vertical extension pattern:
  - [`examples/overtourism_molveno/overtourism-getting-started.md`](../examples/overtourism_molveno/overtourism-getting-started.md)
- Read the reference documentation:
  - [Engine layer](design/dd-cdt-engine.md) — graph nodes, topological sorting, NumPy executor.
  - [Model / simulation layer](design/dd-cdt-model.md) — `Model`, `Evaluation`, `EvaluationResult`, vertical extension pattern, design rationale.
