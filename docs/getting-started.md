# Getting Started

|              | Document data                                  |
|--------------| ---------------------------------------------- |
| Author       | [@pistore](https://github.com/pistore)         |
| Last-Updated | 2026-04-04                                     |
| Status       | Draft                                          |
| Approved-By  | N/A                                            |

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
`Model` directly).  Use `DistributionIndex` for uncertain parameters and
plain `Index` for formulas and constants.

### Legacy flat-list API

> **Note:** The flat-list constructor is still supported but emits a
> `DeprecationWarning` as of v0.8.0.  Prefer the dataclass-based approach
> shown below.

```python
import numpy as np
from scipy import stats
from civic_digital_twins.dt_model import Model
from civic_digital_twins.dt_model.model.index import DistributionIndex, Index

# Two uncertain parameters
fuel_efficiency = DistributionIndex("fuel_efficiency_km_l", stats.uniform, {"loc": 10.0, "scale": 5.0})
distance        = DistributionIndex("distance_km",          stats.uniform, {"loc": 50.0, "scale": 30.0})

# Derived formula: litres consumed
litres = Index("litres", distance / fuel_efficiency)

# CO2 factor is a known constant (2.31 kg CO2 per litre of petrol)
co2_per_litre = Index("co2_per_litre", 2.31)

# CO2 emitted
co2 = Index("co2_kg", litres * co2_per_litre)

model = Model("co2_model", [fuel_efficiency, distance, litres, co2_per_litre, co2])
```

### Dataclass-based API (v0.8.0+)

The preferred approach declares `Inputs`, `Outputs`, and optionally `Expose`
as inner `@dataclass` classes, making the inter-model interface explicit and
machine-checkable.  Every `DistributionIndex` (or other `GenericIndex`)
received as a constructor parameter — or created internally and needed by the
ensemble — must be declared in `Inputs` so that `model.indexes` includes it
and `DistributionEnsemble` can sample it:

```python
from dataclasses import dataclass
from civic_digital_twins.dt_model import DistributionIndex, Index, Model
from scipy import stats

class Co2Model(Model):

    @dataclass
    class Inputs:
        fuel_efficiency: DistributionIndex
        distance:        DistributionIndex

    @dataclass
    class Outputs:
        litres:        Index
        co2_per_litre: Index
        co2:           Index

    def __init__(self) -> None:
        Inputs  = Co2Model.Inputs
        Outputs = Co2Model.Outputs

        inputs = Inputs(
            fuel_efficiency = DistributionIndex("fuel_efficiency_km_l", stats.uniform, {"loc": 10.0, "scale": 5.0}),
            distance =        DistributionIndex("distance_km",          stats.uniform, {"loc": 50.0, "scale": 30.0}),
        )

        litres        = Index("litres",        inputs.distance / inputs.fuel_efficiency)
        co2_per_litre = Index("co2_per_litre", 2.31)
        co2           = Index("co2_kg",        litres * co2_per_litre)

        super().__init__(
            "co2_model",
            inputs=inputs,
            outputs=Outputs(
                litres=litres,
                co2_per_litre=co2_per_litre,
                co2=co2,
            ),
        )

co2_model = Co2Model()
co2 = co2_model.outputs.co2   # access via contractual output
```

`co2_model.indexes` is derived automatically from `inputs` and `outputs` — no flat
list required.  `abstract_indexes()` and `is_instantiated()` work identically
regardless of which API was used:

```python
co2_model.abstract_indexes()   # → [fuel_efficiency, distance]
co2_model.is_instantiated()    # → False
```

## 2 — Build an ensemble

`DistributionEnsemble` draws `size` independent samples from every abstract
index and yields equally-weighted scenarios.  Two kinds of abstract index are
supported: `DistributionIndex` (sampled via its `scipy.stats` distribution) and
`CategoricalIndex` (sampled from its probability-weighted string outcomes):

```python
from civic_digital_twins.dt_model import DistributionEnsemble

ensemble = DistributionEnsemble(co2_model, size=1000)
```

## 3 — Evaluate

```python
from civic_digital_twins.dt_model import Evaluation

result = Evaluation(co2_model).evaluate(ensemble)
```

`result` is an `EvaluationResult`.  Use `result[idx]` for the raw array
(shape `(S, 1)` here — `S` ENSEMBLE samples, trailing 1 from the DOMAIN
placeholder) and `result.marginalize(idx)` for the weighted expectation:

```python
# Distribution of CO2 across 1000 scenarios
co2_samples = result[co2]          # np.ndarray, shape (1000, 1)

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
from civic_digital_twins.dt_model.model.index import TimeseriesIndex
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import executor

# 24-hour demand time series (one value per hour)
demand_ts = TimeseriesIndex("demand", np.array([10.0, 12.0, 15.0, 14.0] * 6))

# A custom smoothing function applied as a graph node
smoothed = TimeseriesIndex(
    "smoothed_demand",
    graph.function_call("smooth", demand_ts.node),
)

model    = ...  # define a suitable model that includes demand_ts and smoothed
ensemble = ...  # define a suitable ensemble (or pass [(1.0, {})] if there are no abstract indexes)

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

## 5 — Model modularity

For larger models, split the computation into sub-models using the
dataclass I/O API.  Each sub-model declares its `Inputs` and `Outputs` as
inner `@dataclass` classes; the root model wires them by passing outputs of
one sub-model into the constructor of the next.

See [docs/design/dd-cdt-modularity.md](design/dd-cdt-modularity.md)
for the full concept guide, including `ModelVariant`, decomposition
patterns, and a step-by-step Bologna walkthrough.

`ModelVariant` supports two selection modes:

* **Static mode** (`selector: str`) — the active sub-model is fixed at
  construction time.
* **Runtime mode** (`selector: CategoricalIndex | graph.Node`) — a merged
  computation graph is built so variant dispatch happens at evaluation time,
  enabling probabilistic selection via `DistributionEnsemble`.

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
  - [Model Modularity guide](design/dd-cdt-modularity.md) — sub-models, `ModelVariant`, and the dataclass I/O API.
