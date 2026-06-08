<!-- SPDX-License-Identifier: Apache-2.0 -->

# Getting Started

|              | Document data                                  |
|--------------| ---------------------------------------------- |
| Author       | [@pistore](https://github.com/pistore)         |
| Last-Updated | 2026-06-07                                     |
| Status       | Draft                                          |
| Approved-By  | N/A                                            |

This guide walks through the core workflow of the `civic_digital_twins.dt_model`
package: define a model with `@define`/`compute()`, sample uncertain parameters
with `DistributionEnsemble`, and evaluate with `Evaluation`.

For a more complex example that extends the framework with domain-specific
concepts (constraints, categorical context variables, a multi-dimensional
evaluation grid), see
[`examples/overtourism_molveno/overtourism-getting-started.md`](../examples/overtourism_molveno/overtourism-getting-started.md).

Full working examples are in the
[`examples/`](../examples) directory:
[`examples/mobility_bologna/`](../examples/mobility_bologna)
uses the direct pattern;
[`examples/overtourism_molveno/`](../examples/overtourism_molveno)
uses the vertical extension pattern.

(For reference documentation see
[`docs/design/dd-cdt-model.md`](design/dd-cdt-model.md),
[`docs/design/dd-cdt-modularity.md`](design/dd-cdt-modularity.md), and
[`docs/design/dd-cdt-simulation.md`](design/dd-cdt-simulation.md).)

---

## 1 — Define the model

Use `@define` to declare a model.  `compute()` builds the computation graph
from indexes; `@inputs` and `@outputs` declare the contractual interface.
Use `DistributionIndex` for uncertain parameters and plain `Index` for
formulas and constants:

```python
from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, Model, define, inputs, outputs

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
```

`co2_model.indexes` is derived automatically from `inputs` and `outputs` — no flat list
required.  `abstract_indexes()` and `is_instantiated()` work as expected:

```python
co2_model.abstract_indexes()   # → [fuel_efficiency, distance]
co2_model.is_instantiated()    # → False
```

> **Note:** Earlier API styles (`@dataclass` + `def __init__`, and `Model("name", [indexes…])`)
> still work but emit a `DeprecationWarning`.  Use `@define` + `compute()` for new code.
> See [dd-cdt-model.md](design/dd-cdt-model.md) for migration notes.

## 2 — Build an ensemble

Wrap the model in a `Scenario`, then build an ensemble.  `DistributionEnsemble`
draws `size` independent samples from every abstract index and yields
equally-weighted scenarios.  Both `DistributionIndex` (sampled via its
`scipy.stats` distribution) and `CategoricalIndex` (sampled from its
probability-weighted string outcomes) are supported:

```python
from civic_digital_twins.dt_model import DistributionEnsemble, Scenario

scenario = Scenario(co2_model)
ensemble = DistributionEnsemble(scenario, size=1000)
```

## 3 — Evaluate

```python
from civic_digital_twins.dt_model import Evaluation

result = Evaluation(scenario).evaluate(ensemble=ensemble)
```

`result` is an `EvaluationResult`.  Use `result[idx]` for the raw array
(shape `(S,)` here — `S` ENSEMBLE samples) and `result.expected_value(idx)`
for the weighted expectation:

```python
# Distribution of CO2 across 1000 scenarios
co2_samples = result[co2]          # np.ndarray, shape (1000,)

# Expected (mean) CO2
co2_mean = result.expected_value(co2) # scalar
print(f"Expected CO2: {co2_mean:.1f} kg")
```

## 4 — Timeseries and user-defined functions

For time-indexed quantities use `TimeseriesIndex`.  If a computation
cannot be expressed as a graph formula (e.g. an iterative solver), wrap
it in a `function_call` node and bind the implementation to the numpy
backend using `NumpyBackend.adapt()` at evaluation time.

```python
import numpy as np

from civic_digital_twins.dt_model import NumpyBackend, TimeseriesIndex, graph

# 24-hour demand time series (one value per hour)
demand_ts = TimeseriesIndex("demand", np.array([10.0, 12.0, 15.0, 14.0] * 6))

# A custom smoothing function applied as a graph node
smoothed = TimeseriesIndex(
    "smoothed_demand",
    graph.function_call("smooth", demand_ts),
)

model = ...  # define a suitable model that includes demand_ts and smoothed

# Register the implementation at evaluation time — no abstract indexes, so no ensemble needed
result = Evaluation(Scenario(model)).evaluate(
    functions={
        "smooth": NumpyBackend.adapt(
            lambda ts: np.convolve(ts, np.ones(3) / 3, mode="same")
        )
    },
)
```

---

## 5 — Model modularity

For larger models, split the computation into sub-models using `@define`/`compute()`.
Each sub-model declares its `Inputs` and `Outputs`; the root (composite) model wires
them by passing outputs of one sub-model into the constructor of the next, using
`legacy=True` to opt out of the `@define` constraint.

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
  - [Model / simulation layer](design/dd-cdt-model.md) — `Model`, `Evaluation`, `EvaluationResult`, design rationale.
  - [Modularity guide](design/dd-cdt-modularity.md) — sub-models, `ModelVariant`, and `@define`/`compute()`.
  - [Simulation guide](design/dd-cdt-simulation.md) — `Scenario`, `EvaluationHandle`, `ModelEvaluator`, incremental evaluation.
