# Getting Started

This guide walks through the two main usage patterns of the
`civic_digital_twins.dt_model` package:

1. **Direct pattern** — build a model entirely from built-in index types,
   sample uncertain parameters with `DistributionEnsemble`, and evaluate
   with `Evaluation`.
2. **Vertical extension pattern** — subclass `Model` and introduce
   domain-specific index types (context variables, presence variables,
   constraints) to evaluate a sustainability field on a multi-dimensional
   grid.

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

## Part 1 — Direct Pattern

The direct pattern is appropriate when uncertainty comes entirely from
probability distributions over scalar or timeseries parameters, with no
categorical context variables.

### 1.1 Define the model

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

### 1.2 Build an ensemble

`DistributionEnsemble` draws `size` independent samples from every
distribution-backed abstract index and yields equally-weighted scenarios:

```python
from civic_digital_twins.dt_model import DistributionEnsemble

ensemble = DistributionEnsemble(model, size=1000)
```

### 1.3 Evaluate

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

### 1.4 Timeseries and user-defined functions

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

## Part 2 — Vertical Extension Pattern

The vertical extension pattern adds domain-specific semantics on top of
the core model layer.  The overtourism domain introduces:

- **Context variables** — categorical or continuous factors outside the
  modeller's control (season, weather, day of the week, …).
- **Presence variables** — visitor counts, distributed according to the
  current context variable values.
- **Constraints** — named (usage formula, capacity) pairs; satisfaction
  of each constraint contributes to the sustainability field.

The classes below live in `examples/overtourism_molveno/overtourism_metamodel.py`
because they are domain-specific.  The core library provides the
foundation (`Index`, `Model`, `Evaluation`); the domain layer builds on top.

### 2.1 Context variables

```python
from overtourism_molveno.overtourism_metamodel import (
    CategoricalContextVariable,
    UniformCategoricalContextVariable,
)

# Season: weighted categorical
CV_season = CategoricalContextVariable(
    "season",
    {"low": 0.6, "high": 0.4},
)

# Weather: uniform categorical
CV_weather = UniformCategoricalContextVariable(
    "weather",
    ["good", "unsettled", "bad"],
)
```

A `ContextVariable` is an `Index` with `value=None` (it acts as a
placeholder); `OvertourismEnsemble` fills it in with a concrete value for
each scenario.

### 2.2 Presence variable

```python
from overtourism_molveno.overtourism_metamodel import PresenceVariable

# A mapping from (CV assignments) to (mean, std) of a truncated-normal
# presence distribution, keyed by (season_value, weather_value)
presence_stats = {
    ("low",  "good"):      (2_000,  500),
    ("low",  "unsettled"): (1_500,  400),
    ("low",  "bad"):       (1_000,  300),
    ("high", "good"):      (8_000, 2_000),
    ("high", "unsettled"): (6_000, 1_500),
    ("high", "bad"):       (4_000, 1_000),
}

PV_visitors = PresenceVariable(
    "visitors",
    [CV_season, CV_weather],
    presence_stats,
)
```

`PV_visitors` is also an `Index` with `value=None`.  In grid evaluation
it is provided as an *axis* (not resolved per-scenario), so it sweeps a
dense range of visitor counts.

### 2.3 Constraints

```python
from civic_digital_twins.dt_model import piecewise
from civic_digital_twins.dt_model.model.index import Index, TriangDistIndex
from overtourism_molveno.overtourism_metamodel import Constraint

# Capacity with uncertainty
I_C_beach = TriangDistIndex("beach_capacity", loc=3000.0, scale=2000.0, c=0.5)

# Usage factor: depends on context variable (bad weather reduces beach use)
I_U_beach_visitors = Index(
    "beach_usage_factor",
    piecewise((0.30, CV_weather == "bad"), (0.70, True)),
)

# Usage formula: visitors × usage_factor / capacity
C_beach = Constraint(
    name="beach",
    usage=Index("beach_usage", PV_visitors * I_U_beach_visitors),
    capacity=I_C_beach,
)
```

`piecewise((expr, cond), …)` builds a conditional formula node that the
engine evaluates lazily — the condition `CV_weather == "bad"` is a graph
node that resolves to `True` or `False` once `CV_weather` is assigned a
concrete value in a scenario.

### 2.4 OvertourismModel

```python
from overtourism_molveno.overtourism_metamodel import OvertourismModel

model = OvertourismModel(
    name="minimal_overtourism",
    cvs=[CV_season, CV_weather],
    pvs=[PV_visitors],
    indexes=[I_U_beach_visitors],
    capacities=[I_C_beach],
    constraints=[C_beach],
)
```

`OvertourismModel` automatically adds each constraint's usage index to the
flat `indexes` list so that `Evaluation` can find it.

### 2.5 Ensemble

```python
from overtourism_molveno.overtourism_metamodel import OvertourismEnsemble
from civic_digital_twins.dt_model.model.index import ContextVariable

scenario: dict[ContextVariable, list] = {
    CV_season:  ["low", "high"],
    CV_weather: ["good", "unsettled", "bad"],
}

ensemble = OvertourismEnsemble(model, scenario, cv_ensemble_size=10)
scenarios = list(ensemble)
# 2 × 3 × 10 = 60 weighted scenarios
```

For each combination of CV values, `OvertourismEnsemble` draws
`cv_ensemble_size` samples from every distribution-backed non-PV non-CV
abstract index (here: `I_C_beach`).

### 2.6 Grid evaluation

Presence variables are not resolved per-scenario; instead they define the
grid axes over which the sustainability field is computed:

```python
import numpy as np
from civic_digital_twins.dt_model import Evaluation
from civic_digital_twins.dt_model.model.index import Distribution

visitors_axis = np.linspace(0, 20_000, 201)

result = Evaluation(model).evaluate(
    scenarios,
    axes={PV_visitors: visitors_axis},
)
# result.full_shape == (201, 60)
```

### 2.7 Sustainability field

The sustainability field measures what fraction of the weighted scenario
population considers each visitor count sustainable:

```python
field = np.ones(visitors_axis.size)

for c in model.constraints:
    usage = np.broadcast_to(result[c.usage], result.full_shape)  # (201, 60)

    if isinstance(c.capacity.value, Distribution):
        # Probabilistic capacity: probability that usage ≤ capacity
        mask = 1.0 - c.capacity.value.cdf(usage)
    else:
        cap = np.broadcast_to(result[c.capacity], result.full_shape)
        mask = (usage <= cap).astype(float)

    # Marginalise over scenarios → shape (201,)
    field *= np.tensordot(mask, result.weights, axes=([-1], [0]))

# field[i] ∈ [0, 1]: sustainability score for visitors_axis[i] visitors
```

With a 2-D grid (tourists × excursionists) the same pattern extends
naturally — see
[`examples/overtourism_molveno/overtourism_molveno.py`](../examples/overtourism_molveno/overtourism_molveno.py)
for the full Molveno implementation.

---

## Next Steps

- Browse the full examples:
  - [`examples/mobility_bologna/`](../examples/mobility_bologna) — direct pattern with timeseries and a custom solver function.
  - [`examples/overtourism_molveno/`](../examples/overtourism_molveno) — complete vertical extension with four constraints and visualisation.
- Read the reference documentation:
  - [Engine layer](design/dd-cdt-engine.md) — graph nodes, topological sorting, NumPy executor.
  - [Model / simulation layer](design/dd-cdt-model.md) — `Model`, `Evaluation`, `EvaluationResult`, vertical extension pattern, design rationale.
