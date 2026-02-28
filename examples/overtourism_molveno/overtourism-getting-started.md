# Getting Started with the Overtourism Model

This guide walks through the **vertical extension pattern** of the
`civic_digital_twins.dt_model` package, using the overtourism domain as
the running example.

The vertical extension pattern adds domain-specific semantics on top of
the core model layer.  The overtourism domain introduces:

- **Context variables** — categorical or continuous factors outside the
  modeller's control (season, weather, day of the week, …).
- **Presence variables** — visitor counts, distributed according to the
  current context variable values.
- **Constraints** — named (usage formula, capacity) pairs; satisfaction
  of each constraint contributes to the sustainability field.

The classes below live in
[`overtourism_metamodel.py`](overtourism_metamodel.py)
because they are domain-specific.  The core library provides the
foundation (`Index`, `Model`, `Evaluation`); the domain layer builds on top.

For the **direct pattern** (no context variables, plain distribution
sampling) see [`docs/getting-started.md`](../../docs/getting-started.md).

(For reference documentation on the model/simulation layer see
[`docs/design/dd-cdt-model.md`](../../docs/design/dd-cdt-model.md); for the
engine layer see
[`docs/design/dd-cdt-engine.md`](../../docs/design/dd-cdt-engine.md).)

---

## 1 — Context variables

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

## 2 — Presence variable

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

## 3 — Constraints

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

# Usage formula: visitors × usage_factor
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

## 4 — OvertourismModel

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

## 5 — Ensemble

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

## 6 — Grid evaluation

Presence variables are not resolved per-scenario; instead they define the
grid axes over which the sustainability field is computed:

```python
import numpy as np
from civic_digital_twins.dt_model import Evaluation

visitors_axis = np.linspace(0, 20_000, 201)

result = Evaluation(model).evaluate(
    scenarios,
    axes={PV_visitors: visitors_axis},
)
# result.full_shape == (201, 60)
```

## 7 — Sustainability field

The sustainability field measures what fraction of the weighted scenario
population considers each visitor count sustainable:

```python
from civic_digital_twins.dt_model.model.index import Distribution

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
[`overtourism_molveno.py`](overtourism_molveno.py)
for the full Molveno implementation.

---

## Next Steps

- Browse the full Molveno example: [`overtourism_molveno.py`](overtourism_molveno.py) — four constraints, 2-D grid, visualisation.
- Read the reference documentation:
  - [Model / simulation layer](../../docs/design/dd-cdt-model.md) — `Model`, `Evaluation`, `EvaluationResult`, vertical extension pattern, design rationale.
  - [Engine layer](../../docs/design/dd-cdt-engine.md) — graph nodes, topological sorting, NumPy executor.
