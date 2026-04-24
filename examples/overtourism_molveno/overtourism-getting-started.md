<!-- SPDX-License-Identifier: Apache-2.0 -->

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
from civic_digital_twins.dt_model import CategoricalIndex

# Season: weighted categorical
CV_season = CategoricalIndex(
    "season",
    {"low": 0.6, "high": 0.4},
)

# Weather: uniform categorical (probabilities provided explicitly)
CV_weather = CategoricalIndex(
    "weather",
    {"good": 1 / 3, "unsettled": 1 / 3, "bad": 1 / 3},
)
```

A `CategoricalIndex` is an `Index` with `value=None` (it acts as a
placeholder); `OvertourismEnsemble` fills it in with a concrete value for
each scenario.

## 2 — Presence variable

```python
from scipy import stats
from overtourism_molveno.overtourism_metamodel import PresenceVariable

def visitors_distribution(season, weather):
    """Return a uniform distribution for visitor presence."""
    presence_stats = {
        ("low",  "good"):      (1_500,  2_500),
        ("low",  "unsettled"): (1_100,  1_900),
        ("low",  "bad"):       (1_000,  1_300),
        ("high", "good"):      (6_000, 10_000),
        ("high", "unsettled"): (4_500,  7_500),
        ("high", "bad"):       (3_000,  5_000),
    }
    low, high = presence_stats[(season, weather)]
    return stats.uniform(loc=low, scale=high - low)

PV_visitors = PresenceVariable(
    "visitors",
    [CV_season, CV_weather],
    visitors_distribution,
)
```

`PV_visitors` is also an `Index` with `value=None`.  In grid evaluation
it is provided as an *axis* (not resolved per-scenario), so it sweeps a
dense range of visitor counts.

## 3 — Constraints

```python
from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, graph
from overtourism_molveno.overtourism_metamodel import Constraint

# Capacity with uncertainty
I_C_beach = DistributionIndex("beach_capacity", stats.triang, {"loc": 3000.0, "scale": 2000.0, "c": 0.5})

# Usage factor: depends on context variable (bad weather reduces beach use)
I_U_beach_visitors = Index(
    "beach_usage_factor",
    graph.piecewise((0.30, CV_weather == "bad"), (0.70, True)),
)

# Usage formula: visitors × usage_factor
C_beach = Constraint(
    name="beach",
    usage=Index("beach_usage", PV_visitors * I_U_beach_visitors),
    capacity=I_C_beach,
)
```

`graph.piecewise((expr, cond), …)` builds a conditional formula node that the
engine evaluates lazily — the condition `CV_weather == "bad"` is a graph
node that resolves to `True` or `False` once `CV_weather` is assigned a
concrete value in a scenario.

## 4 — Model

Define a `Model` subclass with `Inputs` and `Outputs` dataclasses that
declare the abstract-index contract.  Expose `.cvs`, `.pvs`, and
`.constraints` attributes so that `OvertourismEnsemble` and the
sustainability-field loop can find them.

```python
from dataclasses import dataclass

from civic_digital_twins.dt_model import CategoricalIndex, GenericIndex, Model
from overtourism_molveno.overtourism_metamodel import Constraint, PresenceVariable


class MinimalOvertourismModel(Model):
    @dataclass
    class Inputs:
        cvs: list[CategoricalIndex]
        pvs: list[PresenceVariable]
        domain_indexes: list[GenericIndex]
        capacities: list[GenericIndex]

    @dataclass
    class Outputs:
        usage_indexes: list[GenericIndex]

    def __init__(self, name, *, cvs, pvs, indexes, capacities, constraints):
        super().__init__(
            name,
            inputs=self.Inputs(cvs=cvs, pvs=pvs, domain_indexes=indexes, capacities=capacities),
            outputs=self.Outputs(usage_indexes=[c.usage for c in constraints]),
        )
        self.cvs = cvs
        self.pvs = pvs
        self.constraints = constraints


model = MinimalOvertourismModel(
    name="minimal_overtourism",
    cvs=[CV_season, CV_weather],
    pvs=[PV_visitors],
    indexes=[I_U_beach_visitors],
    capacities=[I_C_beach],
    constraints=[C_beach],
)
```

All abstract indexes (CVs, PVs, domain indexes, capacities) are declared in
`Inputs`; usage indexes in `Outputs`.  `OvertourismEnsemble` duck-types on
`.cvs` and `.pvs`; the sustainability-field loop uses `.constraints`.  For a
production model with multiple sub-models see `MolvenoModel` in
`molveno_model.py`.

## 5 — Ensemble

```python
from overtourism_molveno.overtourism_metamodel import OvertourismEnsemble

scenario: dict[CategoricalIndex, list[str]] = {
    CV_season:  ["low", "high"],
    CV_weather: ["good", "unsettled", "bad"],
}

ensemble = OvertourismEnsemble(model, scenario, cv_ensemble_size=10)
# 2 × 3 = 6 scenarios (all CV combinations enumerated)
```

`OvertourismEnsemble` implements `AxisEnsemble`: it enumerates all
combinations of CV values and materialises the results into a single batched
ENSEMBLE axis — here 2 × 3 = 6 scenarios, one per (season, weather) pair.
Each scenario also includes one sample of every distribution-backed
non-PV non-CV abstract index (here: `I_C_beach`).

The `cv_ensemble_size` parameter controls random sampling only when a CV's
support is too large (or continuous) to enumerate fully.  For the small
finite CVs above every value is enumerated and `cv_ensemble_size` is
unused.

## 6 — Grid evaluation

Presence variables are not resolved per-scenario; instead they define the
grid axes over which the sustainability field is computed:

```python
import numpy as np
from civic_digital_twins.dt_model import Evaluation

visitors_axis = np.linspace(0, 20_000, 201)

result = Evaluation(model).evaluate(
    ensemble=ensemble,
    parameters={PV_visitors: visitors_axis},
)
# result.full_shape == (201, 6)
```

## 7 — Sustainability field

The sustainability field measures what fraction of the weighted scenario
population considers each visitor count sustainable:

```python
from civic_digital_twins.dt_model import Distribution

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
