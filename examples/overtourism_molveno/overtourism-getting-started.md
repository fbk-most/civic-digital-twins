<!-- SPDX-License-Identifier: Apache-2.0 -->

# Getting Started with the Overtourism Model

This guide walks through the **context-variable pattern** of the
`civic_digital_twins.dt_model` package, using the overtourism domain as
the running example.  The pattern applies whenever a model has:

- **Context variables** — categorical factors outside the modeller's
  control (season, weather, day of the week, …), modelled as
  `CategoricalIndex` instances from the core library.
- **Presence variables** — visitor counts whose distribution depends on
  the current context, modelled as `ConditionalDistributionIndex` instances
  from the core library.
- **Constraints** — named (usage formula, capacity) pairs; satisfaction of
  each constraint contributes to the sustainability field.  Each domain
  defines its own `Constraint` dataclass.

The core library provides `CategoricalIndex`, `ConditionalDistributionIndex`,
`CrossProductEnsemble`, and the evaluation pipeline.  The domain contributes
the `Constraint` definition and the `Model` subclass.

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
placeholder); `CrossProductEnsemble` fills it in with a concrete value for
each scenario.

## 2 — Presence variable

```python
from scipy import stats

from civic_digital_twins.dt_model import ConditionalDistributionIndex

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

PV_visitors = ConditionalDistributionIndex(
    "visitors",
    [CV_season, CV_weather],
    visitors_distribution,
)
```

`PV_visitors` is also an `Index` with `value=None`.  In grid evaluation
it is provided as an *axis* (not resolved per-scenario), so it sweeps a
dense range of visitor counts.

## 3 — Constraints

A `Constraint` is a domain-specific concept: a named pairing of a usage
formula and a capacity.  Each domain defines its own dataclass — the core
library only knows about `Index` and `Model`.  The pattern is a one-liner:

```python
from dataclasses import dataclass

from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, graph


@dataclass(eq=False)
class Constraint:
    """Named pairing of a usage formula index and a capacity index."""

    name: str
    usage: Index
    capacity: Index


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

`@dataclass(eq=False)` keeps `Constraint` instances usable as dict keys via
identity, matching the convention used by `graph.Node` and `GenericIndex`.

`graph.piecewise((expr, cond), …)` builds a conditional formula node that the
engine evaluates lazily — the condition `CV_weather == "bad"` is a graph
node that resolves to `True` or `False` once `CV_weather` is assigned a
concrete value in a scenario.

## 4 — Model

Define a `Model` subclass with `Inputs` and `Outputs` dataclasses that
declare the abstract-index contract.  Expose `.cvs`, `.pvs`, and
`.constraints` attributes so that `CrossProductEnsemble` and the
sustainability-field loop can find them.

```python
from dataclasses import dataclass

from civic_digital_twins.dt_model import CategoricalIndex, ConditionalDistributionIndex, GenericIndex, Model


class MinimalOvertourismModel(Model):
    @dataclass
    class Inputs:
        cvs: list[CategoricalIndex]
        pvs: list[ConditionalDistributionIndex]
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
`Inputs`; usage indexes in `Outputs`.  `CrossProductEnsemble` discovers
abstract indexes via `model.abstract_indexes()`; the sustainability-field loop
uses `.constraints`.  For a production model with multiple sub-models see
`MolvenoModel` in `molveno_model.py`.

## 5 — Ensemble

```python
from civic_digital_twins.dt_model import CrossProductEnsemble

scenario: dict[CategoricalIndex, list[str]] = {
    CV_season:  ["low", "high"],
    CV_weather: ["good", "unsettled", "bad"],
}

ensemble = CrossProductEnsemble(model, restrictions=scenario, max_categorical_size=10, exclude=model.pvs)
# 2 × 3 = 6 scenarios (all CV combinations enumerated)
```

`CrossProductEnsemble` implements `AxisEnsemble`: it discovers the model's
abstract indexes, enumerates all combinations of categorical CV values, and
materialises the results into a single batched ENSEMBLE axis — here
2 × 3 = 6 scenarios, one per (season, weather) pair.  Each scenario also
includes one sample of every distribution-backed non-excluded abstract index
(here: `I_C_beach`).

The `restrictions` parameter projects each categorical to a subset of its
support.  The `exclude` parameter marks PARAMETER-axis indexes (presence
variables) so they are not included in the ensemble cross-product.
`max_categorical_size` controls random sampling when a categorical's support
exceeds the size threshold; for the small finite CVs above every value is
enumerated and `max_categorical_size` is unused.

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
  - [Model / simulation layer](../../docs/design/dd-cdt-model.md) — `Model`, `Evaluation`, `EvaluationResult`, `CrossProductEnsemble`, domain modeling pattern, design rationale.
  - [Engine layer](../../docs/design/dd-cdt-engine.md) — graph nodes, topological sorting, NumPy executor.
