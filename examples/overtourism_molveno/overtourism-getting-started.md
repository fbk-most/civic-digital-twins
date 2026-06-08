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

(For reference documentation see
[`docs/design/dd-cdt-model.md`](../../docs/design/dd-cdt-model.md),
[`docs/design/dd-cdt-simulation.md`](../../docs/design/dd-cdt-simulation.md),
and [`docs/design/dd-cdt-engine.md`](../../docs/design/dd-cdt-engine.md).)

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

Use `@define` to declare a `Model` subclass.  `Inputs` declares the contract
(context variables, presence variables, parameter indexes, and capacities);
`Outputs` carries the usage formula indexes.  `compute()` builds the usage
formula from its inputs, stores `.constraints` on the instance for the
sustainability-field loop, and returns the output indexes.

```python
from civic_digital_twins.dt_model import CategoricalIndex, ConditionalDistributionIndex, DistributionIndex, GenericIndex, Index, Model, define, inputs, outputs


@define("minimal overtourism")
class MinimalOvertourismModel(Model):
    @inputs
    class Inputs:
        cv_season: CategoricalIndex
        cv_weather: CategoricalIndex
        pv_visitors: ConditionalDistributionIndex
        i_u_beach_visitors: Index
        i_c_beach: DistributionIndex

    @outputs
    class Outputs:
        usage_indexes: list[GenericIndex]

    def compute(self, inputs: Inputs) -> Outputs:
        usage = Index("beach_usage", inputs.pv_visitors * inputs.i_u_beach_visitors)
        self.constraints = [Constraint(name="beach", usage=usage, capacity=inputs.i_c_beach)]
        return MinimalOvertourismModel.Outputs(usage_indexes=[c.usage for c in self.constraints])


model = MinimalOvertourismModel(inputs=MinimalOvertourismModel.Inputs(
    cv_season=CV_season,
    cv_weather=CV_weather,
    pv_visitors=PV_visitors,
    i_u_beach_visitors=I_U_beach_visitors,
    i_c_beach=I_C_beach,
))
```

All abstract indexes (context and presence variables, capacities) are declared
in `Inputs`; usage indexes in `Outputs`.  `compute()` stores `.constraints` on
the instance as a plain attribute — the sustainability-field loop uses it
directly.  For a production model with multiple concern sub-models using the
same `@define` pattern see `MolvenoModel` in `molveno_model.py`.

## 5 — Ensemble

```python
from civic_digital_twins.dt_model import CrossProductEnsemble, DomainValue, GenericIndex, Scenario

scenario_overrides: dict[GenericIndex, DomainValue] = {
    model.inputs.cv_season:  ["low", "high"],
    model.inputs.cv_weather: ["good", "unsettled", "bad"],
}

scenario_obj = Scenario(
    model,
    overrides=scenario_overrides,
    parameter_axes=[model.inputs.pv_visitors],
)
ensemble = CrossProductEnsemble(scenario_obj, max_categorical_size=10)
# 2 × 3 = 6 scenarios (all CV combinations enumerated)
```

`CrossProductEnsemble` implements `AxisEnsemble`: it discovers the model's
abstract indexes, enumerates all combinations of categorical CV values, and
materialises the results into a single batched ENSEMBLE axis — here
2 × 3 = 6 scenarios, one per (season, weather) pair.  Each scenario also
includes one sample of every distribution-backed non-parameter abstract index
(here: `I_C_beach`).

`Scenario(model, overrides={…: [...]})` restricts each categorical to a
subset of its support and renormalises the probabilities.  Presence-variable
indexes declared as ``parameter_axes`` on the :class:`~dt_model.Scenario`
are automatically excluded from the ensemble cross-product and swept over
the grid in step 6 instead.
`max_categorical_size` controls random sampling when a categorical's support
exceeds the size threshold; for the small finite CVs above every value is
enumerated and `max_categorical_size` is unused.

## 6 — Grid evaluation

Presence variables are not resolved per-scenario; instead they define the
grid axes over which the sustainability field is computed:

```python
import numpy as np
from civic_digital_twins.dt_model import Evaluation, Scenario

visitors_axis = np.linspace(0, 20_000, 201)

result = Evaluation(scenario_obj).evaluate(
    ensemble=ensemble,
    parameters={model.inputs.pv_visitors: visitors_axis},
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
    usage = np.broadcast_to(result[c.usage], result.full_shape)  # (201, 6)

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
  - [Simulation guide](../../docs/design/dd-cdt-simulation.md) — `Scenario`, `CrossProductEnsemble`, `EvaluationHandle`, incremental evaluation.
  - [Engine layer](../../docs/design/dd-cdt-engine.md) — graph nodes, topological sorting, NumPy executor.
