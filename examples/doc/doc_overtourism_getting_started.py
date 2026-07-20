"""Runnable snippets from examples/overtourism_molveno/overtourism-getting-started.md."""
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Ensure examples/ is on sys.path so overtourism_molveno can be imported
# when running this script directly (e.g. `uv run python examples/doc/doc_overtourism_getting_started.py`).
_examples_dir = Path(__file__).parent.parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from civic_digital_twins.dt_model import (  # noqa: E402
    CategoricalIndex,
    ConditionalDistributionIndex,
    CrossProductEnsemble,
    DistributionIndex,
    DomainValue,
    Evaluation,
    GenericIndex,
    Index,
    Model,
    Scenario,
    define,
    graph,
    inputs,
    outputs,
)

# ---------------------------------------------------------------------------
# overtourism-getting-started.md §1 — Context variables
# ---------------------------------------------------------------------------

CV_season = CategoricalIndex(
    "season",
    {"low": 0.6, "high": 0.4},
)

CV_weather = CategoricalIndex(
    "weather",
    {"good": 1 / 3, "unsettled": 1 / 3, "bad": 1 / 3},
)

assert CV_season.is_abstract  # placeholder
assert CV_weather.is_abstract


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §2 — Presence variable
# ---------------------------------------------------------------------------


def visitors_distribution(season, weather):
    """Return a uniform distribution for visitor presence."""
    presence_stats = {
        ("low", "good"): (1_500, 2_500),
        ("low", "unsettled"): (1_100, 1_900),
        ("low", "bad"): (1_000, 1_300),
        ("high", "good"): (6_000, 10_000),
        ("high", "unsettled"): (4_500, 7_500),
        ("high", "bad"): (3_000, 5_000),
    }
    low, high = presence_stats[(season, weather)]
    return stats.uniform(loc=low, scale=high - low)


PV_visitors = ConditionalDistributionIndex(
    "visitors",
    [CV_season, CV_weather],
    visitors_distribution,
)

assert PV_visitors.is_abstract  # placeholder (axis in grid evaluation)


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §3 — Constraints
# ---------------------------------------------------------------------------


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

assert C_beach.name == "beach"


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §4 — Model
# ---------------------------------------------------------------------------


@define("minimal overtourism")
class MinimalOvertourismModel(Model):  # noqa: D101
    @inputs
    class Inputs:  # noqa: D106
        cv_season: CategoricalIndex
        cv_weather: CategoricalIndex
        pv_visitors: ConditionalDistributionIndex
        i_u_beach_visitors: Index
        i_c_beach: DistributionIndex

    @outputs
    class Outputs:  # noqa: D106
        usage_indexes: list[GenericIndex]

    def compute(self, inputs: Inputs) -> Outputs:  # noqa: D102
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

assert len(model.constraints) == 1
assert model.constraints[0].name == "beach"


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §5 — Ensemble
# ---------------------------------------------------------------------------

scenario_overrides: dict[GenericIndex, DomainValue] = {
    model.inputs.cv_season: ["low", "high"],
    model.inputs.cv_weather: ["good", "unsettled", "bad"],
}

scenario_obj = Scenario(
    model,
    overrides=scenario_overrides,
    parameter_axes=[model.inputs.pv_visitors],
)
ensemble = CrossProductEnsemble(scenario_obj, max_categorical_size=10)
# 2 × 3 = 6 scenarios (max_categorical_size=10 >= support sizes 2 and 3,
# so all CV values are enumerated rather than sampled randomly)
assert len(ensemble) == 6

assert abs(ensemble.ensemble_weights[0].sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §6 — Grid evaluation
# ---------------------------------------------------------------------------

visitors_axis = np.linspace(0, 20_000, 201)

result = Evaluation(scenario_obj).evaluate(
    ensemble=ensemble,
    parameters={model.inputs.pv_visitors: visitors_axis},
)

assert result.full_shape == (201, 6)


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §7 — Sustainability field
# ---------------------------------------------------------------------------

field = np.ones(visitors_axis.size)

for c in model.constraints:
    usage = np.broadcast_to(result[c.usage], result.full_shape)  # (201, 6)

    if isinstance(c.capacity, DistributionIndex):
        # Probabilistic capacity: probability that usage ≤ capacity
        mask = 1.0 - c.capacity.frozen_distribution.cdf(usage)
    else:
        cap = np.broadcast_to(result[c.capacity], result.full_shape)
        mask = (usage <= cap).astype(float)

    # Marginalise over scenarios → shape (201,)
    field *= np.tensordot(mask, result.weights, axes=([-1], [0]))

# field[i] ∈ [0, 1]: sustainability score for visitors_axis[i] visitors
assert field.shape == (201,)
assert np.all(field >= 0.0) and np.all(field <= 1.0)

# Field should be monotonically non-increasing (more visitors → less sustainable)
assert field[0] >= field[-1], "Sustainability should decrease with more visitors"

# At 0 visitors the field should be 1 (or very close)
assert field[0] > 0.9


if __name__ == "__main__":
    print(f"doc_overtourism_getting_started.py: all snippets OK  (field[0]={field[0]:.3f}, field[-1]={field[-1]:.3f})")
