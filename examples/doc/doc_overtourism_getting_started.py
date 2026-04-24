"""Runnable snippets from examples/overtourism_molveno/overtourism-getting-started.md."""
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Ensure examples/ is on sys.path so overtourism_molveno can be imported
# when running this script directly (e.g. `uv run python examples/doc/doc_overtourism_getting_started.py`).
_examples_dir = Path(__file__).parent.parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from dataclasses import dataclass

import numpy as np
from overtourism_molveno.overtourism_metamodel import (
    Constraint,
    OvertourismEnsemble,
    PresenceVariable,
)
from scipy import stats

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    Distribution,
    DistributionIndex,
    Evaluation,
    GenericIndex,
    Index,
    Model,
    graph,
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

assert CV_season.value is None  # placeholder
assert CV_weather.value is None


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


PV_visitors = PresenceVariable(
    "visitors",
    [CV_season, CV_weather],
    visitors_distribution,
)

assert PV_visitors.value is None  # placeholder (axis in grid evaluation)


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §3 — Constraints
# ---------------------------------------------------------------------------

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

assert len(model.constraints) == 1
assert model.constraints[0] is C_beach


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §5 — Ensemble
# ---------------------------------------------------------------------------

scenario: dict[CategoricalIndex, list[str]] = {
    CV_season: ["low", "high"],
    CV_weather: ["good", "unsettled", "bad"],
}

ensemble = OvertourismEnsemble(model, scenario, cv_ensemble_size=10)
# 2 × 3 = 6 scenarios (cv_ensemble_size=10 >= support sizes 2 and 3,
# so all CV values are enumerated rather than sampled randomly)
assert len(ensemble) == 6

assert abs(ensemble.ensemble_weights[0].sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §6 — Grid evaluation
# ---------------------------------------------------------------------------

visitors_axis = np.linspace(0, 20_000, 201)

result = Evaluation(model).evaluate(
    ensemble=ensemble,
    parameters={PV_visitors: visitors_axis},
)

assert result.full_shape == (201, 6)


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §7 — Sustainability field
# ---------------------------------------------------------------------------

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
assert field.shape == (201,)
assert np.all(field >= 0.0) and np.all(field <= 1.0)

# Field should be monotonically non-increasing (more visitors → less sustainable)
assert field[0] >= field[-1], "Sustainability should decrease with more visitors"

# At 0 visitors the field should be 1 (or very close)
assert field[0] > 0.9


if __name__ == "__main__":
    print(f"doc_overtourism_getting_started.py: all snippets OK  (field[0]={field[0]:.3f}, field[-1]={field[-1]:.3f})")
