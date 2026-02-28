"""Runnable snippets from examples/overtourism_molveno/overtourism-getting-started.md."""

import sys
from pathlib import Path

# Ensure examples/ is on sys.path so overtourism_molveno can be imported
# when running this script directly (e.g. `uv run python examples/doc/doc_overtourism_getting_started.py`).
_examples_dir = Path(__file__).parent.parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

import numpy as np

from civic_digital_twins.dt_model import Evaluation, piecewise
from civic_digital_twins.dt_model.model.index import Distribution, Index, TriangDistIndex
from overtourism_molveno.overtourism_metamodel import (
    CategoricalContextVariable,
    Constraint,
    ContextVariable,
    OvertourismEnsemble,
    OvertourismModel,
    PresenceVariable,
    UniformCategoricalContextVariable,
)

# ---------------------------------------------------------------------------
# overtourism-getting-started.md §1 — Context variables
# ---------------------------------------------------------------------------

CV_season = CategoricalContextVariable(
    "season",
    {"low": 0.6, "high": 0.4},
)

CV_weather = UniformCategoricalContextVariable(
    "weather",
    ["good", "unsettled", "bad"],
)

assert CV_season.value is None    # placeholder
assert CV_weather.value is None


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §2 — Presence variable
# ---------------------------------------------------------------------------

presence_stats = {
    ("low", "good"): (2_000, 500),
    ("low", "unsettled"): (1_500, 400),
    ("low", "bad"): (1_000, 300),
    ("high", "good"): (8_000, 2_000),
    ("high", "unsettled"): (6_000, 1_500),
    ("high", "bad"): (4_000, 1_000),
}

PV_visitors = PresenceVariable(
    "visitors",
    [CV_season, CV_weather],
    presence_stats,
)

assert PV_visitors.value is None  # placeholder (axis in grid evaluation)


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §3 — Constraints
# ---------------------------------------------------------------------------

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

assert C_beach.name == "beach"


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §4 — OvertourismModel
# ---------------------------------------------------------------------------

model = OvertourismModel(
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

scenario: dict[ContextVariable, list] = {
    CV_season: ["low", "high"],
    CV_weather: ["good", "unsettled", "bad"],
}

ensemble = OvertourismEnsemble(model, scenario, cv_ensemble_size=10)
scenarios = list(ensemble)
# 2 × 3 = 6 scenarios (cv_ensemble_size=10 >= support sizes 2 and 3,
# so all CV values are enumerated rather than sampled randomly)
assert len(scenarios) == 6

weights = np.array([w for w, _ in scenarios])
assert abs(weights.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# overtourism-getting-started.md §6 — Grid evaluation
# ---------------------------------------------------------------------------

visitors_axis = np.linspace(0, 20_000, 201)

result = Evaluation(model).evaluate(
    scenarios,
    axes={PV_visitors: visitors_axis},
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
    print(
        f"doc_overtourism_getting_started.py: all snippets OK  "
        f"(field[0]={field[0]:.3f}, field[-1]={field[-1]:.3f})"
    )
