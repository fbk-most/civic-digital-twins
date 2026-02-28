"""Runnable snippets from docs/design/dd-cdt-model.md."""

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import Evaluation, Model
from civic_digital_twins.dt_model.model.index import (
    ConstIndex,
    Index,
    TimeseriesIndex,
    UniformDistIndex,
)
from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

# ---------------------------------------------------------------------------
# dd-cdt-model.md — Index Types: Index modes
# ---------------------------------------------------------------------------

# Distribution-backed (abstract)
mu = Index("mu", stats.norm(loc=0.5, scale=0.1))  # type: ignore[arg-type]

# Constant
cap = ConstIndex("capacity", 500.0)

# Formula referencing other indexes
load = Index("load", mu * cap)

# Explicit placeholder
demand = Index("demand", None)

assert mu.value is not None      # Distribution
assert cap.value == 500.0
assert demand.value is None      # placeholder


# ---------------------------------------------------------------------------
# dd-cdt-model.md — TimeseriesIndex
# ---------------------------------------------------------------------------

flow = TimeseriesIndex("flow", np.array([10.0, 20.0, 30.0]))
demand_ts = TimeseriesIndex("demand_ts")  # placeholder (no value argument)

assert flow.value is not None
assert demand_ts.value is None


# ---------------------------------------------------------------------------
# dd-cdt-model.md — Model: abstract_indexes / is_instantiated
# ---------------------------------------------------------------------------

x = UniformDistIndex("x", loc=0.0, scale=10.0)
y = UniformDistIndex("y", loc=0.0, scale=10.0)
z = Index("z", x + y)

model = Model("demo", [x, y, z])
assert len(model.abstract_indexes()) == 2     # x and y
assert model.is_instantiated() is False


# ---------------------------------------------------------------------------
# dd-cdt-model.md — DistributionEnsemble
# ---------------------------------------------------------------------------

ensemble = DistributionEnsemble(model, size=100)
scenarios = list(ensemble)

assert len(scenarios) == 100
weight, assignments = scenarios[0]
assert abs(weight - 0.01) < 1e-12      # 1/100
assert x in assignments
assert y in assignments


# ---------------------------------------------------------------------------
# dd-cdt-model.md — End-to-End Example (1-D mode)
# ---------------------------------------------------------------------------

x2 = UniformDistIndex("x2", loc=0.0, scale=10.0)
y2 = UniformDistIndex("y2", loc=0.0, scale=10.0)
z2 = Index("z2", x2 + y2)
model2 = Model("demo2", [x2, y2, z2])

ensemble2 = DistributionEnsemble(model2, size=200)
result = Evaluation(model2).evaluate(ensemble2)

# E[x2 + y2] = 5 + 5 = 10 (both U(0,10))
mean_z2 = result.marginalize(z2)
assert 7.0 < mean_z2 < 13.0, f"Expected ~10, got {mean_z2}"


if __name__ == "__main__":
    print(f"doc_model.py: all snippets OK  (E[z2] ≈ {mean_z2:.2f})")
