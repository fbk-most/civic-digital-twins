"""Runnable snippets from docs/design/dd-cdt-engine.md and README.md."""

import numpy as np

from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor

# ---------------------------------------------------------------------------
# README — engine layer snippet
# ---------------------------------------------------------------------------

a = graph.placeholder("a")
b = graph.placeholder("b")
c = a * 2 + b

state = executor.State(values={a: np.asarray(3.0), b: np.asarray(1.0)})
executor.evaluate_nodes(state, *linearize.forest(c))
assert float(state.get_node_value(c)) == 7.0


# ---------------------------------------------------------------------------
# dd-cdt-engine.md — End-to-End Example
# ---------------------------------------------------------------------------


class TimeDimension:
    """Represents nodes in the time dimension."""


class EnsembleDimension:
    """Represents nodes in the ensemble dimension."""


a2 = graph.placeholder[TimeDimension]("a")
b2 = graph.placeholder[TimeDimension]("b")
k0 = graph.constant(3, name="k0")
c2 = a2 + b2 * k0
c1 = graph.function_call("reduce", c2)
d2 = a2 * k0 - b2
d1 = graph.function_call("reduce", d2)

nodes = linearize.forest(c1, d1)

state2 = executor.State(
    values={
        a2: np.asarray([100, 10, 1]),
        b2: np.asarray([200, 20, 2]),
    },
    functions={
        "reduce": executor.LambdaAdapter(
            lambda n: np.divide(n, np.asarray(5)),
        ),
    },
)

executor.evaluate_nodes(state2, *nodes)

# c = a + b*k0 = [100,10,1] + [600,60,6] = [700,70,7]; c1 = c/5
np.testing.assert_allclose(state2.get_node_value(c1), [140.0, 14.0, 1.4])
# d = a*k0 - b = [300,30,3] - [200,20,2] = [100,10,1]; d1 = d/5
np.testing.assert_allclose(state2.get_node_value(d1), [20.0, 2.0, 0.2])


# ---------------------------------------------------------------------------
# dd-cdt-engine.md — frontend/graph.py: Writing a DAG
# ---------------------------------------------------------------------------

a3 = graph.placeholder("a3")
b3 = graph.placeholder("b3")
scale = graph.constant(1024)
c3 = graph.exp(a3) + 55 / a3
d3 = c3 * b3 + scale
e3 = graph.power(a3, c3) * 144

# Verify the DAG was constructed (no evaluation needed for this snippet)
assert a3 is not None
assert d3 is not None
assert e3 is not None


# ---------------------------------------------------------------------------
# dd-cdt-engine.md — frontend/graph.py: function_call
# ---------------------------------------------------------------------------

f = graph.function_call("reduce", c3, d3, e3)
assert f is not None


# ---------------------------------------------------------------------------
# dd-cdt-engine.md — Timeseries Nodes
# ---------------------------------------------------------------------------

demand = graph.timeseries_constant(np.arange(24, dtype=float), "demand")
traffic = graph.timeseries_placeholder("traffic")
scaled = demand * graph.constant(0.5)

# Evaluate the timeseries formula
ts_state = executor.State(values={})
executor.evaluate_nodes(ts_state, *linearize.forest(scaled))
np.testing.assert_allclose(
    ts_state.get_node_value(scaled),
    np.arange(24, dtype=float) * 0.5,
)


# ---------------------------------------------------------------------------
# dd-cdt-engine.md — frontend/linearize.py: Topological Sorting
# ---------------------------------------------------------------------------

# Using c3 from the DAG writing section above
plan = linearize.forest(c3)
assert len(plan) > 0

for node in plan:
    assert str(node)  # each node has a string representation


if __name__ == "__main__":
    print("doc_engine.py: all snippets OK")
