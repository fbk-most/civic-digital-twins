""" Runnable snippets from docs/design/dd-cdt-engine.md."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np

from civic_digital_twins.dt_model.engine import compileflags
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor


# ---------------------------------------------------------------------------
# Block 00 — End-to-End Example
# ---------------------------------------------------------------------------


class TimeDimension:
    """Represents nodes in the time dimension."""


class EnsembleDimension:
    """Represents nodes in the ensemble dimension."""


def _demo_end_to_end() -> None:
    """dd-cdt-engine.md — End-to-End Example (block 00)."""
    a = graph.placeholder[TimeDimension]("a")
    b = graph.placeholder[TimeDimension]("b")
    k0 = graph.constant(3, name="k0")
    c = a + b * k0
    c1 = graph.function_call("reduce", c)
    d = a * k0 - b
    d1 = graph.function_call("reduce", d)

    nodes = linearize.forest(c1, d1)

    state = executor.State(
        values={
            a: np.asarray([100, 10, 1]),
            b: np.asarray([200, 20, 2]),
        },
        functions={
            "reduce": executor.LambdaAdapter(
                lambda n: np.divide(n, np.asarray(5)),
            ),
        },
    )

    executor.evaluate_nodes(state, *nodes)

    # c = a + b*k0 = [100,10,1] + [600,60,6] = [700,70,7]; c1 = c/5
    np.testing.assert_allclose(state.get_node_value(c1), [140.0, 14.0, 1.4])
    # d = a*k0 - b = [300,30,3] - [200,20,2] = [100,10,1]; d1 = d/5
    np.testing.assert_allclose(state.get_node_value(d1), [20.0, 2.0, 0.2])


# ---------------------------------------------------------------------------
# Blocks 01, 03, 05 — frontend/graph.py: Writing a DAG
# ---------------------------------------------------------------------------


def _demo_dag_writing() -> None:
    """dd-cdt-engine.md — Writing a DAG (blocks 01, 03, 05)."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    scale = graph.constant(1024)
    c = graph.exp(a) + 55 / a
    d = c * b + scale
    e = graph.power(a, c) * 144

    assert a is not None
    assert d is not None
    assert e is not None

    f = graph.function_call("reduce", c, d, e)
    assert f is not None


# ---------------------------------------------------------------------------
# Block 07 — Timeseries Nodes
# ---------------------------------------------------------------------------


def _demo_timeseries() -> None:
    """dd-cdt-engine.md — Timeseries Nodes (block 07)."""
    demand = graph.timeseries_constant(np.arange(24, dtype=float), "demand")
    traffic = graph.timeseries_placeholder("traffic")
    scaled = demand * graph.constant(0.5)

    assert traffic is not None

    ts_state = executor.State(values={})
    executor.evaluate_nodes(ts_state, *linearize.forest(scaled))
    np.testing.assert_allclose(
        ts_state.get_node_value(scaled),
        np.arange(24, dtype=float) * 0.5,
    )


# ---------------------------------------------------------------------------
# Blocks 08–13 — Printing DAG Nodes and Topological Sorting
# ---------------------------------------------------------------------------


def _demo_ssa_and_sorting() -> None:
    """dd-cdt-engine.md — Printing DAG Nodes + Topological Sorting (blocks 08–13).

    The ``n1``, ``n3``, ... variable names are Python names chosen to mirror the
    SSA-form identifiers that appear in ``str(node)`` output.  The actual runtime
    IDs will differ (the global counter has already advanced), but the *source
    lines* are what the doc-sync test checks.
    """
    # --- Blocks 08, 09, 10: individual node SSA form ---
    # These are the lines that print(a), print(scale), and print(c) would produce
    # in a fresh context where `a` is node #1 and `scale` is node #3.
    n1 = graph.placeholder(name='a', default_value=None)
    n3 = graph.constant(value=1024, name='')
    n4 = graph.exp(node=n1, name='')
    n5 = graph.constant(value=55, name='')
    n6 = graph.divide(left=n5, right=n1, name='')
    n7 = graph.add(left=n4, right=n6, name='c')   # block 10: print(c) — named root

    assert n3 is not None

    # --- Block 12: topological sort of c = graph.exp(a) + 55/a ---
    # Same structural nodes; the root add-node has an empty name in this context.
    n7 = graph.add(left=n4, right=n6, name='')    # noqa: F841  block 12 line 5

    # --- Blocks 11, 13: shorthand notation and linearize.forest ---
    a = graph.placeholder("a")
    c = graph.exp(a) + 55 / a

    plan = linearize.forest(c)
    for node in plan:
        assert str(node)  # each node has a string representation


# ---------------------------------------------------------------------------
# Block 15 — numpybackend/executor.py: actual usage
# ---------------------------------------------------------------------------


def _demo_executor_usage() -> None:
    """dd-cdt-engine.md — numpybackend/executor.py usage (block 15)."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    scale = graph.constant(1024)
    c = graph.exp(a) + 55 / a
    d = c * b + scale
    e = graph.power(a, c) * 144

    state = executor.State(
        values={
            a: np.asarray(4),
            b: np.asarray(2),
        }
    )

    nodes = linearize.forest(c, d, e)
    executor.evaluate_nodes(state, *nodes)

    assert state.get_node_value(c) is not None


# ---------------------------------------------------------------------------
# Block 16 — Tracing execution: numpy execution trace output
# ---------------------------------------------------------------------------


def _demo_numpy_trace() -> None:
    """dd-cdt-engine.md — DTMODEL_ENGINE_FLAGS=trace output (block 16).

    These are the numpy operations emitted when the executor runs with TRACE
    enabled for the formula ``c = graph.exp(a) + 55/a``, ``d = c*b + scale``,
    ``e = graph.power(a,c) * 144`` with ``a=4``, ``b=2``, ``scale=1024``.
    """
    n1 = np.asarray(4)
    n2 = np.asarray(2)
    n3 = np.asarray(1024)
    n5 = np.asarray(55)
    n11 = np.asarray(144)
    n4 = np.exp(n1)
    n6 = np.divide(n5, n1)
    n7 = np.add(n4, n6)
    n10 = np.power(n1, n7)
    n12 = np.multiply(n10, n11)
    n8 = np.multiply(n7, n2)
    n9 = np.add(n8, n3)

    # spot-check a few values
    np.testing.assert_allclose(n4, np.exp(4))
    np.testing.assert_allclose(n6, 55 / 4)
    np.testing.assert_allclose(n7, np.exp(4) + 55 / 4)
    assert float(n9) > 0
    assert n12 is not None


# ---------------------------------------------------------------------------
# Block 17 — More DTMODEL_ENGINE_FLAGS: compileflags API
# ---------------------------------------------------------------------------


def _demo_compileflags() -> None:
    """dd-cdt-engine.md — programmatic compileflags (block 17).

    The state is constructed with TRACE + BREAK flags to demonstrate the API.
    ``evaluate_nodes`` is intentionally not called here: BREAK raises after the
    first node, and TRACE would flood stdout.
    """
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    scale = graph.constant(1024)
    c = graph.exp(a) + 55 / a
    d = c * b + scale
    e = graph.power(a, c) * 144

    flags: int = 0
    flags |= compileflags.TRACE  # print each node and its value
    flags |= compileflags.BREAK  # stop after evaluating a node

    state = executor.State(
        values={
            a: np.asarray(4),
            b: np.asarray(2),
        },
        flags=flags,
    )

    # Verify construction succeeded without evaluating (BREAK/TRACE side-effects).
    assert state is not None
    assert e is not None
    assert d is not None


# ---------------------------------------------------------------------------
# Block 18 — User-Defined Functions
# ---------------------------------------------------------------------------


def _demo_udf() -> None:
    """dd-cdt-engine.md — User-Defined Functions (block 18)."""
    k0 = graph.constant(100)
    k1 = graph.constant(50)
    k2 = graph.constant(117)
    k3 = graph.constant(1000)

    scaled1 = graph.function_call("scale1", k0, k1)
    scaled2 = graph.function_call("scale2", k0=k0, k1=k1)
    scaled3 = graph.function_call("scale3", k0, k1, k2=k2, k3=k3, scaled2=scaled2)

    nodes = linearize.forest(scaled1, scaled2, scaled3)

    state = executor.State(
        values={},
        functions={
            "scale1": executor.LambdaAdapter(
                lambda k0, k1: np.add(k0, k1),
            ),
            "scale2": executor.LambdaAdapter(
                lambda **kwargs: np.add(kwargs["k0"], kwargs["k1"]),
            ),
            "scale3": executor.LambdaAdapter(
                lambda k0, k1, **kwargs: np.add(
                    k0, np.add(k1, np.add(kwargs["k2"], np.add(kwargs["k3"], kwargs["scaled2"])))
                ),
            ),
        },
    )

    executor.evaluate_nodes(state, *nodes)

    assert int(state.get_node_value(scaled1)) == 150
    assert int(state.get_node_value(scaled2)) == 150
    assert int(state.get_node_value(scaled3)) == 1417


# ---------------------------------------------------------------------------
# Block 19 — UDF execution trace output
# ---------------------------------------------------------------------------


def _demo_udf_trace() -> None:
    """dd-cdt-engine.md — UDF DTMODEL_ENGINE_FLAGS=trace output (block 19).

    These are the numpy calls emitted when the executor traces scale1/2/3
    with k0=100, k1=50, k2=117, k3=1000.
    """
    scale1 = lambda k0, k1: np.add(k0, k1)  # noqa: E731
    scale2 = lambda **kwargs: np.add(kwargs["k0"], kwargs["k1"])  # noqa: E731
    scale3 = lambda k0, k1, **kwargs: np.add(  # noqa: E731
        k0, np.add(k1, np.add(kwargs["k2"], np.add(kwargs["k3"], kwargs["scaled2"])))
    )

    n1 = np.asarray(100)
    n2 = np.asarray(50)
    n3 = np.asarray(117)
    n4 = np.asarray(1000)
    n5 = scale1(n1, n2)
    n6 = scale2(k0=n1, k1=n2)
    n7 = scale3(n1, n2, k2=n3, k3=n4, scaled2=n6)

    assert int(n5) == 150
    assert int(n6) == 150
    assert int(n7) == 1417


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------

_demo_end_to_end()
_demo_dag_writing()
_demo_timeseries()
_demo_ssa_and_sorting()
_demo_executor_usage()
_demo_numpy_trace()
_demo_compileflags()
_demo_udf()
_demo_udf_trace()

if __name__ == "__main__":
    print("doc_engine.py: all snippets OK")
