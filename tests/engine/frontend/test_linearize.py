"""Tests for the dt_model.engine.frontend.linearize module."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from dt_model.engine.frontend import graph, linearize


def test_simple_chain():
    """Test linearization of a simple linear chain of nodes."""
    a = graph.placeholder("a")
    b = graph.add(a, graph.constant(1.0))
    c = graph.multiply(b, graph.constant(2.0))

    plan = linearize.forest(c)

    # Check plan length
    assert len(plan) == 5

    # Check node order - dependencies should come before dependents
    assert plan.index(a) < plan.index(b)
    assert plan.index(b) < plan.index(c)

    # All nodes should be in the plan
    assert set(plan) == {a, b, c, b.right, c.right}


def test_diamond_graph():
    """Test linearization of a diamond-shaped graph."""
    x = graph.placeholder("x")
    left_branch = graph.add(x, graph.constant(1.0))
    right_branch = graph.multiply(x, graph.constant(2.0))
    output = graph.add(left_branch, right_branch)

    plan = linearize.forest(output)

    # Check plan contains all nodes
    assert len(plan) == 6

    # x should come before both branches
    assert plan.index(x) < plan.index(left_branch)
    assert plan.index(x) < plan.index(right_branch)

    # Both branches should come before output
    assert plan.index(left_branch) < plan.index(output)
    assert plan.index(right_branch) < plan.index(output)


def test_multi_output():
    """Test linearization with multiple output nodes."""
    a = graph.placeholder("a")
    b = graph.add(a, graph.constant(1.0))
    c = graph.multiply(a, graph.constant(2.0))

    plan = linearize.forest(b, c)

    # Should contain all nodes
    assert len(plan) == 5

    # Dependencies should come before dependents
    assert plan.index(a) < plan.index(b)
    assert plan.index(a) < plan.index(c)


def test_shared_subgraph():
    """Test linearization with shared subgraph components."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    # Common subgraph
    common = graph.add(a, b)

    # Two outputs that use the common subgraph
    out1 = graph.multiply(common, graph.constant(2.0))
    out2 = graph.add(common, graph.constant(3.0))

    plan = linearize.forest(out1, out2)

    # Check that common is computed only once
    assert len([n for n in plan if n is common]) == 1

    # Check dependencies
    assert plan.index(a) < plan.index(common)
    assert plan.index(b) < plan.index(common)
    assert plan.index(common) < plan.index(out1)
    assert plan.index(common) < plan.index(out2)


def test_cycle_detection():
    """Test that the linearizer detects cycles properly."""
    a = graph.placeholder("a")
    b = graph.add(a, graph.constant(1.0))

    # Create a cycle by mounting the BinaryOp nodes left side to point to itself
    # This isn't normally possible with the API but we're testing error detection
    b.left = b  # type: ignore

    with pytest.raises(ValueError, match="cycle detected"):
        linearize.forest(b)


def test_conditional_nodes():
    """Test linearization with conditional operations."""
    cond = graph.placeholder("condition")
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    # Simple where
    where_result = graph.where(cond, x, y)

    plan = linearize.forest(where_result)

    # Check dependencies
    assert plan.index(cond) < plan.index(where_result)
    assert plan.index(x) < plan.index(where_result)
    assert plan.index(y) < plan.index(where_result)

    # Multi-clause where
    clauses = [(cond, x), (graph.less(x, y), graph.constant(1.0))]
    multi_where = graph.multi_clause_where(clauses, graph.constant(0.0))

    plan = linearize.forest(multi_where)

    # Check dependencies
    assert plan.index(cond) < plan.index(multi_where)
    assert plan.index(x) < plan.index(multi_where)
    assert plan.index(y) < plan.index(multi_where)


def test_complex_graph():
    """Test a more complex computation graph."""
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    # (x + 2) * (y - 1)
    a = graph.add(x, graph.constant(2.0))
    b = graph.subtract(y, graph.constant(1.0))
    c = graph.multiply(a, b)

    # exp(y) / log(x + 1)
    d = graph.exp(y)
    e = graph.add(x, graph.constant(1.0))
    f = graph.log(e)
    g = graph.divide(d, f)

    # max(c, g)
    h = graph.maximum(c, g)

    plan = linearize.forest(h)

    # Check dependencies
    assert plan.index(x) < plan.index(a)
    assert plan.index(y) < plan.index(b)
    assert plan.index(a) < plan.index(c)
    assert plan.index(b) < plan.index(c)
    assert plan.index(y) < plan.index(d)
    assert plan.index(x) < plan.index(e)
    assert plan.index(e) < plan.index(f)
    assert plan.index(d) < plan.index(g)
    assert plan.index(f) < plan.index(g)
    assert plan.index(c) < plan.index(h)
    assert plan.index(g) < plan.index(h)


def test_axes_operations():
    """Test linearization with shape-changing operations."""
    x = graph.placeholder("x")

    # expand_dims followed by reduce_sum
    expanded = graph.expand_dims(x, axis=0)
    reduced = graph.reduce_sum(expanded, axis=1)

    plan = linearize.forest(reduced)

    # Check node ordering
    assert plan.index(x) < plan.index(expanded)
    assert plan.index(expanded) < plan.index(reduced)


def test_multiple_independent_graphs():
    """Test linearization of multiple independent computation graphs."""
    a = graph.placeholder("a")
    a_result = graph.exp(a)

    b = graph.placeholder("b")
    b_result = graph.log(b)

    plan = linearize.forest(a_result, b_result)

    # Check plan - independent graphs can be linearized in any order
    # but dependencies should still be respected
    assert len(plan) == 4
    assert plan.index(a) < plan.index(a_result)
    assert plan.index(b) < plan.index(b_result)


def test_deterministic_ordering():
    """Test that linearization produces a deterministic ordering."""
    # Create a graph with multiple equal-priority paths
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    # Multiple independent operations on x and y
    a1 = graph.exp(x)
    a2 = graph.log(x)
    b1 = graph.exp(y)
    b2 = graph.log(y)

    # Run linearization multiple times
    plan1 = linearize.forest(a1, a2, b1, b2)
    plan2 = linearize.forest(a1, a2, b1, b2)

    # Plans should be identical despite having multiple valid orderings
    assert [n.id for n in plan1] == [n.id for n in plan2]


def test_unknown_node_type():
    """Test that an appropriate error is raised for unknown node types."""

    # Create a custom node type that doesn't follow the patterns
    class CustomNode(graph.Node):
        pass

    custom_node = CustomNode()

    with pytest.raises(TypeError, match="unknown node type"):
        linearize.forest(custom_node)
