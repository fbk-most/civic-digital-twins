"""Tests for the dt_model.engine.frontend.replace module."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from dt_model.engine.frontend import graph, linearize, replace
from dt_model.engine.numpybackend import executor


def test_basic_node_replacements():
    """Test basic node replacement functionality for different node types."""
    # Direct replacement tests
    a = graph.placeholder("a")
    b = graph.constant(5.0)
    c = graph.placeholder("c")

    # Direct replacement
    assert replace.nodes(a, {a: c}) is c
    assert replace.nodes(b, {b: c}) is c

    # Binary operator replacement
    add_node = graph.add(a, b)
    d = graph.placeholder("d")

    # Replace one operand
    result = replace.nodes(add_node, {a: d})
    assert isinstance(result, graph.add)
    assert result.left is d
    assert result.right is b
    assert result is not add_node

    # Replace both operands
    e = graph.placeholder("e")
    result = replace.nodes(add_node, {a: d, b: e})
    assert isinstance(result, graph.add)
    assert result.left is d
    assert result.right is e

    # Unary operator replacement
    exp_node = graph.exp(a)
    result = replace.nodes(exp_node, {a: c})
    assert isinstance(result, graph.exp)
    assert result.node is c

    # Axis operator replacement
    sum_node = graph.reduce_sum(a, axis=0)
    result = replace.nodes(sum_node, {a: c})
    assert isinstance(result, graph.reduce_sum)
    assert result.node is c
    assert result.axis == 0


def test_complex_node_replacements():
    """Test replacement in more complex node structures."""
    # Where operation
    cond = graph.placeholder("cond")
    then_branch = graph.placeholder("then")
    else_branch = graph.placeholder("else")
    where_op = graph.where(cond, then_branch, else_branch)

    new_cond = graph.placeholder("new_cond")
    new_then = graph.placeholder("new_then")

    result = replace.nodes(where_op, {cond: new_cond, then_branch: new_then})
    assert isinstance(result, graph.where)
    assert result.condition is new_cond
    assert result.then is new_then
    assert result.otherwise is else_branch

    # Multi-clause where operation
    cond1 = graph.placeholder("cond1")
    val1 = graph.placeholder("val1")
    cond2 = graph.placeholder("cond2")
    val2 = graph.placeholder("val2")
    default = graph.placeholder("default")

    mcw = graph.multi_clause_where([(cond1, val1), (cond2, val2)], default)

    new_cond1 = graph.placeholder("new_cond1")
    new_val2 = graph.placeholder("new_val2")
    new_default = graph.placeholder("new_default")

    result = replace.nodes(mcw, {cond1: new_cond1, val2: new_val2, default: new_default})
    assert isinstance(result, graph.multi_clause_where)
    assert result.clauses[0][0] is new_cond1
    assert result.clauses[0][1] is val1
    assert result.clauses[1][0] is cond2
    assert result.clauses[1][1] is new_val2
    assert result.default_value is new_default

    # Default-only replacement in multi-clause where
    only_default_mcw = graph.multi_clause_where([(cond1, val1)], default)
    only_default_result = replace.nodes(only_default_mcw, {default: new_default})
    assert isinstance(only_default_result, graph.multi_clause_where)
    assert only_default_result.clauses[0][0] is cond1
    assert only_default_result.clauses[0][1] is val1
    assert only_default_result.default_value is new_default


def test_recursive_graph_replacement():
    """Test replacement in deeply nested graph structures."""
    # Create a complex graph
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    c = graph.multiply(a, graph.constant(2.0))
    d = graph.add(c, b)
    e = graph.exp(d)

    # Create replacement nodes
    a_new = graph.placeholder("a_new")
    b_new = graph.placeholder("b_new")

    # Replace nodes deep in the graph
    result = replace.nodes(e, {a: a_new, b: b_new})

    # Verify structure is properly transformed
    assert isinstance(result, graph.exp)
    assert isinstance(result.node, graph.add)

    add_node = result.node
    assert isinstance(add_node.left, graph.multiply)
    assert add_node.right is b_new

    mul_node = add_node.left
    assert mul_node.left is a_new
    assert isinstance(mul_node.right, graph.constant)
    assert mul_node.right.value == 2.0

    # Verify original graph is unchanged
    assert e.node is d
    assert d.left is c
    assert d.right is b
    assert c.left is a


def test_attribute_preservation():
    """Test that node attributes are preserved during replacement."""
    a = graph.placeholder("a")
    b = graph.constant(3.0)
    c = graph.add(a, b)
    c.name = "addition"
    c.flags = graph.NODE_FLAG_TRACE

    new_a = graph.placeholder("new_a")
    result = replace.nodes(c, {a: new_a})

    assert isinstance(result, graph.add)
    assert result.left is new_a
    assert result.right is b
    assert result.name == "addition"
    assert result.flags == graph.NODE_FLAG_TRACE


def test_identity_cases():
    """Test cases where no replacements are made (identity cases)."""
    # Test with direct nodes
    a = graph.placeholder("a")
    b = graph.constant(5.0)
    c = graph.add(a, b)

    # Empty replacements
    assert replace.nodes(a, {}) is a
    assert replace.nodes(b, {}) is b
    assert replace.nodes(c, {}) is c

    # Non-matching replacements
    d = graph.placeholder("d")
    e = graph.placeholder("e")
    assert replace.nodes(c, {d: e}) is c

    # Test with various node types
    exp_a = graph.exp(a)
    assert replace.nodes(exp_a, {}) is exp_a
    assert replace.nodes(exp_a, {d: e}) is exp_a

    # Where operation identity case
    where_op = graph.where(a, b, c)
    assert replace.nodes(where_op, {}) is where_op

    # Multi-clause where identity case
    mcw = graph.multi_clause_where([(a, b), (c, graph.constant(5.0))], graph.constant(0.0))
    assert replace.nodes(mcw, {}) is mcw

    # Axis operator identity case
    sum_a = graph.reduce_sum(a, axis=0)
    assert replace.nodes(sum_a, {d: e}) is sum_a


def test_replace_nodes_functional_evaluation():
    """Test that replacements produce functionally equivalent graphs."""
    # Create original graph: (a * 2) + (b * 3)
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    a_times_2 = graph.multiply(a, graph.constant(2.0))
    b_times_3 = graph.multiply(b, graph.constant(3.0))
    original = graph.add(a_times_2, b_times_3)

    # Create replacement graph: (x * 4) + (y * 3)
    # This should double the effect of 'a' without changing 'b'
    x = graph.placeholder("x")
    x_times_4 = graph.multiply(x, graph.constant(4.0))

    # Create replacements: a -> x, a_times_2 -> x_times_4
    replacements = {a: x, a_times_2: x_times_4}
    result = replace.nodes(original, replacements)

    # Create execution plans
    original_plan = linearize.forest(original)
    result_plan = linearize.forest(result)

    # Evaluate with test values
    a_val = np.array([1.0, 2.0, 3.0])
    b_val = np.array([4.0, 5.0, 6.0])
    x_val = a_val  # Use same values for testing

    # Evaluate original
    orig_state = executor.State({a: a_val, b: b_val})
    for node in original_plan:
        executor.evaluate(orig_state, node)

    # Evaluate replacement
    result_state = executor.State({x: x_val, b: b_val})
    for node in result_plan:
        executor.evaluate(result_state, node)

    # Verify the transformed graph produces expected results
    expected_result = (a_val * 4.0) + (b_val * 3.0)
    assert np.array_equal(result_state.values[result], expected_result)


def test_replace_nodes_unsupported_type():
    """Test that an error is raised for unsupported node types."""

    # Create a custom node type not handled by the replace function
    class CustomNode(graph.Node):
        pass

    custom_node = CustomNode()

    with pytest.raises(TypeError, match="Unsupported node type"):
        replace.nodes(custom_node, {graph.placeholder("a"): graph.placeholder("b")})
