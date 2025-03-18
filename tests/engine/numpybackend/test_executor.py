"""Tests for the dt_model.engine.numpybackend.executor module."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from dt_model.engine.frontend import graph, linearize
from dt_model.engine.numpybackend import executor


def test_constant_evaluation():
    """Test evaluation of constant nodes with the executor."""
    # Create and linearize constant nodes
    node1 = graph.constant(1.0)
    node2 = graph.constant(True)
    node3 = graph.constant(42)

    # Create execution plans
    plan1 = linearize.forest(node1)
    plan2 = linearize.forest(node2)
    plan3 = linearize.forest(node3)

    # Create state for executor
    state1 = executor.State({})
    state2 = executor.State({})
    state3 = executor.State({})

    # Execute each node in the plan
    for node in plan1:
        executor.evaluate(state1, node)
    for node in plan2:
        executor.evaluate(state2, node)
    for node in plan3:
        executor.evaluate(state3, node)

    # Check results
    assert np.array_equal(state1.values[node1], np.array(1.0))
    assert np.array_equal(state2.values[node2], np.array(True))
    assert np.array_equal(state3.values[node3], np.array(42))


def test_placeholder_evaluation():
    """Test evaluation of placeholder nodes with the executor."""
    # Create placeholder nodes
    x = graph.placeholder("x")
    y = graph.placeholder("y", default_value=3.14)

    # Create execution plans
    plan_x = linearize.forest(x)
    plan_y = linearize.forest(y)

    # Test with binding
    x_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    state_x = executor.State({x: x_value})
    for node in plan_x:
        executor.evaluate(state_x, node)
    assert np.array_equal(state_x.values[x], x_value)

    # Test default value
    state_y = executor.State({})
    for node in plan_y:
        executor.evaluate(state_y, node)
    assert np.array_equal(state_y.values[y], np.array(3.14))

    # Test missing binding
    state_missing = executor.State({})
    with pytest.raises(executor.PlaceholderValueNotProvided):
        for node in plan_x:
            executor.evaluate(state_missing, node)


def test_arithmetic_operations():
    """Test evaluation of arithmetic operations with the executor."""
    # Create placeholder nodes
    x_node = graph.placeholder("x")
    y_node = graph.placeholder("y")

    # Create operation nodes
    add_node = graph.add(x_node, y_node)
    sub_node = graph.subtract(x_node, y_node)
    mul_node = graph.multiply(x_node, y_node)
    div_node = graph.divide(x_node, y_node)

    # Create execution plans
    add_plan = linearize.forest(add_node)
    sub_plan = linearize.forest(sub_node)
    mul_plan = linearize.forest(mul_node)
    div_plan = linearize.forest(div_node)

    # Prepare test data
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    # Test addition
    add_state = executor.State({x_node: x, y_node: y})
    for node in add_plan:
        executor.evaluate(add_state, node)
    assert np.array_equal(add_state.values[add_node], x + y)

    # Test subtraction
    sub_state = executor.State({x_node: x, y_node: y})
    for node in sub_plan:
        executor.evaluate(sub_state, node)
    assert np.array_equal(sub_state.values[sub_node], x - y)

    # Test multiplication
    mul_state = executor.State({x_node: x, y_node: y})
    for node in mul_plan:
        executor.evaluate(mul_state, node)
    assert np.array_equal(mul_state.values[mul_node], x * y)

    # Test division
    div_state = executor.State({x_node: x, y_node: y})
    for node in div_plan:
        executor.evaluate(div_state, node)
    assert np.array_equal(div_state.values[div_node], x / y)


def test_where_operation():
    """Test where operation with the executor."""
    # Create nodes
    cond_node = graph.placeholder("cond")
    x_node = graph.placeholder("x")
    y_node = graph.placeholder("y")
    where_node = graph.where(cond_node, x_node, y_node)

    # Create execution plan
    plan = linearize.forest(where_node)

    # Test case
    cond = np.array([[True, False, True], [False, True, False], [True, False, True]])
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
    expected = np.where(cond, x, y)

    # Evaluate with executor
    state = executor.State({cond_node: cond, x_node: x, y_node: y})
    for node in plan:
        executor.evaluate(state, node)

    # Check result
    assert np.array_equal(state.values[where_node], expected)


def test_multi_clause_where():
    """Test multi-clause where operations with the executor."""
    # Create nodes
    cond1 = graph.placeholder("cond1")
    cond2 = graph.placeholder("cond2")
    val1 = graph.constant(1.0)
    val2 = graph.constant(2.0)
    default = graph.constant(0.0)

    node = graph.multi_clause_where([(cond1, val1), (cond2, val2)], default)

    # Create execution plan
    plan = linearize.forest(node)

    # Test data
    cond1_val = np.array(
        [[True, False, False], [False, True, False], [False, False, True]]
    )
    cond2_val = np.array(
        [[False, True, False], [True, False, False], [False, True, False]]
    )
    expected = np.select([cond1_val, cond2_val], [1.0, 2.0], default=0.0)

    # Evaluate with executor
    state = executor.State({cond1: cond1_val, cond2: cond2_val})
    for node_in_plan in plan:
        executor.evaluate(state, node_in_plan)

    # Check result
    assert np.array_equal(state.values[node], expected)


def test_complex_graph():
    """Test execution of a complex graph with multiple operations."""
    # Build a more complex graph
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    # x^2 + 2*y
    x_squared = graph.power(x, graph.constant(2.0))
    two_y = graph.multiply(y, graph.constant(2.0))
    sum_term = graph.add(x_squared, two_y)

    # where(x > y, x^2 + 2*y, y - x)
    diff = graph.subtract(y, x)
    condition = graph.greater(x, y)
    result = graph.where(condition, sum_term, diff)

    # Create execution plan
    plan = linearize.forest(result)

    # Test data
    x_val = np.array([[1.0, 5.0], [3.0, 2.0]])
    y_val = np.array([[2.0, 3.0], [1.0, 4.0]])

    # Expected result calculations
    expected_x_squared = x_val**2
    expected_two_y = 2 * y_val
    expected_sum = expected_x_squared + expected_two_y
    expected_diff = y_val - x_val
    expected_condition = x_val > y_val
    expected_result = np.where(expected_condition, expected_sum, expected_diff)

    # Evaluate with executor
    state = executor.State({x: x_val, y: y_val})
    for node_in_plan in plan:
        executor.evaluate(state, node_in_plan)

    # Check intermediate and final results
    assert np.array_equal(state.values[x_squared], expected_x_squared)
    assert np.array_equal(state.values[two_y], expected_two_y)
    assert np.array_equal(state.values[sum_term], expected_sum)
    assert np.array_equal(state.values[diff], expected_diff)
    assert np.array_equal(state.values[condition], expected_condition)
    assert np.array_equal(state.values[result], expected_result)


def test_debug_flags(capsys, monkeypatch):
    """Test debug flags for tracing and breaking."""
    # Mock input function
    mock_input_calls = []

    def mock_input(prompt):
        mock_input_calls.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", mock_input)

    # Create a simple graph
    x = graph.placeholder("x")
    y = graph.multiply(x, graph.constant(2.0))

    # Set trace flag on node
    traced_y = graph.tracepoint(y)

    # Create execution plan
    plan = linearize.forest(traced_y)

    # Evaluate with trace flag
    state = executor.State({x: np.array([1.0, 2.0, 3.0])})
    for node in plan:
        executor.evaluate(state, node)

    # Check trace output
    captured = capsys.readouterr()
    assert "=== begin tracepoint ===" in captured.out

    # Test global trace flag
    z = graph.add(x, graph.constant(5.0))
    plan = linearize.forest(z)

    state = executor.State({x: np.array([1.0, 2.0, 3.0])}, flags=graph.NODE_FLAG_TRACE)
    for node in plan:
        executor.evaluate(state, node)

    captured = capsys.readouterr()
    assert "=== begin tracepoint ===" in captured.out

    # Test break flag on named node
    named_node = graph.add(x, graph.constant(10.0))
    named_node.name = "named_addition"
    plan = linearize.forest(named_node)

    state = executor.State({x: np.array([1.0, 2.0, 3.0])}, flags=graph.NODE_FLAG_BREAK)
    for node in plan:
        executor.evaluate(state, node)

    # Check that input was called (breakpoint triggered)
    assert len(mock_input_calls) > 0


def test_error_handling():
    """Test error handling in the executor."""
    # Create a node with missing dependency
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    z = graph.add(x, y)
    plan = linearize.forest(z)

    # Test missing placeholder value
    state = executor.State({x: np.array([1.0, 2.0])})  # y is missing
    with pytest.raises(executor.PlaceholderValueNotProvided):
        for node in plan:
            executor.evaluate(state, node)

    # Test unknown node type
    class unknown_node(graph.Node):
        pass

    with pytest.raises(executor.UnsupportedNodeType):
        executor.evaluate(executor.State({}), unknown_node())

    # Test unknown operation
    class unknown_binary(graph.BinaryOp):
        pass

    unknown_op = unknown_binary(graph.constant(1.0), graph.constant(2.0))
    plan = linearize.forest(unknown_op)
    state = executor.State({})

    with pytest.raises(executor.UnsupportedOperation):
        for node in plan:
            executor.evaluate(state, node)

    # Test evaluation ordering error (missing dependency)
    x = graph.placeholder("x")
    y = graph.add(x, graph.constant(1.0))

    # Not following the plan order - trying to evaluate y before x
    state = executor.State({x: np.array([1.0])})
    with pytest.raises(executor.NodeValueNotFound):
        executor.evaluate(state, y)  # Should fail, x not evaluated yet


def test_reduction_operations():
    """Test reduction operations with the executor."""
    # Create nodes
    x = graph.placeholder("x")
    sum_node = graph.reduce_sum(x, axis=0)
    mean_node = graph.reduce_mean(x, axis=1)

    # Create execution plans
    sum_plan = linearize.forest(sum_node)
    mean_plan = linearize.forest(mean_node)

    # Test data
    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Test sum reduction
    sum_state = executor.State({x: x_val})
    for node in sum_plan:
        executor.evaluate(sum_state, node)
    assert np.array_equal(sum_state.values[sum_node], np.sum(x_val, axis=0))

    # Test mean reduction
    mean_state = executor.State({x: x_val})
    for node in mean_plan:
        executor.evaluate(mean_state, node)
    assert np.array_equal(mean_state.values[mean_node], np.mean(x_val, axis=1))


def test_expand_dims_operation():
    """Test expand_dims operation with the executor."""
    # Create nodes
    x = graph.placeholder("x")
    expanded0 = graph.expand_dims(x, axis=0)
    expanded1 = graph.expand_dims(x, axis=1)
    expanded2 = graph.expand_dims(x, axis=2)

    # Create execution plans
    plan0 = linearize.forest(expanded0)
    plan1 = linearize.forest(expanded1)
    plan2 = linearize.forest(expanded2)

    # Test data
    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3

    # Test expand dims on axis 0
    state0 = executor.State({x: x_val})
    for node in plan0:
        executor.evaluate(state0, node)
    result0 = state0.values[expanded0]
    assert result0.shape == (1, 2, 3)
    assert np.array_equal(result0[0], x_val)

    # Test expand dims on axis 1
    state1 = executor.State({x: x_val})
    for node in plan1:
        executor.evaluate(state1, node)
    result1 = state1.values[expanded1]
    assert result1.shape == (2, 1, 3)
    assert np.array_equal(result1[:, 0, :], x_val)

    # Test expand dims on axis 2
    state2 = executor.State({x: x_val})
    for node in plan2:
        executor.evaluate(state2, node)
    result2 = state2.values[expanded2]
    assert result2.shape == (2, 3, 1)
    assert np.array_equal(result2[:, :, 0], x_val)


def test_execute_plan_helper():
    """Test a helper function to execute an entire plan in one go."""
    # This test demonstrates how you might create a helper function
    # to simplify the common pattern of executing a linearized plan

    def execute_plan(plan, initial_state):
        """Helper to execute a full plan with the given initial state."""
        state = initial_state
        for node in plan:
            executor.evaluate(state, node)
        return state

    # Create a simple graph
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    z = graph.add(x, y)
    w = graph.multiply(z, graph.constant(2.0))

    # Linearize
    plan = linearize.forest(w)

    # Execute the plan
    x_val = np.array([1.0, 2.0, 3.0])
    y_val = np.array([4.0, 5.0, 6.0])

    state = execute_plan(plan, executor.State({x: x_val, y: y_val}))

    # Check result
    assert np.array_equal(state.values[w], (x_val + y_val) * 2.0)


def test_comparison_operations():
    """Test comparison operations with the executor."""
    # Create placeholder nodes
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    # Create comparison nodes
    lt = graph.less(x, y)
    gt = graph.greater(x, y)
    eq = graph.equal(x, y)
    ne = graph.not_equal(x, y)
    le = graph.less_equal(x, y)
    ge = graph.greater_equal(x, y)

    # Create execution plans
    plans = {op: linearize.forest(op) for op in [lt, gt, eq, ne, le, ge]}

    # Test data
    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_val = np.array([[2.0, 2.0, 2.0], [5.0, 5.0, 5.0]])

    # Expected results
    expected = {
        lt: x_val < y_val,
        gt: x_val > y_val,
        eq: x_val == y_val,
        ne: x_val != y_val,
        le: x_val <= y_val,
        ge: x_val >= y_val,
    }

    # Test each operation
    for op, plan in plans.items():
        state = executor.State({x: x_val, y: y_val})
        for node in plan:
            executor.evaluate(state, node)
        assert np.array_equal(state.values[op], expected[op])


def test_state_value_access():
    """Test the State.get_node_value method for accessing node values."""
    # Create a node and a state
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.add(x, y)

    state = executor.State({x: np.array([1.0, 2.0, 3.0])})

    # Evaluate constant node
    executor.evaluate(state, y)

    # Test that node exists in state.values
    assert y in state.values


def test_placeholder_with_state_value():
    """Test evaluation of placeholders with values provided in the state."""
    # Create placeholder nodes with and without default values
    x_no_default = graph.placeholder("x_no_default")
    squared = graph.power(x_no_default, graph.constant(2.0))
    y_with_default = graph.placeholder("y_with_default", default_value=100.0)

    # Create plan for both nodes
    plan = linearize.forest(x_no_default, squared, y_with_default)

    # Test case 1: Both placeholders provided in state
    state1 = executor.State(
        {
            x_no_default: np.array([1.0, 2.0, 3.0]),
            y_with_default: np.array([4.0, 5.0, 6.0]),  # Overrides default value
        }
    )

    for node in plan:
        executor.evaluate(state1, node)

    assert np.array_equal(state1.values[x_no_default], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(state1.values[y_with_default], np.array([4.0, 5.0, 6.0]))

    # Test case 2: Only placeholder without default provided in state
    # (the other one should use default value)
    state2 = executor.State(
        {
            x_no_default: np.array([7.0, 8.0, 9.0]),
            # y_with_default not provided, should use default
        }
    )

    for node in plan:
        executor.evaluate(state2, node)

    assert np.array_equal(state2.values[x_no_default], np.array([7.0, 8.0, 9.0]))
    assert np.array_equal(state2.values[y_with_default], np.array(100.0))

    # Test case 3: Only placeholder with default provided in state
    # (the other one should fail)
    state3 = executor.State(
        {
            # x_no_default not provided, should fail
            y_with_default: np.array([10.0, 11.0, 12.0])
        }
    )

    # This should raise an error when x_no_default is evaluated
    with pytest.raises(executor.PlaceholderValueNotProvided):
        for node in plan:
            executor.evaluate(state3, node)


def test_unary_operations():
    """Test all supported unary operations and an unsupported one."""
    # Create placeholder node
    x = graph.placeholder("x")

    # Create various unary operation nodes
    exp_node = graph.exp(x)
    log_node = graph.log(x)
    not_node = graph.logical_not(x)

    # Create execution plans
    exp_plan = linearize.forest(exp_node)
    log_plan = linearize.forest(log_node)
    not_plan = linearize.forest(not_node)

    # Test data
    x_numeric = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    x_boolean = np.array([[True, False], [False, True]])

    # Test each operation with appropriate input type

    # Test exponential
    exp_state = executor.State({x: x_numeric})
    for node in exp_plan:
        executor.evaluate(exp_state, node)
    assert np.array_equal(exp_state.values[exp_node], np.exp(x_numeric))

    # Test logarithm (using positive values to avoid warnings)
    log_state = executor.State({x: np.abs(x_numeric)})
    for node in log_plan:
        executor.evaluate(log_state, node)
    assert np.array_equal(log_state.values[log_node], np.log(np.abs(x_numeric)))

    # Test logical not
    not_state = executor.State({x: x_boolean})
    for node in not_plan:
        executor.evaluate(not_state, node)
    assert np.array_equal(not_state.values[not_node], np.logical_not(x_boolean))

    # Test unsupported unary operation
    class UnsupportedUnaryOp(graph.UnaryOp):
        pass

    unsupported_node = UnsupportedUnaryOp(x)
    unsupported_plan = linearize.forest(unsupported_node)

    unsupported_state = executor.State({x: x_numeric})

    with pytest.raises(executor.UnsupportedOperation):
        for node in unsupported_plan:
            executor.evaluate(unsupported_state, node)


def test_axis_operations():
    """Test all supported axis operations and an unsupported one."""
    # Create placeholder node
    x = graph.placeholder("x")

    # Create various axis operation nodes
    expand_node = graph.expand_dims(x, axis=1)
    sum_node = graph.reduce_sum(x, axis=0)
    mean_node = graph.reduce_mean(x, axis=1)

    # Create execution plans
    expand_plan = linearize.forest(expand_node)
    sum_plan = linearize.forest(sum_node)
    mean_plan = linearize.forest(mean_node)

    # Test data
    x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # 3x3

    # Test expand_dims
    expand_state = executor.State({x: x_val})
    for node in expand_plan:
        executor.evaluate(expand_state, node)
    expected_expand = np.expand_dims(x_val, axis=1)
    assert np.array_equal(expand_state.values[expand_node], expected_expand)
    assert expand_state.values[expand_node].shape == (3, 1, 3)

    # Test reduce_sum
    sum_state = executor.State({x: x_val})
    for node in sum_plan:
        executor.evaluate(sum_state, node)
    expected_sum = np.sum(x_val, axis=0)
    assert np.array_equal(sum_state.values[sum_node], expected_sum)

    # Test reduce_mean
    mean_state = executor.State({x: x_val})
    for node in mean_plan:
        executor.evaluate(mean_state, node)
    expected_mean = np.mean(x_val, axis=1)
    assert np.array_equal(mean_state.values[mean_node], expected_mean)

    # Test unsupported axis operation
    class UnsupportedAxisOp(graph.AxisOp):
        pass

    unsupported_node = UnsupportedAxisOp(x, axis=0)
    unsupported_plan = linearize.forest(unsupported_node)

    unsupported_state = executor.State({x: x_val})

    with pytest.raises(executor.UnsupportedOperation):
        for node in unsupported_plan:
            executor.evaluate(unsupported_state, node)

    # Test invalid axis value
    with pytest.raises(ValueError):  # NumPy raises ValueError for invalid axes
        # Create a valid node type but with invalid axis
        # Note: x is only 2D, so axis=5 is invalid
        invalid_axis_node = graph.reduce_sum(
            x, axis=5
        )
        invalid_plan = linearize.forest(invalid_axis_node)
        invalid_state = executor.State({x: x_val})

        for node in invalid_plan:
            executor.evaluate(invalid_state, node)


def test_state_post_init_tracing(capsys):
    """Test that State.__post_init__ traces initial values when NODE_FLAG_TRACE is set."""
    # Create nodes
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    # Test data
    x_val = np.array([1.0, 2.0, 3.0])
    y_val = np.array([4.0, 5.0, 6.0])

    # Create state with tracing flag
    state = executor.State({x: x_val, y: y_val}, flags=graph.NODE_FLAG_TRACE)

    # Check that tracing output was generated during initialization
    captured = capsys.readouterr()
    output = captured.out

    # Verify that both nodes were traced in the output
    assert f"name: {x.name}" in output
    assert f"name: {y.name}" in output
    assert "=== begin tracepoint ===" in output

    # Verify the cached indication is shown
    assert "cached: True" in output
