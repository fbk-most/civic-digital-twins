"""Tests for the civic_digital_twins.dt_model.internal.sympyke.operators module."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np

from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.internal.sympyke import Eq, Symbol


def test_eq_basics():
    """Test basic functionality of the Eq operator."""
    # Create symbols
    x = Symbol("x")
    y = Symbol("y")

    # Create equality operation
    equality = Eq(x, y)

    # Linearize and evaluate
    plan = linearize.forest(equality)
    state = executor.State(
        {
            x.node: np.array([1, 2, 3, 2]),
            y.node: np.array([1, 3, 3, 1]),
        }
    )

    executor.evaluate_nodes(state, *plan)

    # Expected: True, False, True, False
    expected = np.array([True, False, True, False])
    assert np.array_equal(state.values[equality], expected)


def test_eq_with_constants():
    """Test Eq with constant values."""
    x = Symbol("x")

    # Equality with constant on right
    eq1 = Eq(x, 3)

    # Equality with constant on left
    eq2 = Eq(3, x)

    # Evaluate both
    plan = linearize.forest(eq1, eq2)
    state = executor.State(
        {
            x.node: np.array([1, 2, 3, 4]),
        }
    )

    executor.evaluate_nodes(state, *plan)

    expected = np.array([False, False, True, False])
    assert np.array_equal(state.values[eq1], expected)
    assert np.array_equal(state.values[eq2], expected)


def test_eq_with_mixed_inputs():
    """Test Eq with a mix of Symbol and graph.Node inputs."""
    x = Symbol("x")
    y = graph.placeholder("y")

    eq = Eq(x, y)

    # Linearize and evaluate
    plan = linearize.forest(eq)
    state = executor.State(
        {
            x.node: np.array([1, 2, 3]),
            y: np.array([1, 0, 3]),
        }
    )

    executor.evaluate_nodes(state, *plan)

    expected = np.array([True, False, True])
    assert np.array_equal(state.values[eq], expected)


def test_eq_chaining():
    """Test multiple equality operations together."""
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    # Create x == y and y == z
    eq_xy = Eq(x, y)
    eq_yz = Eq(y, z)

    # Combine with logical and
    result = eq_xy & eq_yz

    # Linearize and evaluate
    plan = linearize.forest(result)
    state = executor.State(
        {
            x.node: np.array([1, 2, 3, 4]),
            y.node: np.array([1, 2, 2, 5]),
            z.node: np.array([1, 2, 3, 5]),
        }
    )

    executor.evaluate_nodes(state, *plan)

    # Expected: True, True, False, False
    expected = np.array([True, True, False, False])
    assert np.array_equal(state.values[result], expected)
