"""Tests for the dt_model.internal.sympyke.piecewise package."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.engine.frontend import linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.internal.sympyke import Piecewise, Symbol


def test_piecewise_basics():
    """Make sure that Piecewise works as intended with Symbol."""
    # Create the placeholders as symbols
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    condition1 = Symbol("condition1")
    condition2 = Symbol("condition2")

    # Create the resulting piecewise function
    pw = Piecewise(
        (x.node, condition1.node),
        (y.node, condition2.node),
        (z.node, True),
    )

    # Linearize an execution plan out of the piecewise
    plan = linearize.forest(pw)

    # Create the evaluation state with concrete values
    state = executor.State(
        {
            x.node: np.array([2, 9, 16]),
            y.node: np.array([8, 27, 64]),
            z.node: np.array([16, 81, 256]),
            condition1.node: np.array([True, False, False]),
            condition2.node: np.array([False, True, False]),
        }
    )

    # Actually evaluate the piecewise function
    for node in plan:
        executor.evaluate(state, node)

    # Ensure the result is the expected one
    expect = np.array([2, 27, 256])
    rv = state.values[pw]
    assert np.all(rv == expect)


def test_piecewise_with_scalars():
    """Test Piecewise with scalar values."""
    result = Piecewise((2, False), (1, True))

    # Linearize
    plan = linearize.forest(result)
    state = executor.State(values={})

    # Evaluate
    for node in plan:
        executor.evaluate(state, node)

    assert state.values[result] == 1


def test_piecewise_empty():
    """Test Piecewise with no clauses raises ValueError."""
    with pytest.raises(ValueError):
        Piecewise()


def test_piecewise_filtering():
    """Test the internal _filter_clauses function."""
    from civic_digital_twins.dt_model.internal.sympyke.piecewise import _filter_clauses

    clauses = (
        (1, False),
        (2, True),
        (3, False),  # Should be filtered out
        (4, True),  # Should be filtered out
    )

    filtered = _filter_clauses(clauses)
    assert len(filtered) == 2
    assert filtered[0] == (1, False)
    assert filtered[1] == (2, True)


def test_piecewise_with_constant_conditions():
    """Test Piecewise functionality with constant conditions."""
    # Create expression symbols
    expr1 = Symbol("expr1")
    expr2 = Symbol("expr2")
    expr3 = Symbol("expr3")

    # Create piecewise with constant conditions
    pw = Piecewise(
        (expr1.node, False),  # Constant False condition
        (expr2.node, True),  # Constant True condition
        (expr3.node, True),  # This should be ignored as it's after a True condition
    )

    # Linearize the execution plan
    plan = linearize.forest(pw)

    # Set up evaluation state with tensor values
    state = executor.State(
        {
            expr1.node: np.array([10, 20, 30]),
            expr2.node: np.array([40, 50, 60]),
            expr3.node: np.array([70, 80, 90]),
        }
    )

    # Evaluate the piecewise function
    for node in plan:
        executor.evaluate(state, node)

    # Since the second condition is True, the result should be expr2
    result = state.values[pw]
    expected = np.array([40, 50, 60])

    assert np.array_equal(result, expected)


def test_piecewise_only_default_case():
    """Test when filtering clauses leaves only the default case."""
    # Create piecewise where all non-default clauses come after a True condition
    pw = Piecewise(
        (1, True),  # This becomes the default case
        (10, False),  # This gets filtered out
    )

    # Linearize
    plan = linearize.forest(pw)
    state = executor.State(values={})

    # Evaluate
    for node in plan:
        executor.evaluate(state, node)

    assert state.values[pw] == 1
