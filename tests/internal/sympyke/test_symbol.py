"""Tests for the dt_model.internal.sympyke.symbol package."""

# SPDX-License-Identifier: Apache-2.0

from dt_model.internal.sympyke import Symbol
from dt_model.engine.frontend import linearize
from dt_model.engine.numpybackend import executor

import numpy as np


def test_symbol_basics():
    """Test basic Symbol functionality."""
    # Create a new symbol
    x = Symbol("x")

    # Check its properties
    assert x.name == "x"
    assert x.node is not None
    assert x.node.name == "x"


def test_symbol_reuse():
    """Test that symbols with the same name are reused."""
    # Create symbols with the same name
    x1 = Symbol("same_name")
    x2 = Symbol("same_name")

    # Check that they refer to the same node
    assert x1 is x2
    assert x1.node is x2.node


def test_symbol_execution():
    """Test execution with Symbol placeholders."""
    # Create symbols
    x = Symbol("x")
    y = Symbol("y")

    # Create a simple computation
    result = x.node + y.node

    # Linearize the execution plan
    plan = linearize.forest(result)

    # Set up evaluation state
    state = executor.State(
        {
            x.node: np.array([1, 2, 3]),
            y.node: np.array([4, 5, 6]),
        }
    )

    # Evaluate
    for node in plan:
        executor.evaluate(state, node)

    # Check result
    expected = np.array([5, 7, 9])
    assert np.array_equal(state.values[result], expected)


def test_symbol_multiple_operations():
    """Test multiple operations on symbols."""
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    # Create a more complex expression: (x + y) * z
    result = (x.node + y.node) * z.node

    # Linearize and evaluate
    plan = linearize.forest(result)
    state = executor.State(
        {
            x.node: np.array([1, 2, 3]),
            y.node: np.array([4, 5, 6]),
            z.node: np.array([2, 3, 4]),
        }
    )

    for node in plan:
        executor.evaluate(state, node)

    # Expected: (1+4)*2, (2+5)*3, (3+6)*4 = 10, 21, 36
    expected = np.array([10, 21, 36])
    assert np.array_equal(state.values[result], expected)


def test_symbol_table_values():
    """Test accessing all symbols from the symbol table."""
    # Clear any existing symbols by recreating the table
    import dt_model.internal.sympyke.symbol as symbol_mod

    original_table = symbol_mod.symbol_table
    symbol_mod.symbol_table = symbol_mod._SymbolTable()

    try:
        # Create some symbols
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")

        # Get all symbols from the table
        all_symbols = symbol_mod.symbol_table.values()

        # Check that all created symbols are in the table
        assert len(all_symbols) == 3
        symbol_names = {s.name for s in all_symbols}
        assert symbol_names == {"a", "b", "c"}
    finally:
        # Restore the original symbol table
        symbol_mod.symbol_table = original_table
