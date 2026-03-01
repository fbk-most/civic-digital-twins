"""Tests for the JIT layer in numpy_ast (forest_tree_to_ast_function_def, compile_function, call_function)."""

# SPDX-License-Identifier: Apache-2.0

import ast

import numpy as np
import pytest

from civic_digital_twins.dt_model.engine.frontend import forest, graph
from civic_digital_twins.dt_model.engine.numpybackend import numpy_ast


def _make_simple_tree() -> tuple[forest.Tree, graph.placeholder, graph.placeholder]:
    """Build a simple tree: root = exp(a) + b."""
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    root = graph.add(graph.exp(a), b)
    trees = forest.partition(root)
    return trees[0], a, b


def test_forest_tree_to_ast_function_def_structure():
    """Verify that forest_tree_to_ast_function_def produces a valid FunctionDef."""
    tree, a, b = _make_simple_tree()
    func_def = numpy_ast.forest_tree_to_ast_function_def(tree)

    # Must be an ast.FunctionDef
    assert isinstance(func_def, ast.FunctionDef)

    # Function name encodes the root node id
    assert func_def.name == f"t{tree.root().id}"

    # One argument per tree input, sorted by id
    assert len(func_def.args.args) == len(tree.inputs)
    arg_names = {arg.arg for arg in func_def.args.args}
    expected_names = {f"n{node.id}" for node in tree.inputs}
    assert arg_names == expected_names

    # Body is non-empty and last statement is a Return
    assert len(func_def.body) > 0
    assert isinstance(func_def.body[-1], ast.Return)


def test_python_compile_function():
    """Verify that _python_compile_function produces a callable with correct output."""
    tree, a, b = _make_simple_tree()
    func = numpy_ast._python_compile_function(tree)

    a_val = np.asarray([1.0, 2.0])
    b_val = np.asarray([10.0, 20.0])

    # inputs are sorted by id; pass in that order
    input_vals = {node.id: val for node, val in zip([a, b], [a_val, b_val])}
    ordered_inputs = [input_vals[node.id] for node in tree.inputs]
    result = func(*ordered_inputs)

    expected = np.exp(a_val) + b_val
    np.testing.assert_allclose(result, expected)


def test_compile_and_call_function():
    """Verify that compile_function + call_function produce correct results."""
    pytest.importorskip("numba", reason="numba not installed; skipping JIT test")

    a = graph.placeholder("a")
    b = graph.placeholder("b")
    root = graph.multiply(a, b)
    tree = forest.partition(root)[0]

    compiled = numpy_ast.compile_function(tree)

    a_val = np.asarray([2.0, 3.0])
    b_val = np.asarray([4.0, 5.0])

    # Build a minimal state-like object
    class _FakeState:
        def __init__(self, vals: dict) -> None:
            self._vals = vals

        def get_node_value(self, node: graph.Node) -> np.ndarray:
            return self._vals[node.id]

    state = _FakeState({a.id: a_val, b.id: b_val})
    result = numpy_ast.call_function(state, tree, compiled)  # type: ignore[arg-type]

    np.testing.assert_allclose(result, a_val * b_val)
