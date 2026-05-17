"""Comprehensive tests for all axis reduction operators.

This module provides unified tests for all axis reduction operators, covering:
- Node creation and properties
- String representation
- Execution (all semantic axes map to numpy axis -1 by CDT convention)
- keepdims semantics
- Edge cases
- Operator composition

Issues: #116, #117
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import DOMAIN, Axis
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor

# All axis reduction operators and their corresponding numpy functions
AXIS_OPERATORS = [
    (graph.project_using_sum, np.sum, "sum"),
    (graph.project_using_mean, np.mean, "mean"),
    (graph.project_using_min, np.min, "min"),
    (graph.project_using_max, np.max, "max"),
    (graph.project_using_std, np.std, "std"),
    (graph.project_using_var, np.var, "var"),
    (graph.project_using_median, np.median, "median"),
    (graph.project_using_prod, np.prod, "prod"),
    (graph.project_using_any, np.any, "any"),
    (graph.project_using_all, np.all, "all"),
    (graph.project_using_count_nonzero, np.count_nonzero, "count_nonzero"),
]

# Operators that require special q parameter
QUANTILE_OPERATOR = graph.project_using_quantile

TIME_AXIS = Axis("time", DOMAIN)


class TestAxisOperatorCreation:
    """Test creation of axis reduction operator nodes."""

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_creation(self, operator_class, _, name):
        """Test creating an axis reduction operator node."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=TIME_AXIS)
        assert result.node is x
        assert result.axis == TIME_AXIS
        assert isinstance(result, operator_class)

    def test_quantile_creation(self):
        """Test creating a quantile operator node."""
        x = graph.placeholder("x")
        result = QUANTILE_OPERATOR(x, axis=TIME_AXIS, q=0.5)
        assert result.node is x
        assert result.axis == TIME_AXIS
        assert result.q == 0.5
        assert isinstance(result, QUANTILE_OPERATOR)

    def test_quantile_different_q_values(self):
        """Test quantile with different q values."""
        x = graph.placeholder("x")
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = QUANTILE_OPERATOR(x, axis=TIME_AXIS, q=q)
            assert result.q == q


class TestAxisOperatorRepr:
    """Test string representation of axis reduction operators."""

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_repr_contains_name(self, operator_class, _, name):
        """Test that repr contains operator name and semantic axis."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=TIME_AXIS, name=f"test_{name}")
        repr_str = repr(result)
        assert name in repr_str.lower()
        assert "Axis('time', role='DOMAIN')" in repr_str

    def test_quantile_repr_contains_q(self):
        """Test that quantile repr contains q value."""
        x = graph.placeholder("x")
        result = QUANTILE_OPERATOR(x, axis=TIME_AXIS, q=0.75, name="test_quantile")
        repr_str = repr(result)
        assert "quantile" in repr_str.lower()
        assert "q=0.75" in repr_str


class TestAxisOperatorExecution:
    """Test execution of axis reduction operators.

    All ProjectionOp nodes reduce along numpy axis -1 (CDT convention:
    the DOMAIN axis always occupies the last numpy dimension).
    """

    @pytest.mark.parametrize("operator_class,numpy_func,name", AXIS_OPERATORS)
    def test_operator_execution(self, operator_class, numpy_func, name):
        """Test operator execution: always reduces the last (time) dimension."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        # rows = parameters, cols = time steps (last dim is always reduced)
        if name in ("any", "all"):
            x_val = np.array([[True, False, False], [False, False, False]])
        else:
            x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])

        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = numpy_func(x_val, axis=-1, keepdims=True)
        if name in ("std", "var", "median"):
            assert np.allclose(state.values[result], expected)
        else:
            assert np.array_equal(state.values[result], expected)

    def test_quantile_execution_various_q_values(self):
        """Test quantile execution with different q values."""
        x = graph.placeholder("x")
        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])

        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = QUANTILE_OPERATOR(x, axis=TIME_AXIS, q=q)
            plan = linearize.forest(result)

            state = executor.State({x: x_val})
            executor.evaluate_nodes(state, *plan)

            expected = np.quantile(x_val, q, axis=-1, keepdims=True)
            assert np.allclose(state.values[result], expected)


class TestAxisOperatorKeepdims:
    """Test that all axis operators preserve keepdims semantics."""

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_keepdims(self, operator_class, _, name):
        """Test that reducing the last dim results in shape (n_rows, 1)."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        if name in ("any", "all"):
            x_val = np.array([[True, False, True], [False, True, False]])

        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        assert state.values[result].shape == (2, 1)

    def test_quantile_keepdims(self):
        """Test that quantile reducing the last dim results in shape (n_rows, 1)."""
        x = graph.placeholder("x")
        result = QUANTILE_OPERATOR(x, axis=TIME_AXIS, q=0.5)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        assert state.values[result].shape == (2, 1)


class TestAxisOperatorEdgeCases:
    """Test edge cases for axis reduction operators."""

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_single_element(self, operator_class, _, name):
        """Test operator on single element array."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        x_val = np.array([[42.0]])

        if name in ("any", "all"):
            x_val = np.array([[True]])

        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        assert state.values[result].shape == (1, 1)

    def test_operator_min_with_negative_numbers(self):
        """Test min operator with negative numbers."""
        x = graph.placeholder("x")
        result = graph.project_using_min(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        x_val = np.array([[-1.0, -5.0, -3.0], [-2.0, -4.0, -6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(x_val, axis=-1, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_operator_prod_with_zeros(self):
        """Test prod operator with zeros."""
        x = graph.placeholder("x")
        result = graph.project_using_prod(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.prod(x_val, axis=-1, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_operator_std_with_uniform_values(self):
        """Test std operator with uniform values (should be zero)."""
        x = graph.placeholder("x")
        result = graph.project_using_std(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        x_val = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.std(x_val, axis=-1, keepdims=True)
        assert np.allclose(state.values[result], expected)
        assert np.allclose(state.values[result], 0.0)

    def test_operator_count_nonzero_all_zeros(self):
        """Test count_nonzero operator with all zeros."""
        x = graph.placeholder("x")
        result = graph.project_using_count_nonzero(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        x_val = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.count_nonzero(x_val, axis=-1, keepdims=True)
        assert np.array_equal(state.values[result], expected)
        assert np.array_equal(state.values[result], [[0], [0]])

    def test_quantile_extremes(self):
        """Test quantile operator with extreme q values."""
        x = graph.placeholder("x")
        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])

        # q=0.0 should give minimum
        result_min = QUANTILE_OPERATOR(x, axis=TIME_AXIS, q=0.0)
        plan_min = linearize.forest(result_min)
        state_min = executor.State({x: x_val})
        executor.evaluate_nodes(state_min, *plan_min)
        expected_min = np.quantile(x_val, 0.0, axis=-1, keepdims=True)
        assert np.allclose(state_min.values[result_min], expected_min)

        # q=1.0 should give maximum
        result_max = QUANTILE_OPERATOR(x, axis=TIME_AXIS, q=1.0)
        plan_max = linearize.forest(result_max)
        state_max = executor.State({x: x_val})
        executor.evaluate_nodes(state_max, *plan_max)
        expected_max = np.quantile(x_val, 1.0, axis=-1, keepdims=True)
        assert np.allclose(state_max.values[result_max], expected_max)


class TestAxisOperatorComposition:
    """Test composition of axis reduction operators.

    Since all reductions use axis=-1, compositions reduce the last dim twice
    in sequence — useful for testing (params, time) → (params, 1) → (1, 1).
    """

    def test_nested_sum_mean(self):
        """Test composing sum then mean along the same semantic axis."""
        x = graph.placeholder("x")
        sum_result = graph.project_using_sum(x, axis=TIME_AXIS)
        mean_result = graph.project_using_mean(sum_result, axis=TIME_AXIS)
        plan = linearize.forest(mean_result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.mean(np.sum(x_val, axis=-1, keepdims=True), axis=-1, keepdims=True)
        assert np.allclose(state.values[mean_result], expected)

    def test_std_of_prod(self):
        """Test std of prod."""
        x = graph.placeholder("x")
        prod_result = graph.project_using_prod(x, axis=TIME_AXIS)
        std_result = graph.project_using_std(prod_result, axis=TIME_AXIS)
        plan = linearize.forest(std_result)

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.std(np.prod(x_val, axis=-1, keepdims=True), axis=-1, keepdims=True)
        assert np.allclose(state.values[std_result], expected)

    def test_quantile_of_mean(self):
        """Test quantile of mean."""
        x = graph.placeholder("x")
        mean_result = graph.project_using_mean(x, axis=TIME_AXIS)
        quantile_result = QUANTILE_OPERATOR(mean_result, axis=TIME_AXIS, q=0.5)
        plan = linearize.forest(quantile_result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(np.mean(x_val, axis=-1, keepdims=True), 0.5, axis=-1, keepdims=True)
        assert np.allclose(state.values[quantile_result], expected)

    def test_any_of_all(self):
        """Test any of all."""
        x = graph.placeholder("x")
        all_result = graph.project_using_all(x, axis=TIME_AXIS)
        any_result = graph.project_using_any(all_result, axis=TIME_AXIS)
        plan = linearize.forest(any_result)

        x_val = np.array([[True, True, False], [True, True, True]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.any(np.all(x_val, axis=-1, keepdims=True), axis=-1, keepdims=True)
        assert np.array_equal(state.values[any_result], expected)


class TestAxisOperatorBroadcasting:
    """Test broadcasting behavior with axis reduction operators."""

    def test_operator_3d_array(self):
        """Test operator on 3D (batch, params, time) array reducing the time dimension."""
        x = graph.placeholder("x")
        result = graph.project_using_sum(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        x_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # (2, 2, 2)
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.sum(x_val, axis=-1, keepdims=True)
        assert np.array_equal(state.values[result], expected)
        assert state.values[result].shape == (2, 2, 1)

    def test_operator_mean_3d(self):
        """Test mean on 3D array."""
        x = graph.placeholder("x")
        result = graph.project_using_mean(x, axis=TIME_AXIS)
        plan = linearize.forest(result)

        x_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # (2, 2, 2)
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.mean(x_val, axis=-1, keepdims=True)
        assert np.allclose(state.values[result], expected)
        assert state.values[result].shape == (2, 2, 1)
