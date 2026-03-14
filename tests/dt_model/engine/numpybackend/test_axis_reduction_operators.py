"""Comprehensive tests for all axis reduction operators.

This module provides unified tests for all axis reduction operators (both sum/mean
and min/max/std/var/median/prod/any/all/count_nonzero/quantile), covering:
- Node creation and properties
- String representation
- Execution with various data types and shapes
- keepdims semantics
- Edge cases
- Operator composition

Issues: #116, #117
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

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


class TestAxisOperatorCreation:
    """Test creation of axis reduction operator nodes."""

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_creation(self, operator_class, _, name):
        """Test creating an axis reduction operator node."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=0)
        assert result.node is x
        assert result.axis == 0
        assert isinstance(result, operator_class)

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_creation_different_axes(self, operator_class, _, name):
        """Test creating operator nodes with different axis values."""
        x = graph.placeholder("x")
        for axis in [0, 1, -1, -2]:
            result = operator_class(x, axis=axis)
            assert result.axis == axis

    def test_quantile_creation(self):
        """Test creating a quantile operator node."""
        x = graph.placeholder("x")
        result = QUANTILE_OPERATOR(x, axis=0, q=0.5)
        assert result.node is x
        assert result.axis == 0
        assert result.q == 0.5
        assert isinstance(result, QUANTILE_OPERATOR)

    def test_quantile_different_q_values(self):
        """Test quantile with different q values."""
        x = graph.placeholder("x")
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = QUANTILE_OPERATOR(x, axis=0, q=q)
            assert result.q == q


class TestAxisOperatorRepr:
    """Test string representation of axis reduction operators."""

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_repr_contains_name(self, operator_class, _, name):
        """Test that repr contains operator name."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=0, name=f"test_{name}")
        repr_str = repr(result)
        assert name in repr_str.lower()
        assert "axis=0" in repr_str

    def test_quantile_repr_contains_q(self):
        """Test that quantile repr contains q value."""
        x = graph.placeholder("x")
        result = QUANTILE_OPERATOR(x, axis=0, q=0.75, name="test_quantile")
        repr_str = repr(result)
        assert "quantile" in repr_str.lower()
        assert "q=0.75" in repr_str


class TestAxisOperatorExecution:
    """Test execution of axis reduction operators."""

    @pytest.mark.parametrize("operator_class,numpy_func,name", AXIS_OPERATORS)
    def test_operator_execution_axis0(self, operator_class, numpy_func, name):
        """Test operator execution with axis=0."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=0)
        plan = linearize.forest(result)

        # Handle boolean operators specially
        if name in ("any", "all"):
            x_val = np.array([[True, False, False], [False, False, False]])
        else:
            x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])

        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = numpy_func(x_val, axis=0, keepdims=True)
        if name in ("std", "var", "median"):
            assert np.allclose(state.values[result], expected)
        else:
            assert np.array_equal(state.values[result], expected)

    @pytest.mark.parametrize("operator_class,numpy_func,name", AXIS_OPERATORS)
    def test_operator_execution_axis1(self, operator_class, numpy_func, name):
        """Test operator execution with axis=1."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=1)
        plan = linearize.forest(result)

        # Handle boolean operators specially
        if name in ("any", "all"):
            x_val = np.array([[True, True, False], [True, True, True]])
        else:
            x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])

        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = numpy_func(x_val, axis=1, keepdims=True)
        if name in ("std", "var", "median"):
            assert np.allclose(state.values[result], expected)
        else:
            assert np.array_equal(state.values[result], expected)

    @pytest.mark.parametrize("operator_class,numpy_func,name", AXIS_OPERATORS)
    def test_operator_execution_axis_negative(self, operator_class, numpy_func, name):
        """Test operator execution with negative axis."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=-1)
        plan = linearize.forest(result)

        # Handle boolean operators specially
        if name in ("any", "all"):
            x_val = np.array([[True, False], [True, True], [False, True]])
        else:
            x_val = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

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
            result = QUANTILE_OPERATOR(x, axis=0, q=q)
            plan = linearize.forest(result)

            state = executor.State({x: x_val})
            executor.evaluate_nodes(state, *plan)

            expected = np.quantile(x_val, q, axis=0, keepdims=True)
            assert np.allclose(state.values[result], expected)


class TestAxisOperatorKeepdims:
    """Test that all axis operators preserve keepdims semantics."""

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_keepdims_axis0(self, operator_class, _, name):
        """Test that reducing axis 0 results in shape (1, n_cols)."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Handle boolean operators
        if name in ("any", "all"):
            x_val = np.array([[True, False, True], [False, True, False]])

        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        assert state.values[result].shape == (1, 3)

    @pytest.mark.parametrize("operator_class,_,name", AXIS_OPERATORS)
    def test_operator_keepdims_axis1(self, operator_class, _, name):
        """Test that reducing axis 1 results in shape (n_rows, 1)."""
        x = graph.placeholder("x")
        result = operator_class(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Handle boolean operators
        if name in ("any", "all"):
            x_val = np.array([[True, False, True], [False, True, False]])

        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        assert state.values[result].shape == (2, 1)

    def test_quantile_keepdims_axis0(self):
        """Test that quantile reducing axis 0 results in shape (1, n_cols)."""
        x = graph.placeholder("x")
        result = QUANTILE_OPERATOR(x, axis=0, q=0.5)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        assert state.values[result].shape == (1, 3)

    def test_quantile_keepdims_axis1(self):
        """Test that quantile reducing axis 1 results in shape (n_rows, 1)."""
        x = graph.placeholder("x")
        result = QUANTILE_OPERATOR(x, axis=1, q=0.5)
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
        result = operator_class(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[42.0]])

        # Handle boolean operators
        if name in ("any", "all"):
            x_val = np.array([[True]])

        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        # Result should always maintain shape with keepdims
        assert state.values[result].shape == (1, 1)

    def test_operator_min_with_negative_numbers(self):
        """Test min operator with negative numbers."""
        x = graph.placeholder("x")
        result = graph.project_using_min(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[-1.0, -5.0, -3.0], [-2.0, -4.0, -6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(x_val, axis=1, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_operator_prod_with_zeros(self):
        """Test prod operator with zeros."""
        x = graph.placeholder("x")
        result = graph.project_using_prod(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.prod(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_operator_std_with_uniform_values(self):
        """Test std operator with uniform values (should be zero)."""
        x = graph.placeholder("x")
        result = graph.project_using_std(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.std(x_val, axis=0, keepdims=True)
        assert np.allclose(state.values[result], expected)
        assert np.allclose(state.values[result], 0.0)

    def test_operator_count_nonzero_all_zeros(self):
        """Test count_nonzero operator with all zeros."""
        x = graph.placeholder("x")
        result = graph.project_using_count_nonzero(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.count_nonzero(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)
        assert np.array_equal(state.values[result], [[0, 0, 0]])

    def test_quantile_extremes(self):
        """Test quantile operator with extreme q values."""
        x = graph.placeholder("x")
        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])

        # q=0.0 should give minimum
        result_min = QUANTILE_OPERATOR(x, axis=0, q=0.0)
        plan_min = linearize.forest(result_min)
        state_min = executor.State({x: x_val})
        executor.evaluate_nodes(state_min, *plan_min)
        expected_min = np.quantile(x_val, 0.0, axis=0, keepdims=True)
        assert np.allclose(state_min.values[result_min], expected_min)

        # q=1.0 should give maximum
        result_max = QUANTILE_OPERATOR(x, axis=0, q=1.0)
        plan_max = linearize.forest(result_max)
        state_max = executor.State({x: x_val})
        executor.evaluate_nodes(state_max, *plan_max)
        expected_max = np.quantile(x_val, 1.0, axis=0, keepdims=True)
        assert np.allclose(state_max.values[result_max], expected_max)


class TestAxisOperatorComposition:
    """Test composition of axis reduction operators."""

    def test_nested_min_max(self):
        """Test composing min and max operators."""
        x = graph.placeholder("x")
        min_result = graph.project_using_min(x, axis=1)
        max_result = graph.project_using_max(min_result, axis=0)
        plan = linearize.forest(max_result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.max(np.min(x_val, axis=1, keepdims=True), axis=0, keepdims=True)
        assert np.array_equal(state.values[max_result], expected)

    def test_std_of_prod(self):
        """Test std of prod."""
        x = graph.placeholder("x")
        prod_result = graph.project_using_prod(x, axis=0)
        std_result = graph.project_using_std(prod_result, axis=1)
        plan = linearize.forest(std_result)

        x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.std(np.prod(x_val, axis=0, keepdims=True), axis=1, keepdims=True)
        assert np.allclose(state.values[std_result], expected)

    def test_quantile_of_mean(self):
        """Test quantile of mean."""
        x = graph.placeholder("x")
        mean_result = graph.project_using_mean(x, axis=0)
        quantile_result = QUANTILE_OPERATOR(mean_result, axis=1, q=0.5)
        plan = linearize.forest(quantile_result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(np.mean(x_val, axis=0, keepdims=True), 0.5, axis=1, keepdims=True)
        assert np.allclose(state.values[quantile_result], expected)

    def test_any_of_all(self):
        """Test any of all."""
        x = graph.placeholder("x")
        all_result = graph.project_using_all(x, axis=1)
        any_result = graph.project_using_any(all_result, axis=0)
        plan = linearize.forest(any_result)

        x_val = np.array([[True, True, False], [True, True, True]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.any(np.all(x_val, axis=1, keepdims=True), axis=0, keepdims=True)
        assert np.array_equal(state.values[any_result], expected)

    def test_sum_of_min(self):
        """Test sum of min."""
        x = graph.placeholder("x")
        min_result = graph.project_using_min(x, axis=0)
        sum_result = graph.project_using_sum(min_result, axis=1)
        plan = linearize.forest(sum_result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.sum(np.min(x_val, axis=0, keepdims=True), axis=1, keepdims=True)
        assert np.allclose(state.values[sum_result], expected)


class TestAxisOperatorBroadcasting:
    """Test broadcasting behavior with axis reduction operators."""

    def test_operator_3d_array_axis0(self):
        """Test operator on 3D array reducing axis 0."""
        x = graph.placeholder("x")
        result = graph.project_using_sum(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.sum(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)
        assert state.values[result].shape == (1, 2, 2)

    def test_operator_3d_array_axis1(self):
        """Test operator on 3D array reducing axis 1."""
        x = graph.placeholder("x")
        result = graph.project_using_mean(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.mean(x_val, axis=1, keepdims=True)
        assert np.allclose(state.values[result], expected)
        assert state.values[result].shape == (2, 1, 2)

    def test_operator_3d_array_axis2(self):
        """Test operator on 3D array reducing axis 2."""
        x = graph.placeholder("x")
        result = graph.project_using_min(x, axis=2)
        plan = linearize.forest(result)

        x_val = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(x_val, axis=2, keepdims=True)
        assert np.array_equal(state.values[result], expected)
        assert state.values[result].shape == (2, 2, 1)
