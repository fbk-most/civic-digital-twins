"""Comprehensive tests for all index reduction methods.

This module provides unified tests for all index reduction methods (both sum/mean
and min/max/std/var/median/prod/any/all/count_nonzero/quantile), covering:
- Method creation and return types
- Execution with various data types and shapes
- Batched operations
- Edge cases
- Operator composition

Issues: #116, #117
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import Index, TimeseriesIndex

# All index reduction methods and their corresponding graph operators
INDEX_REDUCTION_METHODS = [
    (lambda idx: idx.sum(), graph.project_using_sum, "sum"),
    (lambda idx: idx.mean(), graph.project_using_mean, "mean"),
    (lambda idx: idx.min(), graph.project_using_min, "min"),
    (lambda idx: idx.max(), graph.project_using_max, "max"),
    (lambda idx: idx.std(), graph.project_using_std, "std"),
    (lambda idx: idx.var(), graph.project_using_var, "var"),
    (lambda idx: idx.median(), graph.project_using_median, "median"),
    (lambda idx: idx.prod(), graph.project_using_prod, "prod"),
    (lambda idx: idx.any(), graph.project_using_any, "any"),
    (lambda idx: idx.all(), graph.project_using_all, "all"),
    (lambda idx: idx.count_nonzero(), graph.project_using_count_nonzero, "count_nonzero"),
]


# Quantile method requires special q parameter
def _quantile_method(idx: Index, q: float) -> graph.project_using_quantile:
    """Call quantile method with q parameter."""
    return idx.quantile(q=q)  # type: ignore[return-value]


class TestIndexReductionMethodCreation:
    """Test that index reduction methods return correct node types."""

    @pytest.mark.parametrize("method,operator_class,name", INDEX_REDUCTION_METHODS)
    def test_method_returns_correct_operator(self, method, operator_class, name):
        """Test that reduction method returns correct operator node type."""
        idx = Index("test_index", 5.0)
        result = method(idx)
        assert isinstance(result, operator_class)

    def test_quantile_method_returns_quantile_operator(self):
        """Test that quantile method returns quantile operator."""
        idx = Index("test_index", 5.0)
        result = _quantile_method(idx, q=0.5)
        assert isinstance(result, graph.project_using_quantile)
        assert result.q == 0.5

    def test_quantile_method_different_q_values(self):
        """Test quantile method with different q values."""
        idx = Index("test_index", 5.0)
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _quantile_method(idx, q=q)
            assert result.q == q

    @pytest.mark.parametrize("method,operator_class,name", INDEX_REDUCTION_METHODS)
    def test_method_has_correct_axis(self, method, operator_class, name):
        """Test that reduction method creates operator with correct default axis."""
        idx = Index("test_index", 5.0)
        result = method(idx)
        assert result.axis == -1  # Default axis for index methods


class TestTimeseriesIndexReductionExecution:
    """Test execution of index reduction methods on TimeseriesIndex."""

    def test_sum_evaluation(self):
        """Test sum() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
        node = ts.sum()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        # Sum with keepdims over 1-D array returns shape (1,)
        assert np.isclose(np.sum(result), 6.0)

    def test_mean_evaluation(self):
        """Test mean() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([2.0, 4.0, 6.0]))
        node = ts.mean()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert np.isclose(np.sum(result), 4.0)

    def test_min_evaluation(self):
        """Test min() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([3.0, 1.0, 5.0, 2.0]))
        node = ts.min()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert np.isclose(np.min(result), 1.0)

    def test_max_evaluation(self):
        """Test max() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([3.0, 1.0, 5.0, 2.0]))
        node = ts.max()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert np.isclose(np.max(result), 5.0)

    def test_std_evaluation(self):
        """Test std() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        node = ts.std()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        expected_std = np.std(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), keepdims=True)
        assert np.allclose(result, expected_std)

    def test_median_evaluation(self):
        """Test median() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([1.0, 5.0, 3.0, 2.0, 4.0]))
        node = ts.median()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert np.isclose(np.median(result), 3.0)

    def test_prod_evaluation(self):
        """Test prod() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0, 4.0]))
        node = ts.prod()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert np.isclose(np.prod(result), 24.0)

    def test_any_evaluation(self):
        """Test any() over a 1-D timeseries of booleans."""
        ts = TimeseriesIndex("ts", np.array([False, False, True, False]))
        node = ts.any()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert bool(np.any(result)) is True

    def test_all_evaluation(self):
        """Test all() over a 1-D timeseries of booleans."""
        ts = TimeseriesIndex("ts", np.array([True, True, False, True]))
        node = ts.all()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert bool(np.all(result)) is False

    def test_count_nonzero_evaluation(self):
        """Test count_nonzero() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([1.0, 0.0, 3.0, 0.0, 5.0]))
        node = ts.count_nonzero()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert int(np.sum(result)) == 3

    def test_quantile_evaluation(self):
        """Test quantile() over a 1-D timeseries."""
        ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        node = ts.quantile(q=0.5)
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        expected = np.quantile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.5, keepdims=True)
        assert np.allclose(result, expected)


class TestBatchedReductionExecution:
    """Test execution of reduction methods on batched timeseries.

    The result is (size, 1) so it broadcasts correctly against timeseries
    of shape (T,) or (size, T).
    """

    def test_sum_batched(self):
        """Test sum() over a (size, T) batched timeseries."""
        ts = TimeseriesIndex("ts")
        node = ts.sum()
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({ts.node: values})
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == (2, 1)
        assert np.allclose(result, [[6.0], [15.0]])

    def test_mean_batched(self):
        """Test mean() over a (size, T) batched timeseries."""
        ts = TimeseriesIndex("ts")
        node = ts.mean()
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({ts.node: values})
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == (2, 1)
        assert np.allclose(result, [[2.0], [5.0]])

    def test_min_batched(self):
        """Test min() over a (size, T) batched timeseries."""
        ts = TimeseriesIndex("ts")
        node = ts.min()
        values = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        state = executor.State({ts.node: values})
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == (2, 1)
        assert np.allclose(result, [[1.0], [4.0]])

    def test_max_batched(self):
        """Test max() over a (size, T) batched timeseries."""
        ts = TimeseriesIndex("ts")
        node = ts.max()
        values = np.array([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        state = executor.State({ts.node: values})
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == (2, 1)
        assert np.allclose(result, [[3.0], [6.0]])

    def test_std_batched(self):
        """Test std() over a (size, T) batched timeseries."""
        ts = TimeseriesIndex("ts")
        node = ts.std()
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({ts.node: values})
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == (2, 1)
        expected = np.std(values, axis=-1, keepdims=True)
        assert np.allclose(result, expected)

    def test_var_batched(self):
        """Test var() over a (size, T) batched timeseries."""
        ts = TimeseriesIndex("ts")
        node = ts.var()
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({ts.node: values})
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == (2, 1)
        expected = np.var(values, axis=-1, keepdims=True)
        assert np.allclose(result, expected)

    def test_quantile_batched(self):
        """Test quantile() over a (size, T) batched timeseries."""
        ts = TimeseriesIndex("ts")
        node = ts.quantile(q=0.5)
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({ts.node: values})
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == (2, 1)
        expected = np.quantile(values, 0.5, axis=-1, keepdims=True)
        assert np.allclose(result, expected)


class TestIndexReductionWithPlaceholders:
    """Test index reduction methods with placeholder values."""

    def test_sum_with_placeholder(self):
        """Test sum() with placeholder node."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        sum_node = idx.sum()
        plan = linearize.forest(sum_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.sum(x_val, axis=-1, keepdims=True)
        assert np.array_equal(state.values[sum_node], expected)

    def test_mean_with_placeholder(self):
        """Test mean() with placeholder node."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        mean_node = idx.mean()
        plan = linearize.forest(mean_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.mean(x_val, axis=-1, keepdims=True)
        assert np.allclose(state.values[mean_node], expected)

    def test_min_with_placeholder(self):
        """Test min() with placeholder node."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        min_node = idx.min()
        plan = linearize.forest(min_node)

        x_val = np.array([[3.0, 1.0, 5.0], [6.0, 4.0, 2.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(x_val, axis=-1, keepdims=True)
        assert np.array_equal(state.values[min_node], expected)

    def test_max_with_placeholder(self):
        """Test max() with placeholder node."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        max_node = idx.max()
        plan = linearize.forest(max_node)

        x_val = np.array([[3.0, 1.0, 5.0], [6.0, 4.0, 2.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.max(x_val, axis=-1, keepdims=True)
        assert np.array_equal(state.values[max_node], expected)

    def test_quantile_with_placeholder(self):
        """Test quantile() with placeholder node."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        quantile_node = idx.quantile(q=0.5)
        plan = linearize.forest(quantile_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(x_val, 0.5, axis=-1, keepdims=True)
        assert np.allclose(state.values[quantile_node], expected)


class TestIndexReductionComposition:
    """Test composition of index reduction methods."""

    def test_min_of_min(self):
        """Test composing min methods."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        min1 = idx.min()
        min2 = graph.project_using_min(min1, axis=0)
        plan = linearize.forest(min2)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(np.min(x_val, axis=-1, keepdims=True), axis=0, keepdims=True)
        assert np.array_equal(state.values[min2], expected)

    def test_sum_of_max(self):
        """Test composing sum of max."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        max_node = idx.max()
        sum_node = graph.project_using_sum(max_node, axis=0)
        plan = linearize.forest(sum_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.sum(np.max(x_val, axis=-1, keepdims=True), axis=0, keepdims=True)
        assert np.allclose(state.values[sum_node], expected)

    def test_mean_of_std(self):
        """Test composing mean of std."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        std_node = idx.std()
        mean_node = graph.project_using_mean(std_node, axis=0)
        plan = linearize.forest(mean_node)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.mean(np.std(x_val, axis=-1, keepdims=True), axis=0, keepdims=True)
        assert np.allclose(state.values[mean_node], expected)

    def test_quantile_of_quantile(self):
        """Test composing quantile of quantile."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        q1_node = idx.quantile(q=0.5)
        q2_node = graph.project_using_quantile(q1_node, axis=0, q=0.75)
        plan = linearize.forest(q2_node)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        intermediate = np.quantile(x_val, 0.5, axis=-1, keepdims=True)
        expected = np.quantile(intermediate, 0.75, axis=0, keepdims=True)
        assert np.allclose(state.values[q2_node], expected)


class TestIndexReductionEdgeCases:
    """Test edge cases for index reduction methods."""

    def test_sum_empty_batched_dimension(self):
        """Test sum with minimal batched dimension."""
        ts = TimeseriesIndex("ts")
        node = ts.sum()
        values = np.array([[1.0, 2.0, 3.0]])
        state = executor.State({ts.node: values})
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == (1, 1)

    def test_quantile_all_same_values(self):
        """Test quantile with all identical values."""
        ts = TimeseriesIndex("ts", np.array([5.0, 5.0, 5.0, 5.0]))
        node = ts.quantile(q=0.5)
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert np.isclose(result, 5.0)

    def test_count_nonzero_all_zeros(self):
        """Test count_nonzero with all zeros."""
        ts = TimeseriesIndex("ts", np.array([0.0, 0.0, 0.0, 0.0]))
        node = ts.count_nonzero()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert np.isclose(result, 0.0)

    def test_prod_with_negative_numbers(self):
        """Test prod with negative numbers."""
        ts = TimeseriesIndex("ts", np.array([-1.0, -2.0, -3.0]))
        node = ts.prod()
        plan = linearize.forest(node)
        state = executor.State({})
        executor.evaluate_nodes(state, *plan)
        result = state.values[node]
        assert np.isclose(result, -6.0)
