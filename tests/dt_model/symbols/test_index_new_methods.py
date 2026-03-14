"""Tests for new GenericIndex reduction methods.

Tests for issue #116 (min, max, std, var, median, prod, any, all, count_nonzero)
and issue #117 (quantile).
"""

import numpy as np
import pytest

from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import Index, TimeseriesIndex


class TestGenericIndexNewMethods:
    """Test new reduction methods added to GenericIndex."""

    def test_index_min_method(self):
        """Test Index.min() method."""
        idx = Index("test_index", 5.0)
        result = idx.min(axis=0)
        assert isinstance(result, graph.project_using_min)
        assert result.axis == 0

    def test_index_max_method(self):
        """Test Index.max() method."""
        idx = Index("test_index", 5.0)
        result = idx.max(axis=1)
        assert isinstance(result, graph.project_using_max)
        assert result.axis == 1

    def test_index_std_method(self):
        """Test Index.std() method."""
        idx = Index("test_index", 5.0)
        result = idx.std(axis=0)
        assert isinstance(result, graph.project_using_std)
        assert result.axis == 0

    def test_index_var_method(self):
        """Test Index.var() method."""
        idx = Index("test_index", 5.0)
        result = idx.var(axis=1)
        assert isinstance(result, graph.project_using_var)
        assert result.axis == 1

    def test_index_median_method(self):
        """Test Index.median() method."""
        idx = Index("test_index", 5.0)
        result = idx.median(axis=0)
        assert isinstance(result, graph.project_using_median)
        assert result.axis == 0

    def test_index_prod_method(self):
        """Test Index.prod() method."""
        idx = Index("test_index", 5.0)
        result = idx.prod(axis=1)
        assert isinstance(result, graph.project_using_prod)
        assert result.axis == 1

    def test_index_any_method(self):
        """Test Index.any() method."""
        idx = Index("test_index", True)
        result = idx.any(axis=0)
        assert isinstance(result, graph.project_using_any)
        assert result.axis == 0

    def test_index_all_method(self):
        """Test Index.all() method."""
        idx = Index("test_index", True)
        result = idx.all(axis=1)
        assert isinstance(result, graph.project_using_all)
        assert result.axis == 1

    def test_index_count_nonzero_method(self):
        """Test Index.count_nonzero() method."""
        idx = Index("test_index", 1.0)
        result = idx.count_nonzero(axis=0)
        assert isinstance(result, graph.project_using_count_nonzero)
        assert result.axis == 0

    def test_index_quantile_method(self):
        """Test Index.quantile() method."""
        idx = Index("test_index", 5.0)
        result = idx.quantile(q=0.5, axis=0)
        assert isinstance(result, graph.project_using_quantile)
        assert result.axis == 0
        assert result.q == 0.5

    def test_timeseries_index_min_method(self):
        """Test TimeseriesIndex.min() method."""
        ts_idx = TimeseriesIndex("test_ts", np.array([1.0, 2.0, 3.0]))
        result = ts_idx.min(axis=-1)
        assert isinstance(result, graph.project_using_min)
        assert result.axis == -1

    def test_timeseries_index_quantile_method(self):
        """Test TimeseriesIndex.quantile() method."""
        ts_idx = TimeseriesIndex("test_ts", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        result = ts_idx.quantile(q=0.75, axis=-1)
        assert isinstance(result, graph.project_using_quantile)
        assert result.axis == -1
        assert result.q == 0.75

    def test_default_axis_is_negative_one(self):
        """Test that default axis is -1 for all methods."""
        idx = Index("test", 5.0)

        assert idx.min().axis == -1
        assert idx.max().axis == -1
        assert idx.std().axis == -1
        assert idx.var().axis == -1
        assert idx.median().axis == -1
        assert idx.prod().axis == -1
        assert idx.any().axis == -1
        assert idx.all().axis == -1
        assert idx.count_nonzero().axis == -1
        assert idx.quantile(q=0.5).axis == -1


class TestGenericIndexMethodsExecution:
    """Test execution of new GenericIndex reduction methods."""

    def test_index_min_execution(self):
        """Test executing min through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        min_node = idx.min(axis=0)
        plan = linearize.forest(min_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[min_node], expected)

    def test_index_max_execution(self):
        """Test executing max through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        max_node = idx.max(axis=1)
        plan = linearize.forest(max_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.max(x_val, axis=1, keepdims=True)
        assert np.array_equal(state.values[max_node], expected)

    def test_index_std_execution(self):
        """Test executing std through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        std_node = idx.std(axis=0)
        plan = linearize.forest(std_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.std(x_val, axis=0, keepdims=True)
        assert np.allclose(state.values[std_node], expected)

    def test_index_var_execution(self):
        """Test executing var through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        var_node = idx.var(axis=1)
        plan = linearize.forest(var_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.var(x_val, axis=1, keepdims=True)
        assert np.allclose(state.values[var_node], expected)

    def test_index_median_execution(self):
        """Test executing median through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        median_node = idx.median(axis=0)
        plan = linearize.forest(median_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.median(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[median_node], expected)

    def test_index_prod_execution(self):
        """Test executing prod through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        prod_node = idx.prod(axis=1)
        plan = linearize.forest(prod_node)

        x_val = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.prod(x_val, axis=1, keepdims=True)
        assert np.array_equal(state.values[prod_node], expected)

    def test_index_any_execution(self):
        """Test executing any through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        any_node = idx.any(axis=0)
        plan = linearize.forest(any_node)

        x_val = np.array([[True, False, False], [False, False, False]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.any(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[any_node], expected)

    def test_index_all_execution(self):
        """Test executing all through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        all_node = idx.all(axis=1)
        plan = linearize.forest(all_node)

        x_val = np.array([[True, True, False], [True, True, True]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.all(x_val, axis=1, keepdims=True)
        assert np.array_equal(state.values[all_node], expected)

    def test_index_count_nonzero_execution(self):
        """Test executing count_nonzero through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        count_node = idx.count_nonzero(axis=0)
        plan = linearize.forest(count_node)

        x_val = np.array([[1.0, 0.0, 3.0], [0.0, 4.0, 0.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.count_nonzero(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[count_node], expected)

    def test_index_quantile_execution(self):
        """Test executing quantile through Index method."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        quantile_node = idx.quantile(q=0.5, axis=0)
        plan = linearize.forest(quantile_node)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(x_val, 0.5, axis=0, keepdims=True)
        assert np.allclose(state.values[quantile_node], expected)

    def test_timeseries_index_min_execution(self):
        """Test executing min through TimeseriesIndex method."""
        ts_idx = TimeseriesIndex("x", np.array([1.0, 5.0, 3.0, 2.0, 4.0]))
        min_node = ts_idx.min(axis=-1)
        plan = linearize.forest(min_node)

        state = executor.State({})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(np.array([1.0, 5.0, 3.0, 2.0, 4.0]), axis=-1, keepdims=True)
        assert np.array_equal(state.values[min_node], expected)

    def test_timeseries_index_quantile_execution(self):
        """Test executing quantile through TimeseriesIndex method."""
        ts_idx = TimeseriesIndex("x", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        quantile_node = ts_idx.quantile(q=0.75, axis=-1)
        plan = linearize.forest(quantile_node)

        state = executor.State({})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0.75, axis=-1, keepdims=True)
        assert np.allclose(state.values[quantile_node], expected)


class TestGenericIndexMethodsComposition:
    """Test composition of new GenericIndex methods."""

    def test_min_of_min(self):
        """Test composing min methods."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        min1 = idx.min(axis=1)
        min2 = graph.project_using_min(min1, axis=0)
        plan = linearize.forest(min2)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x_placeholder: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(np.min(x_val, axis=1, keepdims=True), axis=0, keepdims=True)
        assert np.array_equal(state.values[min2], expected)

    def test_quantile_with_default_axis(self):
        """Test quantile with default axis=-1."""
        ts_idx = TimeseriesIndex("x", np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        quantile_node = ts_idx.quantile(q=0.25)
        plan = linearize.forest(quantile_node)

        state = executor.State({})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), 0.25, axis=-1, keepdims=True)
        assert np.allclose(state.values[quantile_node], expected)

    def test_all_methods_preserve_keepdims(self):
        """Test that all Index methods preserve keepdims semantics."""
        x_placeholder = graph.placeholder("x")
        idx = Index("x", x_placeholder)
        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        methods_and_ops = [
            (idx.min(axis=0), graph.project_using_min),
            (idx.max(axis=0), graph.project_using_max),
            (idx.std(axis=0), graph.project_using_std),
            (idx.var(axis=0), graph.project_using_var),
            (idx.median(axis=0), graph.project_using_median),
            (idx.prod(axis=0), graph.project_using_prod),
        ]

        for node, op_type in methods_and_ops:
            assert isinstance(node, op_type)
            plan = linearize.forest(node)
            state = executor.State({x_placeholder: x_val})
            executor.evaluate_nodes(state, *plan)
            result = state.values[node]
            # Should reduce axis 0, resulting in shape (1, 3)
            assert result.shape == (1, 3), f"Failed for {op_type.__name__}"
