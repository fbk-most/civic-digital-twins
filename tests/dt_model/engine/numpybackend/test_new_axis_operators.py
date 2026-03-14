"""Tests for new axis reduction operators (issues #116 and #117)."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np

from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor


class TestAxisOperatorsBasic:
    """Test basic creation of new axis operators."""

    def test_project_using_min_creation(self):
        """Test creating a min projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_min(x, axis=0)
        assert result.node is x
        assert result.axis == 0
        assert isinstance(result, graph.project_using_min)

    def test_project_using_max_creation(self):
        """Test creating a max projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_max(x, axis=1)
        assert result.node is x
        assert result.axis == 1
        assert isinstance(result, graph.project_using_max)

    def test_project_using_std_creation(self):
        """Test creating a std projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_std(x, axis=0)
        assert result.node is x
        assert result.axis == 0
        assert isinstance(result, graph.project_using_std)

    def test_project_using_var_creation(self):
        """Test creating a var projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_var(x, axis=1)
        assert result.node is x
        assert result.axis == 1
        assert isinstance(result, graph.project_using_var)

    def test_project_using_median_creation(self):
        """Test creating a median projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_median(x, axis=0)
        assert result.node is x
        assert result.axis == 0
        assert isinstance(result, graph.project_using_median)

    def test_project_using_prod_creation(self):
        """Test creating a prod projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_prod(x, axis=1)
        assert result.node is x
        assert result.axis == 1
        assert isinstance(result, graph.project_using_prod)

    def test_project_using_any_creation(self):
        """Test creating an any projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_any(x, axis=0)
        assert result.node is x
        assert result.axis == 0
        assert isinstance(result, graph.project_using_any)

    def test_project_using_all_creation(self):
        """Test creating an all projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_all(x, axis=1)
        assert result.node is x
        assert result.axis == 1
        assert isinstance(result, graph.project_using_all)

    def test_project_using_count_nonzero_creation(self):
        """Test creating a count_nonzero projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_count_nonzero(x, axis=0)
        assert result.node is x
        assert result.axis == 0
        assert isinstance(result, graph.project_using_count_nonzero)

    def test_project_using_quantile_creation(self):
        """Test creating a quantile projection node."""
        x = graph.placeholder("x")
        result = graph.project_using_quantile(x, axis=0, q=0.5)
        assert result.node is x
        assert result.axis == 0
        assert result.q == 0.5
        assert isinstance(result, graph.project_using_quantile)

    def test_project_using_quantile_different_q_values(self):
        """Test quantile with different q values."""
        x = graph.placeholder("x")
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = graph.project_using_quantile(x, axis=0, q=q)
            assert result.q == q


class TestAxisOperatorsRepr:
    """Test string representations of new axis operators."""

    def test_min_repr(self):
        """Test min repr."""
        x = graph.placeholder("x")
        result = graph.project_using_min(x, axis=0, name="test_min")
        repr_str = repr(result)
        assert "project_using_min" in repr_str
        assert "axis=0" in repr_str

    def test_max_repr(self):
        """Test max repr."""
        x = graph.placeholder("x")
        result = graph.project_using_max(x, axis=1, name="test_max")
        repr_str = repr(result)
        assert "project_using_max" in repr_str
        assert "axis=1" in repr_str

    def test_std_repr(self):
        """Test std repr."""
        x = graph.placeholder("x")
        result = graph.project_using_std(x, axis=0, name="test_std")
        repr_str = repr(result)
        assert "project_using_std" in repr_str
        assert "axis=0" in repr_str

    def test_quantile_repr(self):
        """Test quantile repr."""
        x = graph.placeholder("x")
        result = graph.project_using_quantile(x, axis=0, q=0.75, name="test_quantile")
        repr_str = repr(result)
        assert "project_using_quantile" in repr_str
        assert "q=0.75" in repr_str


class TestAxisOperatorsExecution:
    """Test execution of new axis operators."""

    def test_min_execution(self):
        """Test min execution."""
        x = graph.placeholder("x")
        result = graph.project_using_min(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_max_execution(self):
        """Test max execution."""
        x = graph.placeholder("x")
        result = graph.project_using_max(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.max(x_val, axis=1, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_std_execution(self):
        """Test std execution."""
        x = graph.placeholder("x")
        result = graph.project_using_std(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.std(x_val, axis=0, keepdims=True)
        assert np.allclose(state.values[result], expected)

    def test_var_execution(self):
        """Test var execution."""
        x = graph.placeholder("x")
        result = graph.project_using_var(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.var(x_val, axis=1, keepdims=True)
        assert np.allclose(state.values[result], expected)

    def test_median_execution(self):
        """Test median execution."""
        x = graph.placeholder("x")
        result = graph.project_using_median(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.median(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_prod_execution(self):
        """Test prod execution."""
        x = graph.placeholder("x")
        result = graph.project_using_prod(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.prod(x_val, axis=1, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_any_execution(self):
        """Test any execution."""
        x = graph.placeholder("x")
        result = graph.project_using_any(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[True, False, False], [False, False, False]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.any(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_all_execution(self):
        """Test all execution."""
        x = graph.placeholder("x")
        result = graph.project_using_all(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[True, True, False], [True, True, True]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.all(x_val, axis=1, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_count_nonzero_execution(self):
        """Test count_nonzero execution."""
        x = graph.placeholder("x")
        result = graph.project_using_count_nonzero(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 0.0, 3.0], [0.0, 4.0, 0.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.count_nonzero(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_quantile_execution_median(self):
        """Test quantile execution for median."""
        x = graph.placeholder("x")
        result = graph.project_using_quantile(x, axis=0, q=0.5)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(x_val, 0.5, axis=0, keepdims=True)
        assert np.allclose(state.values[result], expected)

    def test_quantile_execution_q25(self):
        """Test quantile execution for 25th percentile."""
        x = graph.placeholder("x")
        result = graph.project_using_quantile(x, axis=1, q=0.25)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(x_val, 0.25, axis=1, keepdims=True)
        assert np.allclose(state.values[result], expected)

    def test_quantile_execution_q75(self):
        """Test quantile execution for 75th percentile."""
        x = graph.placeholder("x")
        result = graph.project_using_quantile(x, axis=0, q=0.75)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.quantile(x_val, 0.75, axis=0, keepdims=True)
        assert np.allclose(state.values[result], expected)


class TestAxisOperatorsKeepdims:
    """Test that all axis operators preserve keepdims semantics."""

    def test_min_keepdims(self):
        """Test min preserves keepdims."""
        x = graph.placeholder("x")
        result = graph.project_using_min(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        # Should reduce axis 0, resulting in shape (1, 3)
        assert state.values[result].shape == (1, 3)

    def test_max_keepdims(self):
        """Test max preserves keepdims."""
        x = graph.placeholder("x")
        result = graph.project_using_max(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        # Should reduce axis 1, resulting in shape (2, 1)
        assert state.values[result].shape == (2, 1)

    def test_std_keepdims(self):
        """Test std preserves keepdims."""
        x = graph.placeholder("x")
        result = graph.project_using_std(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        # Should reduce axis 0, resulting in shape (1, 3)
        assert state.values[result].shape == (1, 3)

    def test_quantile_keepdims(self):
        """Test quantile preserves keepdims."""
        x = graph.placeholder("x")
        result = graph.project_using_quantile(x, axis=1, q=0.5)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        # Should reduce axis 1, resulting in shape (2, 1)
        assert state.values[result].shape == (2, 1)


class TestAxisOperatorsCombinations:
    """Test combinations of axis operators."""

    def test_nested_min_max(self):
        """Test composing min and max."""
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
        quantile_result = graph.project_using_quantile(mean_result, axis=1, q=0.5)
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


class TestAxisOperatorsEdgeCases:
    """Test edge cases for axis operators."""

    def test_min_single_element(self):
        """Test min on single element."""
        x = graph.placeholder("x")
        result = graph.project_using_min(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[42.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.min(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_max_negative_numbers(self):
        """Test max with negative numbers."""
        x = graph.placeholder("x")
        result = graph.project_using_max(x, axis=1)
        plan = linearize.forest(result)

        x_val = np.array([[-1.0, -5.0, -3.0], [-2.0, -4.0, -6.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.max(x_val, axis=1, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_std_uniform_values(self):
        """Test std with uniform values (should be zero)."""
        x = graph.placeholder("x")
        result = graph.project_using_std(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.std(x_val, axis=0, keepdims=True)
        assert np.allclose(state.values[result], expected)

    def test_prod_with_zeros(self):
        """Test prod with zeros."""
        x = graph.placeholder("x")
        result = graph.project_using_prod(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[1.0, 0.0, 3.0], [2.0, 4.0, 5.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.prod(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_count_nonzero_all_zeros(self):
        """Test count_nonzero with all zeros."""
        x = graph.placeholder("x")
        result = graph.project_using_count_nonzero(x, axis=0)
        plan = linearize.forest(result)

        x_val = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)

        expected = np.count_nonzero(x_val, axis=0, keepdims=True)
        assert np.array_equal(state.values[result], expected)

    def test_quantile_extremes(self):
        """Test quantile with extreme q values."""
        x = graph.placeholder("x")
        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # q=0
        result_q0 = graph.project_using_quantile(x, axis=0, q=0.0)
        plan = linearize.forest(result_q0)
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)
        expected_q0 = np.quantile(x_val, 0.0, axis=0, keepdims=True)
        assert np.allclose(state.values[result_q0], expected_q0)

        # q=1
        result_q1 = graph.project_using_quantile(x, axis=0, q=1.0)
        plan = linearize.forest(result_q1)
        state = executor.State({x: x_val})
        executor.evaluate_nodes(state, *plan)
        expected_q1 = np.quantile(x_val, 1.0, axis=0, keepdims=True)
        assert np.allclose(state.values[result_q1], expected_q1)
