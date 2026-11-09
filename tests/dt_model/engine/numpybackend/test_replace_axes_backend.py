"""Evaluating replace_axes: the value's data moves from each source dimension to its target."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model import axes
from civic_digital_twins.dt_model.axes import DOMAIN, Axis
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor, kernels, numpy_ast

A = Axis("a", DOMAIN)
B = Axis("b", DOMAIN)
C = Axis("c", DOMAIN)
DOMAIN_AXES = (A, B, C)

# Varies along a (2) and b (3); size 1 along c.
VALUE = np.arange(6.0).reshape(2, 3, 1)


class TestKernel:
    """kernels.replace_axes on its own."""

    def test_source_moves_to_an_unused_target(self):
        """The data along a goes to c; a is left with size 1."""
        out = kernels.replace_axes(VALUE, (-3,), (-1,), 3)
        assert out.shape == (1, 3, 2)
        np.testing.assert_array_equal(out[0, :, :], VALUE[:, :, 0].T)

    def test_swap(self):
        """Two axes exchange their data."""
        out = kernels.replace_axes(VALUE, (-3, -2), (-2, -3), 3)
        np.testing.assert_array_equal(out, np.swapaxes(VALUE, 0, 1))

    def test_chain(self):
        """{a: b, b: c}: a's data goes to b, b's to c, and a is left with size 1."""
        out = kernels.replace_axes(VALUE, (-3, -2), (-2, -1), 3)
        assert out.shape == (1, 2, 3)
        np.testing.assert_array_equal(out[0], VALUE[:, :, 0])

    def test_leading_dimensions_are_kept(self):
        """PARAMETER and ENSEMBLE dimensions in front stay in place."""
        value = np.arange(4 * 6.0).reshape(4, 2, 3, 1)
        out = kernels.replace_axes(value, (-3,), (-1,), 3)
        assert out.shape == (4, 1, 3, 2)

    def test_result_is_a_view(self):
        """Nothing is copied."""
        assert np.shares_memory(kernels.replace_axes(VALUE, (-3,), (-1,), 3), VALUE)

    def test_scalar_is_unchanged(self):
        """A 0-d value does not vary along any axis."""
        assert kernels.replace_axes(np.asarray(5.0), (-3,), (-1,), 3).shape == ()

    def test_partial_rank_raises(self):
        """A value with some DOMAIN dimensions only cannot be matched to axes."""
        with pytest.raises(ValueError, match=r"fewer dimensions than the 3 DOMAIN axes"):
            kernels.replace_axes(np.zeros(3), (-3,), (-1,), 3)


def _evaluate(node, value):
    f = graph.placeholder("f")
    result = node(f.broadcast(A, B))
    state = executor.State({f: value}, domain_axes=DOMAIN_AXES, domain_extents=(2, 3, 2))
    executor.evaluate_nodes(state, *linearize.forest(result))
    return f, result, state


class TestExecutor:
    """The executor evaluates replace_axes nodes through the kernel."""

    def test_replace_then_operate_along_the_target(self):
        """An operator along the target sees the source's data."""
        _, result, state = _evaluate(lambda v: v.replace_axis(A, C).sum(C), VALUE)
        np.testing.assert_array_equal(state.values[result][0, :, 0], VALUE.sum(axis=0)[:, 0])

    def test_partial_rank_raises_naming_the_node(self):
        """The executor reports a malformed operand with the node and the DOMAIN axes."""
        with pytest.raises(executor.UnsupportedOperation, match=r"replace_axes .*fewer dimensions.*DOMAIN axes are"):
            _evaluate(lambda v: v.replace_axis(A, C), np.zeros(3))

    def test_generated_code_reproduces_the_evaluation(self):
        """The printed code calls the same kernel and gives the executor's result."""
        f, result, state = _evaluate(lambda v: v.replace_axes({A: B, B: C}), VALUE)
        namespace = {**vars(numpy_ast), **vars(axes), f"n{f.id}": VALUE}
        for node in linearize.forest(result):
            if node is not f:
                code = numpy_ast.graph_node_to_numpy_code(node, domain_axes=DOMAIN_AXES, domain_extents=(2, 3, 2))
                exec(code, namespace)
        assert "_replace_axes(" in numpy_ast.graph_node_to_numpy_code(result, domain_axes=DOMAIN_AXES)
        np.testing.assert_array_equal(namespace[f"n{result.id}"], state.values[result])
