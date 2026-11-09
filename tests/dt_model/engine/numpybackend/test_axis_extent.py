"""Axis-aware operators on operands stored at size 1 along their axis.

A value that does not vary along a DOMAIN axis is stored with size 1 there.
Every operator that works along an axis must give the same result as on the
value materialised at the axis's full extent (``State.domain_extents``).
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model import axes
from civic_digital_twins.dt_model.axes import DOMAIN, Axis, Constant, Linear, Nearest, Neumann, Wrap
from civic_digital_twins.dt_model.engine import compileflags
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor, kernels, numpy_ast

X = Axis("x", DOMAIN)
Y = Axis("y", DOMAIN)
EXTENTS = (4, 3)  # x, y

# A value varying along y only: stored (1, 3), materialised (4, 3).
COMPRESSED = np.array([[1.0, 0.0, 3.0]])
FULL = np.broadcast_to(COMPRESSED, EXTENTS)


def _evaluate(build, value, *, extents=EXTENTS):
    """Build a node over a placeholder holding *value* and evaluate it with *extents*."""
    f = graph.placeholder("f")
    result = build(f)
    state = executor.State({f: value}, domain_axes=(X, Y), domain_extents=extents)
    executor.evaluate_nodes(state, *linearize.forest(result))
    return state.values[result]


OPERATORS = [
    pytest.param(lambda f: graph.project_using_sum(f, X), id="sum"),
    pytest.param(lambda f: graph.project_using_mean(f, X), id="mean"),
    pytest.param(lambda f: graph.project_using_min(f, X), id="min"),
    pytest.param(lambda f: graph.project_using_max(f, X), id="max"),
    pytest.param(lambda f: graph.project_using_std(f, X), id="std"),
    pytest.param(lambda f: graph.project_using_var(f, X), id="var"),
    pytest.param(lambda f: graph.project_using_median(f, X), id="median"),
    pytest.param(lambda f: graph.project_using_prod(f, X), id="prod"),
    pytest.param(lambda f: graph.project_using_any(f, X), id="any"),
    pytest.param(lambda f: graph.project_using_all(f, X), id="all"),
    pytest.param(lambda f: graph.project_using_count_nonzero(f, X), id="count_nonzero"),
    pytest.param(lambda f: graph.project_using_quantile(f, X, q=0.25), id="quantile"),
    pytest.param(lambda f: graph.shift(f, X, periods=1, fill_value=-1.0), id="shift"),
    pytest.param(lambda f: graph.shift(f, X, periods=-2), id="shift-negative"),
    pytest.param(lambda f: graph.roll(f, X, periods=1), id="roll"),
    pytest.param(lambda f: graph.cumulative(f, X), id="cumulative"),
    pytest.param(lambda f: f.diff(X), id="diff"),
    pytest.param(lambda f: graph.gradient(f, X, spacing=0.5, boundary=Constant(2.0)), id="gradient-constant"),
    pytest.param(lambda f: graph.gradient(f, X, boundary=Neumann(1.0)), id="gradient-neumann"),
    pytest.param(lambda f: graph.gradient(f, X, boundary=Nearest()), id="gradient-nearest"),
    pytest.param(lambda f: graph.gradient(f, X, boundary=Wrap()), id="gradient-wrap"),
    pytest.param(lambda f: graph.gradient(f, X, boundary=Linear()), id="gradient-linear"),
    pytest.param(lambda f: graph.laplacian(f, (X,), (1.0,), (Constant(0.0),)), id="laplacian-constant"),
    pytest.param(lambda f: graph.laplacian(f, (X, Y), (1.0, 2.0), (Constant(1.0), Wrap())), id="laplacian-two-axes"),
]


@pytest.mark.parametrize("build", OPERATORS)
def test_operator_on_a_compressed_operand_matches_the_materialised_operand(build):
    """Along its axis, every operator sees the full extent, whatever size the operand is stored at."""
    np.testing.assert_array_equal(_evaluate(build, COMPRESSED), _evaluate(build, FULL))


@pytest.mark.parametrize("build", OPERATORS)
def test_operator_axis_is_expanded_only_along_the_operator_axes(build):
    """Expansion happens along the operator's axes only: a size-1 dimension elsewhere stays size 1."""
    # Varies along x only, stored (4, 1): the y dimension is not the operator's, except for laplacian over (x, y).
    along_x = np.array([[1.0], [0.0], [3.0], [5.0]])
    result = _evaluate(build, along_x)
    expected = _evaluate(build, np.broadcast_to(along_x, EXTENTS))
    np.testing.assert_array_equal(np.broadcast_to(result, expected.shape), expected)


def test_reduction_counts_every_position_of_a_constant_value():
    """The sum of a constant over an axis of length 4 is four times the constant (#296)."""
    result = _evaluate(lambda f: graph.project_using_sum(f, X), np.full((1, 1), 10.0))
    np.testing.assert_array_equal(result, [[40.0]])


def test_scalar_operand_is_constant_along_every_axis():
    """A 0-d constant carries no dimensions; it is treated as size 1 along every DOMAIN axis."""
    c = graph.constant(10.0)
    result = graph.cumulative(c, Y)
    state = executor.State({}, domain_axes=(X, Y), domain_extents=EXTENTS)
    executor.evaluate_nodes(state, *linearize.forest(result))
    np.testing.assert_array_equal(state.values[result], [[10.0, 20.0, 30.0]])


def test_operand_already_at_full_extent_is_unchanged():
    """An operand already at the extent is passed through as is."""
    full = np.arange(12.0).reshape(EXTENTS)
    result = _evaluate(lambda f: graph.cumulative(f, X), full)
    np.testing.assert_array_equal(result, np.cumsum(full, axis=0))


def test_without_extents_operands_are_not_expanded():
    """A state resolving no extents evaluates operands at their stored size."""
    result = _evaluate(lambda f: graph.project_using_sum(f, X), COMPRESSED, extents=())
    np.testing.assert_array_equal(result, COMPRESSED)


def test_operand_with_some_domain_dimensions_only_is_rejected():
    """A value with fewer dimensions than the DOMAIN block cannot be matched to axes, so it is not guessed."""
    with pytest.raises(executor.UnsupportedOperation, match=r"the operand has shape \(3,\), fewer dimensions"):
        _evaluate(lambda f: graph.project_using_sum(f, X), np.zeros(3))


@pytest.mark.parametrize("build", OPERATORS)
def test_generated_code_reproduces_the_evaluation(build):
    """The NumPy code printed for every node, run on the stored operand, gives the executor's result."""
    f = graph.placeholder("f")
    result = build(f)
    namespace = {**vars(numpy_ast), **vars(axes), f"n{f.id}": COMPRESSED}
    for node in linearize.forest(result):
        if node is not f:
            exec(numpy_ast.graph_node_to_numpy_code(node, domain_axes=(X, Y), domain_extents=EXTENTS), namespace)
    np.testing.assert_array_equal(namespace[f"n{result.id}"], _evaluate(build, COMPRESSED))


def test_trace_prints_the_expansion(capsys):
    """The trace shows the operand expansion the executor applies."""
    f = graph.placeholder("f")
    result = graph.project_using_sum(f, X)
    state = executor.State(
        {f: np.full((1, 1), 10.0)}, domain_axes=(X, Y), domain_extents=EXTENTS, flags=compileflags.TRACE
    )
    executor.evaluate_nodes(state, *linearize.forest(result))
    out = capsys.readouterr().out
    assert f"_expand_to_extent(n{f.id}, axes=(-2,), domain_extents=(4, 3))" in out


def test_expand_kernel_rejects_partial_domain_rank():
    """The kernel itself refuses to guess which axes a partial-rank value carries."""
    with pytest.raises(ValueError, match="fewer dimensions than the 2 DOMAIN axes"):
        kernels.expand_to_extent(np.zeros(3), (-1,), EXTENTS)
