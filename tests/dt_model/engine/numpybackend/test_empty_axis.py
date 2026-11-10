"""Operators over a DOMAIN axis of length 0.

An empty axis (e.g. an entity axis with no entities) is a normal case:
reductions with an identity give it, the others follow NumPy or raise, and
shape-preserving operators give an empty result.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import DOMAIN, Axis, Constant, Linear, Nearest, Neumann, Wrap
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor, kernels

X = Axis("x", DOMAIN)
Y = Axis("y", DOMAIN)
EXTENTS = (0, 3)  # x is empty


def _evaluate(build, value):
    f = graph.placeholder("f")
    result = build(f)
    state = executor.State({f: value}, domain_axes=(X, Y), domain_extents=EXTENTS)
    executor.evaluate_nodes(state, *linearize.forest(result))
    return state.values[result]


EMPTY = np.zeros((0, 3))
CONSTANT = np.full((1, 3), 5.0)  # size 1 along x: constant along it, over no positions


@pytest.mark.parametrize("value", [EMPTY, CONSTANT], ids=["empty", "constant"])
@pytest.mark.parametrize(
    ("build", "identity"),
    [
        pytest.param(lambda f: graph.project_using_sum(f, X), 0.0, id="sum"),
        pytest.param(lambda f: graph.project_using_prod(f, X), 1.0, id="prod"),
        pytest.param(lambda f: graph.project_using_any(f, X), False, id="any"),
        pytest.param(lambda f: graph.project_using_all(f, X), True, id="all"),
        pytest.param(lambda f: graph.project_using_count_nonzero(f, X), 0, id="count_nonzero"),
    ],
)
def test_reduction_with_an_identity_gives_it(build, identity, value):
    """Sum, prod, any, all and count_nonzero over no values give their identity, keeping the other axes."""
    result = _evaluate(build, value)
    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result, np.full((1, 3), identity))


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda f: graph.project_using_mean(f, X), id="mean"),
        pytest.param(lambda f: graph.project_using_std(f, X), id="std"),
        pytest.param(lambda f: graph.project_using_var(f, X), id="var"),
        pytest.param(lambda f: graph.project_using_median(f, X), id="median"),
    ],
)
def test_statistic_of_no_values_is_nan(build):
    """Mean, std, var and median follow NumPy: NaN, with NumPy's warning."""
    with pytest.warns(RuntimeWarning):
        result = _evaluate(build, EMPTY)
    assert result.shape == (1, 3)
    assert np.isnan(result).all()


@pytest.mark.parametrize(
    ("build", "what"),
    [
        pytest.param(lambda f: graph.project_using_min(f, X), "the minimum", id="min"),
        pytest.param(lambda f: graph.project_using_max(f, X), "the maximum", id="max"),
        pytest.param(lambda f: graph.project_using_quantile(f, X, q=0.5), "a quantile", id="quantile"),
    ],
)
def test_reduction_without_an_identity_raises(build, what):
    """Min, max and quantile of no values are undefined; the error names the axis."""
    with pytest.raises(ValueError, match=rf"axis 'x' has length 0, and {what} of no values is undefined"):
        _evaluate(build, EMPTY)


def test_reduction_over_another_axis_keeps_the_empty_axis():
    """Reducing a non-empty axis leaves the empty one empty."""
    result = _evaluate(lambda f: graph.project_using_max(f, Y), EMPTY)
    assert result.shape == (0, 1)


@pytest.mark.parametrize("value", [EMPTY, CONSTANT], ids=["empty", "constant"])
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda f: graph.shift(f, X, periods=1), id="shift"),
        pytest.param(lambda f: graph.roll(f, X, periods=1), id="roll"),
        pytest.param(lambda f: graph.cumulative(f, X), id="cumulative"),
        pytest.param(lambda f: f.diff(X), id="diff"),
        pytest.param(lambda f: graph.gradient(f, X, boundary=Constant(1.0)), id="gradient-constant"),
        pytest.param(lambda f: graph.gradient(f, X, boundary=Neumann(1.0)), id="gradient-neumann"),
        pytest.param(lambda f: graph.gradient(f, X, boundary=Nearest()), id="gradient-nearest"),
        pytest.param(lambda f: graph.gradient(f, X, boundary=Wrap()), id="gradient-wrap"),
        pytest.param(lambda f: graph.gradient(f, X, boundary=Linear()), id="gradient-linear"),
        pytest.param(lambda f: graph.laplacian(f, (X, Y), (1.0, 1.0), (Wrap(), Nearest())), id="laplacian"),
    ],
)
def test_shape_preserving_operator_gives_an_empty_result(build, value):
    """Shape-preserving operators along an empty axis give an empty result."""
    assert _evaluate(build, value).shape == (0, 3)


@pytest.mark.parametrize("boundary", [Constant(0.0), Neumann(0.0), Nearest(), Wrap(), Linear()])
def test_derivative_kernels_accept_an_empty_array(boundary):
    """The kernels return an empty float array rather than padding nothing."""
    empty = np.zeros((0, 3))
    assert kernels.gradient(empty, 0, 1.0, boundary).shape == (0, 3)
    assert kernels.laplacian(empty, (0, 1), (1.0, 1.0), (boundary, boundary)).shape == (0, 3)
