"""Tests for GenericIndex, Index, and TimeseriesIndex."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import (
    AxesInferenceWarning,
    ConstIndex,
    ConstTimeseriesIndex,
    GenericIndex,
    Index,
    TimeseriesIndex,
)
from civic_digital_twins.dt_model.axes import DOMAIN, TIME_AXIS, Axis
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import CategoricalIndex, ConditionalCategoricalIndex, DistributionIndex


def test_timeseries_index_construction():
    """Test basic construction of a TimeseriesIndex — node is an array_placeholder over axes=(TIME_AXIS,)."""
    values = np.array([1.0, 2.0, 3.0])
    idx = TimeseriesIndex("cap", values)
    assert idx.name == "cap"
    assert isinstance(idx.node, graph.array_placeholder)
    assert idx.node.output_axes == (TIME_AXIS,)
    default = idx.concrete_default
    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, values)


def test_timeseries_index_value_attribute():
    """Test that concrete_default holds the numpy array."""
    values = np.array([10.0, 20.0, 30.0])
    idx = TimeseriesIndex("cap", values)
    default = idx.concrete_default
    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, values)


def test_timeseries_index_evaluation():
    """Test that the TimeseriesIndex node evaluates to its values when state is provided."""
    values = np.array([10.0, 20.0, 30.0])
    idx = TimeseriesIndex("cap", values)
    plan = linearize.forest(idx.node)
    state = executor.State({idx.node: values})
    executor.evaluate_nodes(state, *plan)
    assert np.array_equal(state.values[idx.node], values)


def test_timeseries_index_str():
    """Test the string representation of a TimeseriesIndex."""
    idx = TimeseriesIndex("cap", np.array([1.0, 2.0]))
    assert str(idx) == "timeseries_idx([1.0, 2.0])"


def test_timeseries_index_in_arithmetic():
    """Test that a TimeseriesIndex node participates correctly in formulas (state provided)."""
    values = np.array([10.0, 20.0, 30.0])
    idx = TimeseriesIndex("cap", values)
    halved = idx.node * graph.constant(0.5)
    plan = linearize.forest(halved)
    state = executor.State({idx.node: values})
    executor.evaluate_nodes(state, *plan)
    assert np.allclose(state.values[halved], [5.0, 10.0, 15.0])


# ---------------------------------------------------------------------------
# TimeseriesIndex — placeholder (no values)
# ---------------------------------------------------------------------------


def test_timeseries_index_no_values():
    """Test construction of a TimeseriesIndex with no values (placeholder mode)."""
    idx = TimeseriesIndex("inflow")
    assert isinstance(idx.node, graph.array_placeholder)
    assert idx.node.output_axes == (TIME_AXIS,)
    assert idx.is_abstract


def test_timeseries_index_placeholder_raises_without_state():
    """Test that evaluating a value-less TimeseriesIndex without a state entry raises."""
    idx = TimeseriesIndex("inflow")
    plan = linearize.forest(idx.node)
    state = executor.State({})
    with pytest.raises(executor.PlaceholderValueNotProvided):
        executor.evaluate_nodes(state, *plan)


def test_timeseries_index_placeholder_evaluates_with_state():
    """Test that a value-less TimeseriesIndex evaluates correctly when state is provided."""
    idx = TimeseriesIndex("inflow")
    values = np.array([1.0, 2.0, 3.0])
    plan = linearize.forest(idx.node)
    state = executor.State({idx.node: values})
    executor.evaluate_nodes(state, *plan)
    assert np.array_equal(state.values[idx.node], values)


def test_timeseries_index_str_placeholder():
    """Test the string representation of a placeholder TimeseriesIndex."""
    idx = TimeseriesIndex("inflow")
    assert str(idx) == "timeseries_idx(placeholder)"


# ---------------------------------------------------------------------------
# TimeseriesIndex — formula mode (graph.Node as value)
# ---------------------------------------------------------------------------


def test_timeseries_index_formula_construction():
    """TimeseriesIndex accepts a graph.Node and reuses it directly as its node."""
    ts = TimeseriesIndex("inflow")
    formula = ts.node * ts.node
    result = TimeseriesIndex("outflow", formula)
    assert isinstance(result.node, graph.multiply)
    assert result.node is formula


def test_timeseries_index_formula_str():
    """String representation in formula mode."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts.node * ts.node)
    assert str(result).startswith("timeseries_idx(")


def test_timeseries_index_formula_evaluation():
    """TimeseriesIndex in formula mode evaluates correctly."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts.node * ts.node)
    plan = linearize.forest(result.node)
    values = np.array([2.0, 3.0, 4.0])
    state = executor.State({ts.node: values})
    executor.evaluate_nodes(state, *plan)
    assert np.allclose(state.values[result.node], values**2)


def test_timeseries_index_formula_mode():
    """TimeseriesIndex in formula mode wraps the given graph node."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts.node * graph.constant(2.0))
    assert result.node is not None


def test_timeseries_index_formula_via_operators():
    """TimeseriesIndex formula mode works with GenericIndex operators."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts * ts)
    plan = linearize.forest(result.node)
    values = np.array([2.0, 3.0, 4.0])
    state = executor.State({ts.node: values})
    executor.evaluate_nodes(state, *plan)
    assert np.allclose(state.values[result.node], values**2)


# ---------------------------------------------------------------------------
# GenericIndex arithmetic operators
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Placeholder-based node behavior tests
# ---------------------------------------------------------------------------


def test_index_scalar_creates_placeholder():
    """Index(scalar) creates a graph.placeholder node (value lives in model layer)."""
    idx = Index("cost", 8.0)
    assert isinstance(idx.node, graph.placeholder)
    assert idx.concrete_default == 8.0


def test_const_index_scalar_creates_constant():
    """ConstIndex always creates a graph.constant node regardless of the argument type."""
    idx = ConstIndex("cost", 8.0)
    assert isinstance(idx.node, graph.constant)
    assert idx.concrete_default == 8.0


def test_timeseries_index_array_creates_array_placeholder():
    """TimeseriesIndex(arr) creates an array_placeholder node over axes=(TIME_AXIS,)."""
    arr = np.array([1.0, 2.0, 3.0])
    idx = TimeseriesIndex("ts", arr)
    assert isinstance(idx.node, graph.array_placeholder)
    assert idx.node.output_axes == (TIME_AXIS,)
    default = idx.concrete_default
    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, arr)


def test_const_timeseries_index_creates_array_constant():
    """ConstTimeseriesIndex always creates an array_constant node over axes=(TIME_AXIS,)."""
    arr = np.array([1.0, 2.0, 3.0])
    idx = ConstTimeseriesIndex("ts", arr)
    assert isinstance(idx.node, graph.array_constant)
    assert idx.node.output_axes == (TIME_AXIS,)


# ---------------------------------------------------------------------------
# TimeseriesIndex / ConstTimeseriesIndex as Index specializations
#
# Regression suite for time-only parity: the timeseries types are thin
# ``axes=(TIME_AXIS,)`` specializations of Index / ConstIndex, but their
# public API/behaviour must be unchanged.
# ---------------------------------------------------------------------------


def test_timeseries_index_is_index_specialization():
    """TimeseriesIndex is an Index fixing axes=(TIME_AXIS,)."""
    assert issubclass(TimeseriesIndex, Index)
    ts = TimeseriesIndex("ts", np.array([1.0]))
    assert isinstance(ts, Index)
    assert ts.axes == (TIME_AXIS,)


def test_timeseries_index_output_axes_is_time_axis():
    """TimeseriesIndex.output_axes is exactly (TIME_AXIS,) in every mode."""
    from civic_digital_twins.dt_model.axes import TIME_AXIS

    placeholder_idx = TimeseriesIndex("inflow")
    array_idx = TimeseriesIndex("cap", np.array([1.0, 2.0, 3.0]))
    formula_idx = TimeseriesIndex("outflow", array_idx.node * graph.constant(2.0))
    const_idx = ConstTimeseriesIndex("demand", np.array([1.0, 2.0]))

    assert placeholder_idx.output_axes == (TIME_AXIS,)
    assert array_idx.output_axes == (TIME_AXIS,)
    assert formula_idx.output_axes == (TIME_AXIS,)
    assert const_idx.output_axes == (TIME_AXIS,)


def test_timeseries_index_repr_unchanged():
    """TimeseriesIndex/ConstTimeseriesIndex __repr__ strings are unchanged."""
    assert str(TimeseriesIndex("inflow")) == "timeseries_idx(placeholder)"
    assert str(TimeseriesIndex("cap", np.array([1.0, 2.0]))) == "timeseries_idx([1.0, 2.0])"
    assert str(ConstTimeseriesIndex("demand", np.array([10.0, 20.0]))) == "const_timeseries_idx([10.0, 20.0])"


def test_const_timeseries_index_node_exact_type_is_array_constant():
    """ConstTimeseriesIndex.node's exact type is unchanged (executor dispatch relies on this)."""
    idx = ConstTimeseriesIndex("demand", np.array([1.0, 2.0, 3.0]))
    assert type(idx.node) is graph.array_constant


# ---------------------------------------------------------------------------
# Index(axes=...) — the domain-carrying mode of the unified index
# ---------------------------------------------------------------------------


def test_index_fixed_array_creates_array_placeholder_and_deduces_sizes():
    """Index(array, axes=...) creates an array_placeholder and deduces per-axis sizes."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    idx = Index("field", arr, axes=(x, y))
    assert idx.name == "field"
    assert type(idx.node) is graph.array_placeholder
    assert idx.output_axes == (x, y)
    assert idx.axes == (x, y)
    assert idx.sizes == {"x": 2, "y": 3}
    default = idx.concrete_default
    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, arr)
    assert idx.is_abstract is False


def test_index_bare_placeholder_with_axes():
    """Index(axes=...) with no value is a bare domain-carrying placeholder (abstract)."""
    x = Axis("x", DOMAIN)
    idx = Index("field", axes=(x,))
    assert type(idx.node) is graph.array_placeholder
    assert idx.output_axes == (x,)
    assert idx.is_abstract is True
    assert idx.concrete_default is None
    assert idx.sizes == {}


def test_index_array_rank_must_match_declared_axes():
    """A concrete array's rank must match the number of declared axes."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    with pytest.raises(ValueError):
        Index("field", np.array([1.0, 2.0]), axes=(x, y))


def test_index_axes_declaration_is_three_valued():
    """axes= distinguishes 'undeclared' (None) from 'declared scalar' (())."""
    x = Axis("x", DOMAIN)
    assert Index("scalar", 5.0).axes is None
    assert Index("scalar", 5.0, axes=()).axes == ()
    # An empty declaration is scalar: node is a plain placeholder, not an array one.
    assert type(Index("scalar", 5.0, axes=()).node) is graph.placeholder
    assert Index("scalar", 5.0, axes=()).output_axes == ()
    assert Index("field", axes=(x,)).axes == (x,)


def test_index_formula_mode_with_axes():
    """Index(formula_node, axes=...) reuses the formula node directly."""
    x = Axis("x", DOMAIN)
    base = Index("base", np.array([1.0, 2.0]), axes=(x,))
    formula = Index("derived", base.node * graph.constant(2.0), axes=(x,))
    assert formula.node is not base.node
    assert isinstance(formula.node, graph.multiply)
    assert formula.is_abstract is False
    assert formula.concrete_default is None
    assert formula.sizes == {}


def test_index_repr_covers_all_domain_carrying_modes():
    """Index repr distinguishes placeholder, fixed-array, and formula modes when axes are declared."""
    x = Axis("x", DOMAIN)
    axes_repr = f"axes={(x,)!r}"
    assert repr(Index("f", axes=(x,))) == f"idx('f', placeholder, {axes_repr})"
    assert repr(Index("f", np.array([1.0, 2.0]), axes=(x,))) == f"idx('f', [1.0, 2.0], {axes_repr})"
    formula = Index("f", Index("b", np.array([1.0]), axes=(x,)).node + graph.constant(1.0), axes=(x,))
    assert repr(formula) == f"idx('f', <formula>, {axes_repr})"


def test_index_with_axes_is_generic_index():
    """A domain-carrying Index is an ordinary GenericIndex."""
    idx = Index("field", axes=(Axis("x", DOMAIN),))
    assert isinstance(idx, GenericIndex)
    assert isinstance(idx, Index)


def test_index_with_axes_evaluation():
    """A fixed-array domain-carrying Index evaluates to its provided values."""
    x = Axis("x", DOMAIN)
    arr = np.array([1.0, 2.0, 3.0])
    idx = Index("field", arr, axes=(x,))
    plan = linearize.forest(idx.node)
    state = executor.State({idx.node: arr})
    executor.evaluate_nodes(state, *plan)
    assert np.array_equal(state.values[idx.node], arr)


def test_index_unwraps_any_generic_index_to_its_node():
    """Index(name, other_index) reuses the other index's node, whatever its shape."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0]))
    reused = Index("reused", ts)
    assert reused.node is ts.node
    assert reused.output_axes == (TIME_AXIS,)


# ---------------------------------------------------------------------------
# Formula-mode axes= verification (verify, never override)
# ---------------------------------------------------------------------------


def test_index_formula_axes_mismatch_raises():
    """A declared axes= that contradicts the formula's inferred axes raises ValueError."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    base = Index("base", np.array([1.0, 2.0]), axes=(x,))
    with pytest.raises(ValueError, match="do not match"):
        Index("derived", base.node * graph.constant(2.0), axes=(y,))


def test_index_formula_axes_compared_as_a_set():
    """Declared axes are matched as a set — inferred order is a traversal artifact."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    a = Index("a", np.array([1.0, 2.0]), axes=(x,))
    b = Index("b", np.array([1.0, 2.0, 3.0]), axes=(y,))
    formula = a.node * b.node
    assert formula.output_axes == (x, y)
    # The reversed declaration is equally valid: the formula could as easily
    # have been written b * a, which infers (y, x) for the same value.
    idx = Index("outer", formula, axes=(y, x))
    assert idx.axes == (y, x)
    assert idx.output_axes == (x, y)


def test_index_formula_empty_axes_asserts_scalar():
    """axes=() on a formula asserts the result is scalar, and fails when it is not."""
    x = Axis("x", DOMAIN)
    scalar = Index("scalar", graph.constant(1.0) + graph.constant(2.0), axes=())
    assert scalar.output_axes == ()
    base = Index("base", np.array([1.0, 2.0]), axes=(x,))
    with pytest.raises(ValueError, match="do not match"):
        Index("not_scalar", base.node * graph.constant(2.0), axes=())


def test_index_formula_undeclared_axes_are_not_verified():
    """axes=None declares nothing: the formula's inferred axes are accepted as-is."""
    x = Axis("x", DOMAIN)
    base = Index("base", np.array([1.0, 2.0]), axes=(x,))
    derived = Index("derived", base.node * graph.constant(2.0))
    assert derived.axes is None
    assert derived.output_axes == (x,)


def test_timeseries_index_formula_verifies_the_time_axis():
    """TimeseriesIndex fixes axes=(TIME_AXIS,), so a non-time formula is rejected."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0]))
    TimeseriesIndex("scaled", ts.node * graph.constant(2.0))
    with pytest.raises(ValueError, match="do not match"):
        TimeseriesIndex("collapsed", graph.constant(1.0) + graph.constant(2.0))


# ---------------------------------------------------------------------------
# Surprising-inference warning on undeclared formulas
# ---------------------------------------------------------------------------


def test_index_warns_on_emergent_outer_product():
    """Combining operands with disjoint axes broadens the result and warns."""
    space = Axis("space", DOMAIN)
    time_series = TimeseriesIndex("a", np.array([1.0, 2.0]))
    field = Index("b", np.array([1.0, 2.0, 3.0]), axes=(space,))
    with pytest.warns(AxesInferenceWarning, match="broader than any single operand"):
        Index("c", time_series.node * field.node)


def test_index_declared_axes_suppress_the_warning(recwarn):
    """Declaring axes= states the intent, so the outer product is no longer surprising."""
    space = Axis("space", DOMAIN)
    time_series = TimeseriesIndex("a", np.array([1.0, 2.0]))
    field = Index("b", np.array([1.0, 2.0, 3.0]), axes=(space,))
    Index("c", time_series.node * field.node, axes=(TIME_AXIS, space))
    assert len(recwarn) == 0


def test_index_shared_axes_do_not_warn(recwarn):
    """Operands that already carry the result's axes produce nothing new to flag."""
    a = TimeseriesIndex("a", np.array([1.0, 2.0]))
    b = TimeseriesIndex("b", np.array([3.0, 4.0]))
    Index("c", a.node + b.node)
    assert len(recwarn) == 0


def test_index_scalar_broadcast_does_not_warn(recwarn):
    """Broadcasting a scalar against a domain-carrying operand introduces no new axis."""
    a = TimeseriesIndex("a", np.array([1.0, 2.0]))
    Index("c", a.node + graph.constant(5.0))
    assert len(recwarn) == 0


def test_index_warns_on_emergent_outer_product_in_a_where():
    """The warning covers where(), not just binary operators."""
    space = Axis("space", DOMAIN)
    time_series = TimeseriesIndex("a", np.array([1.0, 2.0]))
    field = Index("b", np.array([1.0, 2.0, 3.0]), axes=(space,))
    cond = graph.greater(time_series.node, graph.constant(0.0))
    with pytest.warns(AxesInferenceWarning):
        Index("c", graph.where(cond, field.node, graph.constant(0.0)))


def test_index_warns_on_emergent_outer_product_in_a_piecewise():
    """The warning covers multi-clause nodes such as graph.piecewise."""
    space = Axis("space", DOMAIN)
    time_series = TimeseriesIndex("a", np.array([1.0, 2.0]))
    field = Index("b", np.array([1.0, 2.0, 3.0]), axes=(space,))
    with pytest.warns(AxesInferenceWarning):
        Index("c", graph.piecewise((field.node, time_series.node > 0.0), (0.0, True)))


def test_index_piecewise_over_one_axis_does_not_warn(recwarn):
    """A piecewise whose clauses all live on the same axis introduces nothing new."""
    time_series = TimeseriesIndex("a", np.array([1.0, 2.0]))
    Index("c", graph.piecewise((time_series.node, time_series.node > 0.0), (0.0, True)))
    assert len(recwarn) == 0


def test_index_unsigned_function_call_warns():
    """A function_call's union may over-estimate, so an unsigned one always warns."""
    a = TimeseriesIndex("a", np.array([1.0, 2.0]))
    with pytest.warns(AxesInferenceWarning, match="function_call"):
        Index("r", graph.function_call("reduce", a.node))


def test_index_scalar_function_call_does_not_warn(recwarn):
    """A function_call over scalar arguments has an empty union: nothing to over-estimate."""
    Index("r", graph.function_call("scale", graph.constant(2.0)))
    assert len(recwarn) == 0


def test_index_signed_function_call_does_not_warn(recwarn):
    """A functor declaring output_axes makes the inference exact, so no warning."""
    a = TimeseriesIndex("a", np.array([1.0, 2.0]))
    functor = executor.NumpyBackend.adapt(lambda x: x.sum(axis=-1), output_axes=())
    reduced = Index("r", graph.function_call("reduce", a.node, functor=functor))
    assert reduced.output_axes == ()
    assert len(recwarn) == 0


# ---------------------------------------------------------------------------
# Reduction-axis default resolution
# ---------------------------------------------------------------------------


def test_reduction_defaults_to_unique_non_time_domain_axis():
    """On a single-DOMAIN-axis index, a no-axis reduction defaults to that axis, not time."""
    x = Axis("x", DOMAIN)
    idx = Index("field", np.array([1.0, 2.0, 3.0]), axes=(x,))
    node = idx.sum()
    assert isinstance(node, graph.project_using_sum)
    assert node.axis == x


def test_reduction_requires_explicit_axis_when_multiple_domain_axes():
    """On a multi-DOMAIN-axis index, a no-axis reduction is ambiguous and raises."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    idx = Index("field", np.array([[1.0, 2.0], [3.0, 4.0]]), axes=(x, y))
    with pytest.raises(ValueError, match="explicit axis="):
        idx.sum()
    # An explicit axis resolves the ambiguity.
    node = idx.sum(axis=y)
    assert isinstance(node, graph.project_using_sum)
    assert node.axis == y


def test_reduction_without_a_domain_axis_requires_an_explicit_axis():
    """A scalar index has no dimension to reduce, so no default can be inferred.

    It used to fall back to the time axis, reproducing a "reduce the last
    dimension" convention from when every array was a timeseries.  The executor
    now reduces the dimension an axis *names*, so that fallback would ask it to
    reduce a time axis the evaluation need not carry.
    """
    idx = Index("scalar", 5.0)
    with pytest.raises(ValueError, match="carries no DOMAIN axis"):
        idx.sum()


# ---------------------------------------------------------------------------
# ConstIndex(axes=...) — value baked into the graph as an array_constant
# ---------------------------------------------------------------------------


def test_const_index_with_axes_creates_array_constant():
    """ConstIndex(axes=...) bakes its values into an array_constant node carrying its axes."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    idx = ConstIndex("field", arr, axes=(x, y))
    assert type(idx.node) is graph.array_constant
    assert idx.output_axes == (x, y)
    assert idx.axes == (x, y)
    assert idx.sizes == {"x": 2, "y": 3}
    assert idx.is_abstract is False


def test_const_index_without_axes_is_scalar_constant():
    """ConstIndex with no declared axes keeps building a plain scalar constant."""
    idx = ConstIndex("factor", 2.5)
    assert type(idx.node) is graph.constant
    assert idx.output_axes == ()
    assert idx.axes is None
    assert idx.sizes == {}


def test_const_index_with_axes_repr_roundtrips_axes():
    """ConstIndex repr surfaces name, values, and declared axes when domain-carrying."""
    x = Axis("x", DOMAIN)
    idx = ConstIndex("field", np.array([1.0, 2.0]), axes=(x,))
    assert repr(idx) == f"const_idx('field', [1.0, 2.0], axes={(x,)!r})"


# ---------------------------------------------------------------------------
# GenericIndex arithmetic operators (original section)
# ---------------------------------------------------------------------------


def _eval(node: graph.Node) -> np.ndarray:
    """Evaluate a graph node with no external placeholder values."""
    state = executor.State({})
    executor.evaluate_nodes(state, *linearize.forest(node))
    return state.values[node]


def test_generic_index_is_abstract():
    """GenericIndex cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GenericIndex()  # type: ignore[abstract]


def test_index_is_generic_index():
    """Index is a subclass of GenericIndex."""
    assert issubclass(Index, GenericIndex)
    assert isinstance(Index("x", 1.0), GenericIndex)


def test_timeseries_index_is_generic_index():
    """TimeseriesIndex is a subclass of GenericIndex."""
    assert isinstance(TimeseriesIndex("ts", np.array([1.0])), GenericIndex)


def test_index_add_scalar():
    """Index + scalar produces an add node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = idx + 2.0
    assert isinstance(node, graph.add)
    assert np.isclose(_eval(node), 5.0)


def test_index_radd_scalar():
    """Scalar + index produces an add node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = 2.0 + idx
    assert isinstance(node, graph.add)
    assert np.isclose(_eval(node), 5.0)


def test_index_sub():
    """Index - scalar produces a subtract node with the correct value."""
    idx = ConstIndex("a", 5.0)
    node = idx - 2.0
    assert isinstance(node, graph.subtract)
    assert np.isclose(_eval(node), 3.0)


def test_index_rsub():
    """Scalar - index produces a subtract node with the correct value."""
    idx = ConstIndex("a", 2.0)
    node = 5.0 - idx
    assert isinstance(node, graph.subtract)
    assert np.isclose(_eval(node), 3.0)


def test_index_mul_scalar():
    """Index * scalar produces a multiply node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = idx * 4.0
    assert isinstance(node, graph.multiply)
    assert np.isclose(_eval(node), 12.0)


def test_index_rmul_scalar():
    """Scalar * index produces a multiply node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = 4.0 * idx
    assert isinstance(node, graph.multiply)
    assert np.isclose(_eval(node), 12.0)


def test_index_truediv():
    """Index / scalar produces a divide node with the correct value."""
    idx = ConstIndex("a", 6.0)
    node = idx / 2.0
    assert isinstance(node, graph.divide)
    assert np.isclose(_eval(node), 3.0)


def test_index_rtruediv():
    """Scalar / index produces a divide node with the correct value."""
    idx = ConstIndex("a", 2.0)
    node = 6.0 / idx
    assert isinstance(node, graph.divide)
    assert np.isclose(_eval(node), 3.0)


def test_index_pow():
    """Index ** scalar produces a power node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = idx**2.0
    assert isinstance(node, graph.power)
    assert np.isclose(_eval(node), 9.0)


def test_index_rpow():
    """Scalar ** index produces a power node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = 2.0**idx
    assert isinstance(node, graph.power)
    assert np.isclose(_eval(node), 8.0)


def test_index_add_index():
    """Two Index objects can be combined with operators directly."""
    a = ConstIndex("a", 3.0)
    b = ConstIndex("b", 4.0)
    node = a + b
    assert isinstance(node, graph.add)
    assert np.isclose(_eval(node), 7.0)


def test_index_mul_index():
    """Two Index objects multiplied together produce a multiply node."""
    a = ConstIndex("a", 3.0)
    b = ConstIndex("b", 4.0)
    node = a * b
    assert isinstance(node, graph.multiply)
    assert np.isclose(_eval(node), 12.0)


def test_timeseries_index_arithmetic():
    """TimeseriesIndex participates in formulas without .node access (state provided)."""
    arr = np.array([1.0, 2.0, 3.0])
    ts = TimeseriesIndex("ts", arr)
    node = ts * 2.0
    assert isinstance(node, graph.multiply)
    plan = linearize.forest(node)
    state = executor.State({ts.node: arr})
    executor.evaluate_nodes(state, *plan)
    assert np.allclose(state.values[node], [2.0, 4.0, 6.0])


def test_index_comparison_operators():
    """Comparison operators return graph nodes (lazy evaluation)."""
    a = ConstIndex("a", 3.0)
    b = ConstIndex("b", 5.0)

    assert isinstance(a == b, graph.equal)
    assert isinstance(a != b, graph.not_equal)
    assert isinstance(a < b, graph.less)
    assert isinstance(a <= b, graph.less_equal)
    assert isinstance(a > b, graph.greater)
    assert isinstance(a >= b, graph.greater_equal)


def test_index_eq_evaluates_correctly():
    """== returns True when both operands have the same value."""
    node = ConstIndex("a", 3.0) == ConstIndex("b", 3.0)
    assert bool(_eval(node))


def test_index_lt_evaluates_correctly():
    """< returns True when the left operand is smaller."""
    node = ConstIndex("a", 2.0) < ConstIndex("b", 5.0)
    assert bool(_eval(node))


def test_index_hash_is_identity_based():
    """Index objects remain usable as dict keys despite overriding __eq__."""
    a = ConstIndex("a", 1.0)
    b = ConstIndex("b", 1.0)
    d = {a: "x", b: "y"}
    assert d[a] == "x"
    assert d[b] == "y"
    assert a in d
    assert b in d


# ---------------------------------------------------------------------------
# GenericIndex.__neg__
# ---------------------------------------------------------------------------


def test_index_neg_returns_negate_node():
    """__neg__ returns a negate graph node wrapping the index's node."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
    result = -ts
    assert isinstance(result, graph.negate)
    assert result.node is ts.node


def test_index_neg_evaluates_correctly():
    """__neg__ evaluates to the element-wise negation of the index values (state provided)."""
    arr = np.array([1.0, -2.0, 3.0])
    ts = TimeseriesIndex("ts", arr)
    neg_node = -ts
    plan = linearize.forest(neg_node)
    state = executor.State({ts.node: arr})
    executor.evaluate_nodes(state, *plan)
    assert np.allclose(state.values[neg_node], [-1.0, 2.0, -3.0])


# ---------------------------------------------------------------------------
# Index reduction methods (sum, mean, min, max, etc.) are comprehensively tested in
# tests/dt_model/symbols/test_index_reduction_methods.py


# ---------------------------------------------------------------------------
# DistributionIndex — properties and params setter
# ---------------------------------------------------------------------------


def test_distribution_index_distribution_property():
    """DistributionIndex.distribution returns the callable used at construction."""
    from scipy import stats

    from civic_digital_twins.dt_model import DistributionIndex

    idx = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 1.0})
    assert idx.distribution is stats.uniform


def test_distribution_index_params_property_returns_copy():
    """DistributionIndex.params returns a copy of the params dict."""
    from scipy import stats

    from civic_digital_twins.dt_model import DistributionIndex

    idx = DistributionIndex("x", stats.uniform, {"loc": 1.0, "scale": 2.0})
    p = idx.params
    assert p == {"loc": 1.0, "scale": 2.0}
    # Mutating the returned copy must not affect the stored params.
    p["loc"] = 99.0
    assert idx.params["loc"] == 1.0


# ---------------------------------------------------------------------------
# ConstIndex
# ---------------------------------------------------------------------------


def test_const_index_value():
    """ConstIndex.concrete_default returns the constant value."""
    idx = ConstIndex("c", 42.0)
    assert idx.concrete_default == 42.0


def test_const_index_str():
    """ConstIndex.__repr__ returns the expected representation."""
    idx = ConstIndex("c", 5.0)
    assert str(idx) == "const_idx(5.0)"


# ---------------------------------------------------------------------------
# ConstTimeseriesIndex — construction, values property, setter, str, hierarchy
# ---------------------------------------------------------------------------


def test_const_timeseries_index_construction():
    """ConstTimeseriesIndex holds a concrete array backed by an array_constant."""
    arr = np.array([1.0, 2.0, 3.0])
    ts = ConstTimeseriesIndex("demand", arr)
    assert ts.name == "demand"
    default = ts.concrete_default
    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, arr)
    assert isinstance(ts.node, graph.array_constant)
    assert ts.node.output_axes == (TIME_AXIS,)


def test_const_timeseries_index_is_both_const_index_and_timeseries_index():
    """ConstTimeseriesIndex specializes ConstIndex and stays a TimeseriesIndex.

    ``ConstIndex`` carries the const behaviour (Scenario refuses to override
    it); ``TimeseriesIndex`` is the shape declaration that model
    ``Inputs``/``Outputs`` contracts annotate a time-shaped field with, so a
    const timeseries must remain assignable to it.
    """
    ts = ConstTimeseriesIndex("demand", np.array([1.0]))
    assert isinstance(ts, ConstIndex)
    assert isinstance(ts, TimeseriesIndex)
    assert isinstance(ts, Index)
    assert isinstance(ts, GenericIndex)
    assert ts.axes == (TIME_AXIS,)


def test_const_timeseries_index_mro_resolves_construction_to_const_index():
    """The ConstIndex base wins for construction, so the node is baked, not a placeholder."""
    ts = ConstTimeseriesIndex("demand", np.array([1.0, 2.0]))
    assert type(ts.node) is graph.array_constant
    assert ts.is_abstract is False
    assert ConstTimeseriesIndex.__mro__.index(ConstIndex) < ConstTimeseriesIndex.__mro__.index(TimeseriesIndex)


def test_const_timeseries_index_evaluates_correctly():
    """ConstTimeseriesIndex node evaluates to its stored array."""
    arr = np.array([10.0, 20.0, 30.0])
    ts = ConstTimeseriesIndex("demand", arr)
    state = executor.State({}, domain_axes=(TIME_AXIS,))
    executor.evaluate_nodes(state, *linearize.forest(ts.node))
    assert np.array_equal(state.values[ts.node], arr)


def test_const_timeseries_index_str():
    """ConstTimeseriesIndex.__str__ uses the const_timeseries_idx prefix."""
    ts = ConstTimeseriesIndex("demand", np.array([1.0, 2.0]))
    assert str(ts) == "const_timeseries_idx([1.0, 2.0])"


def test_index_rejects_distribution():
    """Index cannot be initialised directly with a Distribution; raises TypeError."""
    dist = stats.norm(loc=0, scale=1)
    with pytest.raises(TypeError, match="DistributionIndex"):
        Index("x", dist)  # type: ignore[arg-type]


def test_index_unwraps_nested_index():
    """Index(name, other_index) shares other_index's node instead of orphaning a new one."""
    base = Index("base", None)
    copy = Index("copy", base)
    assert copy.node is base.node


def test_timeseries_index_unwraps_nested_timeseries_index():
    """TimeseriesIndex(name, other_ts_index) shares other's node instead of orphaning a new one.

    Regression test for https://github.com/fbk-most/civic-digital-twins/issues/223: before the
    fix, TimeseriesIndex.__init__ had no unwrap logic for a nested GenericIndex, so passing
    another TimeseriesIndex (instead of its .node) fell through to np.asarray(other_index),
    which silently wrapped the whole object in a 0-d dtype=object array and minted a brand-new,
    disconnected timeseries_placeholder node.
    """
    ts_base = TimeseriesIndex("ts_base")
    ts_copy = TimeseriesIndex("ts_copy", ts_base)
    assert ts_copy.node is ts_base.node
    assert ts_copy.concrete_default is None


def test_timeseries_index_unwraps_nested_timeseries_index_with_default():
    """The unwrap also covers a base index that carries a concrete default array."""
    ts_base = TimeseriesIndex("ts_base", np.array([1.0, 2.0, 3.0]))
    ts_copy = TimeseriesIndex("ts_copy", ts_base)
    assert ts_copy.node is ts_base.node


def test_index_repr_formula_mode():
    """Index repr shows '<formula>' when the value is a graph node."""
    n = graph.constant(42.0)
    idx = Index("r", n)
    assert repr(idx) == "idx('r', <formula>)"


def test_distribution_index_sample():
    """DistributionIndex.sample() returns an array of the requested size."""
    idx = DistributionIndex("x", stats.norm, {"loc": 0.0, "scale": 1.0})
    rng = np.random.default_rng(0)
    samples = idx.sample(rng=rng, size=10)
    assert samples.shape == (10,)


def test_conditional_categorical_sample_for_no_rng():
    """ConditionalCategoricalIndex.sample_for without rng uses global numpy random state."""
    parent = CategoricalIndex("season", {"summer": 0.5, "winter": 0.5})
    idx = ConditionalCategoricalIndex(
        "temp_band",
        parents=[parent],
        support=["hot", "cold"],
        factory=lambda season: {"hot": 0.99, "cold": 0.01} if season == "summer" else {"hot": 0.01, "cold": 0.99},
    )
    np.random.seed(0)
    samples = idx.sample_for(size=100, season="summer")
    assert "hot" in samples
