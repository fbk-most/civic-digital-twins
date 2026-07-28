"""Tests for GenericIndex, Index, and TimeseriesIndex."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import (
    ConstDomainIndex,
    ConstIndex,
    ConstTimeseriesIndex,
    DomainIndex,
    GenericIndex,
    Index,
    TimeseriesIndex,
)
from civic_digital_twins.dt_model.axes import DOMAIN, TIME_AXIS, Axis
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import CategoricalIndex, ConditionalCategoricalIndex, DistributionIndex


def test_timeseries_index_construction():
    """Test basic construction of a TimeseriesIndex — node is timeseries_placeholder."""
    values = np.array([1.0, 2.0, 3.0])
    idx = TimeseriesIndex("cap", values)
    assert idx.name == "cap"
    assert isinstance(idx.node, graph.timeseries_placeholder)
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
    assert isinstance(idx.node, graph.timeseries_placeholder)
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


def test_timeseries_index_array_creates_timeseries_placeholder():
    """TimeseriesIndex(arr) creates a timeseries_placeholder node."""
    arr = np.array([1.0, 2.0, 3.0])
    idx = TimeseriesIndex("ts", arr)
    assert isinstance(idx.node, graph.timeseries_placeholder)
    default = idx.concrete_default
    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, arr)


def test_const_timeseries_index_creates_timeseries_constant():
    """ConstTimeseriesIndex always creates a timeseries_constant node."""
    arr = np.array([1.0, 2.0, 3.0])
    idx = ConstTimeseriesIndex("ts", arr)
    assert isinstance(idx.node, graph.timeseries_constant)


# ---------------------------------------------------------------------------
# TimeseriesIndex / ConstTimeseriesIndex as DomainIndex specializations
#
# Regression suite for time-only parity: TimeseriesIndex became a
# DomainIndex subclass in this step, but its public API/behaviour must be
# unchanged.
# ---------------------------------------------------------------------------


def test_timeseries_index_is_domain_index():
    """TimeseriesIndex is now a DomainIndex specialization."""
    assert issubclass(TimeseriesIndex, DomainIndex)
    assert isinstance(TimeseriesIndex("ts", np.array([1.0])), DomainIndex)


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


def test_const_timeseries_index_node_still_timeseries_constant_exact_type():
    """ConstTimeseriesIndex.node's exact type is unchanged (executor dispatch relies on this)."""
    idx = ConstTimeseriesIndex("demand", np.array([1.0, 2.0, 3.0]))
    assert type(idx.node) is graph.timeseries_constant


# ---------------------------------------------------------------------------
# DomainIndex — generic domain-carrying index
# ---------------------------------------------------------------------------


def test_domain_index_fixed_array_creates_array_placeholder_and_deduces_sizes():
    """DomainIndex(array, axes=...) creates an array_placeholder and deduces per-axis sizes."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    idx = DomainIndex("field", arr, axes=(x, y))
    assert idx.name == "field"
    assert type(idx.node) is graph.array_placeholder
    assert idx.output_axes == (x, y)
    assert idx.axes == (x, y)
    assert idx.sizes == {"x": 2, "y": 3}
    default = idx.concrete_default
    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, arr)
    assert idx.is_abstract is False


def test_domain_index_bare_placeholder():
    """DomainIndex(axes=...) with no value is a bare placeholder (abstract)."""
    x = Axis("x", DOMAIN)
    idx = DomainIndex("field", axes=(x,))
    assert type(idx.node) is graph.array_placeholder
    assert idx.output_axes == (x,)
    assert idx.is_abstract is True
    assert idx.concrete_default is None
    assert idx.sizes == {}


def test_domain_index_requires_at_least_one_axis():
    """DomainIndex must declare at least one DOMAIN axis (an axis-less one is a scalar Index)."""
    with pytest.raises(ValueError, match="at least one DOMAIN axis"):
        DomainIndex("field", np.array([1.0]), axes=())
    # axes is a required keyword: omitting it entirely is a TypeError.
    with pytest.raises(TypeError):
        DomainIndex("field")  # type: ignore[call-arg]


def test_domain_index_formula_mode():
    """DomainIndex(formula_node, axes=...) reuses the formula node directly."""
    x = Axis("x", DOMAIN)
    base = DomainIndex("base", np.array([1.0, 2.0]), axes=(x,))
    formula = DomainIndex("derived", base.node * graph.constant(2.0), axes=(x,))
    assert formula.node is not base.node
    assert isinstance(formula.node, graph.multiply)
    assert formula.is_abstract is False
    assert formula.concrete_default is None


def test_domain_index_repr_covers_all_modes():
    """DomainIndex repr distinguishes placeholder, fixed-array, and formula modes."""
    x = Axis("x", DOMAIN)
    axes_repr = f"axes={(x,)!r}"
    assert repr(DomainIndex("f", axes=(x,))) == f"domain_idx('f', placeholder, {axes_repr})"
    assert repr(DomainIndex("f", np.array([1.0, 2.0]), axes=(x,))) == f"domain_idx('f', [1.0, 2.0], {axes_repr})"
    formula = DomainIndex("f", DomainIndex("b", np.array([1.0]), axes=(x,)).node + graph.constant(1.0), axes=(x,))
    assert repr(formula) == f"domain_idx('f', <formula>, {axes_repr})"


def test_domain_index_is_generic_index_not_domain_index_special_cased():
    """DomainIndex is a plain GenericIndex sibling, like TimeseriesIndex."""
    assert issubclass(DomainIndex, GenericIndex)
    idx = DomainIndex("field", axes=(Axis("x", DOMAIN),))
    assert isinstance(idx, GenericIndex)


def test_domain_index_evaluation():
    """A fixed-array DomainIndex evaluates to its provided values."""
    x = Axis("x", DOMAIN)
    arr = np.array([1.0, 2.0, 3.0])
    idx = DomainIndex("field", arr, axes=(x,))
    plan = linearize.forest(idx.node)
    state = executor.State({idx.node: arr})
    executor.evaluate_nodes(state, *plan)
    assert np.array_equal(state.values[idx.node], arr)


# ---------------------------------------------------------------------------
# Reduction-axis default resolution
# ---------------------------------------------------------------------------


def test_reduction_defaults_to_unique_non_time_domain_axis():
    """On a single-DOMAIN-axis index, a no-axis reduction defaults to that axis, not time."""
    x = Axis("x", DOMAIN)
    idx = DomainIndex("field", np.array([1.0, 2.0, 3.0]), axes=(x,))
    node = idx.sum()
    assert isinstance(node, graph.project_using_sum)
    assert node.axis == x


def test_reduction_requires_explicit_axis_when_multiple_domain_axes():
    """On a multi-DOMAIN-axis index, a no-axis reduction is ambiguous and raises."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    idx = DomainIndex("field", np.array([[1.0, 2.0], [3.0, 4.0]]), axes=(x, y))
    with pytest.raises(ValueError, match="explicit axis="):
        idx.sum()
    # An explicit axis resolves the ambiguity.
    node = idx.sum(axis=y)
    assert isinstance(node, graph.project_using_sum)
    assert node.axis == y


def test_reduction_zero_domain_axes_falls_back_to_time():
    """A scalar/untyped index keeps the legacy time-axis default (reduce last dimension)."""
    idx = Index("scalar", 5.0)
    node = idx.sum()
    assert isinstance(node, graph.project_using_sum)
    assert node.axis == TIME_AXIS


# ---------------------------------------------------------------------------
# ConstDomainIndex — value baked into the graph as an array_constant
# ---------------------------------------------------------------------------


def test_const_domain_index_creates_array_constant():
    """ConstDomainIndex bakes its values into an array_constant node carrying its axes."""
    x = Axis("x", DOMAIN)
    y = Axis("y", DOMAIN)
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    idx = ConstDomainIndex("field", arr, axes=(x, y))
    assert type(idx.node) is graph.array_constant
    assert idx.output_axes == (x, y)
    assert idx.axes == (x, y)
    assert idx.sizes == {"x": 2, "y": 3}
    assert idx.is_abstract is False
    assert issubclass(ConstDomainIndex, DomainIndex)


def test_const_domain_index_requires_at_least_one_axis():
    """ConstDomainIndex, like DomainIndex, requires at least one DOMAIN axis."""
    with pytest.raises(ValueError, match="at least one DOMAIN axis"):
        ConstDomainIndex("field", np.array([1.0]), axes=())


def test_const_domain_index_repr_roundtrips_axes():
    """ConstDomainIndex repr surfaces name, values, and declared axes."""
    x = Axis("x", DOMAIN)
    idx = ConstDomainIndex("field", np.array([1.0, 2.0]), axes=(x,))
    assert repr(idx) == f"const_domain_idx('field', [1.0, 2.0], axes={(x,)!r})"


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


def test_timeseries_index_is_not_index():
    """TimeseriesIndex is a sibling of Index, not a subclass."""
    assert not issubclass(TimeseriesIndex, Index)
    assert not isinstance(TimeseriesIndex("ts", np.array([1.0])), Index)


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
    """ConstTimeseriesIndex holds a concrete array backed by timeseries_constant."""
    arr = np.array([1.0, 2.0, 3.0])
    ts = ConstTimeseriesIndex("demand", arr)
    assert ts.name == "demand"
    default = ts.concrete_default
    assert isinstance(default, np.ndarray)
    assert np.array_equal(default, arr)
    assert isinstance(ts.node, graph.timeseries_constant)


def test_const_timeseries_index_is_timeseries_index():
    """ConstTimeseriesIndex is a subclass of TimeseriesIndex and GenericIndex."""
    ts = ConstTimeseriesIndex("demand", np.array([1.0]))
    assert isinstance(ts, TimeseriesIndex)
    assert isinstance(ts, GenericIndex)
    assert not isinstance(ts, Index)


def test_const_timeseries_index_evaluates_correctly():
    """ConstTimeseriesIndex node evaluates to its stored array."""
    arr = np.array([10.0, 20.0, 30.0])
    ts = ConstTimeseriesIndex("demand", arr)
    state = executor.State({})
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


def test_index_rejects_timeseries_index():
    """Index rejects a differently-shaped sibling (TimeseriesIndex) rather than silently misfiring."""
    ts = TimeseriesIndex("t", np.array([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match="TimeseriesIndex"):
        Index("x", ts)  # type: ignore[arg-type]


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
