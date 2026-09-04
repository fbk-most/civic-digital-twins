# SPDX-License-Identifier: Apache-2.0
"""Repr round-trip idempotency suite for all concrete graph.Node subtypes.

Property under test: for every concrete Node subtype, exec'ing the node's
repr (with dependency nodes already in scope) produces a structurally identical
node whose repr body — the part after the ``nX = `` SSA prefix — is unchanged.

Node IDs are globally auto-incrementing, so the self-ID half of the repr will
differ between the original and the reconstructed node.  The invariant is
therefore:

    body(repr(node2)) == body(repr(node))

where ``body(r)`` strips the leading ``nX = `` assignment.
"""

from typing import Any

from civic_digital_twins.dt_model.axes import DOMAIN, Axis, DomainAxis, TimeType
from civic_digital_twins.dt_model.engine.frontend import graph


def _body(r: str) -> str:
    """Return the constructor-call portion of an SSA repr string."""
    return r.split(" = ", 1)[1]


def _ctx(*nodes: graph.Node) -> dict[str, Any]:
    """Build an eval context mapping each node's SSA name to the node object."""
    return {f"n{n.id}": n for n in nodes}


def _assert_roundtrip(node: graph.Node, extra_ctx: dict[str, Any] | None = None) -> None:
    """Assert that the repr round-trip property holds for *node*.

    Exec's ``repr(node)`` in a context that contains ``graph``, ``Axis``,
    and all provided dependency nodes, then checks body equality.
    """
    ctx: dict[str, Any] = {"graph": graph, "Axis": Axis, "DomainAxis": DomainAxis, "TimeType": TimeType}
    if extra_ctx:
        ctx.update(extra_ctx)
    exec(repr(node), ctx)  # noqa: S102
    node2: graph.Node = ctx[f"n{node.id}"]
    assert _body(repr(node2)) == _body(repr(node))


# ---------------------------------------------------------------------------
# Leaf nodes (no dependencies)
# ---------------------------------------------------------------------------


def test_leaf_nodes() -> None:
    """Leaf nodes have no dep references in their repr — empty eval context."""
    _assert_roundtrip(graph.constant(42, name="c"))
    _assert_roundtrip(graph.constant(3.14))
    _assert_roundtrip(graph.placeholder("p", default_value=11))
    _assert_roundtrip(graph.placeholder("q"))


def test_array_nodes() -> None:
    """Round-trip test for the generic array_constant / array_placeholder nodes."""
    x_axis = Axis("x", DOMAIN)
    y_axis = Axis("y", DOMAIN)
    _assert_roundtrip(graph.array_constant([[1.0, 2.0], [3.0, 4.0]], axes=(x_axis, y_axis), name="grid"))
    _assert_roundtrip(graph.array_placeholder("field", axes=(x_axis, y_axis)))
    _assert_roundtrip(graph.array_constant(1.0))
    _assert_roundtrip(graph.array_placeholder("noaxes"))


# ---------------------------------------------------------------------------
# Unary ops
# ---------------------------------------------------------------------------


def test_unary_ops() -> None:
    """Round-trip test for all UnaryOp subclasses."""
    a = graph.constant(5, name="a")
    deps = _ctx(a)
    _assert_roundtrip(graph.negate(a), deps)
    _assert_roundtrip(graph.logical_not(a), deps)
    _assert_roundtrip(graph.exp(a), deps)
    _assert_roundtrip(graph.log(a), deps)


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------


def test_binary_ops() -> None:
    """Round-trip test for all BinaryOp subclasses."""
    a = graph.constant(2, name="a")
    b = graph.constant(3, name="b")
    deps = _ctx(a, b)
    _assert_roundtrip(graph.add(a, b), deps)
    _assert_roundtrip(graph.subtract(a, b), deps)
    _assert_roundtrip(graph.multiply(a, b), deps)
    _assert_roundtrip(graph.divide(a, b), deps)
    _assert_roundtrip(graph.power(a, b), deps)
    _assert_roundtrip(graph.equal(a, b), deps)
    _assert_roundtrip(graph.not_equal(a, b), deps)
    _assert_roundtrip(graph.less(a, b), deps)
    _assert_roundtrip(graph.less_equal(a, b), deps)
    _assert_roundtrip(graph.greater(a, b), deps)
    _assert_roundtrip(graph.greater_equal(a, b), deps)
    _assert_roundtrip(graph.logical_and(a, b), deps)
    _assert_roundtrip(graph.logical_or(a, b), deps)
    _assert_roundtrip(graph.logical_xor(a, b), deps)
    _assert_roundtrip(graph.maximum(a, b), deps)


# ---------------------------------------------------------------------------
# Conditional ops
# ---------------------------------------------------------------------------


def test_where() -> None:
    """Round-trip test for where and multi_clause_where."""
    cond = graph.placeholder("cond")
    then = graph.constant(1, name="then")
    otherwise = graph.constant(0, name="otherwise")
    _assert_roundtrip(graph.where(cond, then, otherwise), _ctx(cond, then, otherwise))

    cond2 = graph.placeholder("cond2")
    default = graph.constant(-1, name="default")
    mcw = graph.multi_clause_where([(cond, then), (cond2, otherwise)], default)
    _assert_roundtrip(mcw, _ctx(cond, then, cond2, otherwise, default))


# ---------------------------------------------------------------------------
# variant_selector and exclusive_multi_clause_where (mutually referential)
# ---------------------------------------------------------------------------


def test_variant_selector_and_exclusive_where() -> None:
    """Round-trip test for variant_selector and exclusive_multi_clause_where."""
    selector = graph.placeholder("branch")
    v_a = graph.constant(1.0, name="v_a")
    v_b = graph.constant(2.0, name="v_b")
    default = graph.constant(0.0, name="default")
    cond_a = graph.placeholder("cond_a")
    cond_b = graph.placeholder("cond_b")

    # Build variant_selector with empty merge_nodes; populate after.
    vs = graph.variant_selector(
        selector_node=selector,
        branch_map={"a": [v_a], "b": [v_b]},
        merge_nodes=[],
    )
    emcw = graph.exclusive_multi_clause_where(
        clauses=[(cond_a, v_a), (cond_b, v_b)],
        default_value=default,
        companion=vs,
    )
    vs.merge_nodes.append(emcw)

    # Test variant_selector: deps include everything referenced in its repr.
    _assert_roundtrip(vs, _ctx(selector, v_a, v_b, emcw))

    # Test exclusive_multi_clause_where: deps include the companion vs.
    _assert_roundtrip(emcw, _ctx(cond_a, v_a, cond_b, v_b, default, vs))


# ---------------------------------------------------------------------------
# Projection ops
# ---------------------------------------------------------------------------


def test_projection_ops() -> None:
    """Round-trip test for all ProjectionOp subclasses."""
    a = graph.constant(7, name="a")
    ax = Axis("time", DOMAIN)
    deps = _ctx(a)
    _assert_roundtrip(graph.project_using_sum(a, ax), deps)
    _assert_roundtrip(graph.project_using_mean(a, ax), deps)
    _assert_roundtrip(graph.project_using_min(a, ax), deps)
    _assert_roundtrip(graph.project_using_max(a, ax), deps)
    _assert_roundtrip(graph.project_using_std(a, ax), deps)
    _assert_roundtrip(graph.project_using_var(a, ax), deps)
    _assert_roundtrip(graph.project_using_median(a, ax), deps)
    _assert_roundtrip(graph.project_using_prod(a, ax), deps)
    _assert_roundtrip(graph.project_using_any(a, ax), deps)
    _assert_roundtrip(graph.project_using_all(a, ax), deps)
    _assert_roundtrip(graph.project_using_count_nonzero(a, ax), deps)
    _assert_roundtrip(graph.project_using_quantile(a, ax, q=0.95), deps)


# ---------------------------------------------------------------------------
# function_call
# ---------------------------------------------------------------------------


def test_function_call() -> None:
    """Round-trip test for function_call with positional, keyword, and mixed args."""
    x = graph.constant(1, name="x")
    y = graph.constant(2, name="y")
    z = graph.constant(3, name="z")

    # Positional args only
    _assert_roundtrip(graph.function_call("f", x, y), _ctx(x, y))
    # Keyword args only
    _assert_roundtrip(graph.function_call("g", a=x, b=y), _ctx(x, y))
    # Mixed
    _assert_roundtrip(graph.function_call("h", x, b=y, c=z), _ctx(x, y, z))


# ---------------------------------------------------------------------------
# Completeness guard
# ---------------------------------------------------------------------------

# Declared set of Node subclasses covered by this suite.
# graph.multiply is deliberately absent to demonstrate the guard catches it.
_TESTED_TYPES: frozenset[type] = frozenset(
    {
        graph.constant,
        graph.placeholder,
        graph.array_constant,
        graph.array_placeholder,
        graph.negate,
        graph.logical_not,
        graph.exp,
        graph.log,
        graph.add,
        graph.subtract,
        graph.multiply,
        graph.divide,
        graph.power,
        graph.equal,
        graph.not_equal,
        graph.less,
        graph.less_equal,
        graph.greater,
        graph.greater_equal,
        graph.logical_and,
        graph.logical_or,
        graph.logical_xor,
        graph.maximum,
        graph.where,
        graph.multi_clause_where,
        graph.exclusive_multi_clause_where,
        graph.variant_selector,
        graph.project_using_sum,
        graph.project_using_mean,
        graph.project_using_min,
        graph.project_using_max,
        graph.project_using_std,
        graph.project_using_var,
        graph.project_using_median,
        graph.project_using_prod,
        graph.project_using_any,
        graph.project_using_all,
        graph.project_using_count_nonzero,
        graph.project_using_quantile,
        graph.function_call,
    }
)


def _concrete_node_types() -> frozenset[type]:
    """Return every Node subclass that defines its own ``__repr__``."""

    def collect(cls: type) -> set[type]:
        result: set[type] = set()
        for sub in cls.__subclasses__():
            if "__repr__" in sub.__dict__:
                result.add(sub)
            result |= collect(sub)
        return result

    return frozenset(collect(graph.Node))


def test_completeness() -> None:
    """Fail if any concrete Node subclass with __repr__ is missing from _TESTED_TYPES."""
    all_concrete = _concrete_node_types()
    missing = all_concrete - _TESTED_TYPES
    extra = _TESTED_TYPES - all_concrete
    assert not missing, f"Node types with __repr__ not in _TESTED_TYPES: {missing}"
    assert not extra, f"Types in _TESTED_TYPES are not concrete Node subclasses: {extra}"
