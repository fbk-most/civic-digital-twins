"""Topologically-Sorted-Graph Executor.

An evaluator for computation graphs that processes nodes sorted in
topological order. Unlike recursive evaluators, this executor requires
pre-linearized graphs where nodes are sorted such that all dependencies
of a node appear before the node itself in the evaluation sequence.

This approach offers several advantages over walking the AST:
- Clearer debugging: execution follows a predictable linear sequence
- Better tracing: provides a coherent view of computation flow
- Explicit error handling: clearly identifies missing dependency errors

The executor expects all placeholder values to be provided in the initial
state and evaluates each node exactly once, storing results for later reuse.
"""
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np

from ...axes import DOMAIN, Axis, domain_axis_position
from .. import compileflags
from ..frontend import graph
from . import kernels, numpy_ast

# Type aliases for operation function signatures
type _BinaryOpFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
type _UnaryOpFunc = Callable[[np.ndarray], np.ndarray]
type _ProjectionOpFunc = Callable[[np.ndarray, int], np.ndarray]

_binary_operations: dict[type[graph.BinaryOp], _BinaryOpFunc] = {
    graph.add: np.add,
    graph.subtract: np.subtract,
    graph.multiply: np.multiply,
    graph.divide: np.divide,
    graph.power: np.power,
    graph.equal: np.equal,
    graph.not_equal: np.not_equal,
    graph.less: np.less,
    graph.less_equal: np.less_equal,
    graph.greater: np.greater,
    graph.greater_equal: np.greater_equal,
    graph.logical_and: np.logical_and,
    graph.logical_or: np.logical_or,
    graph.logical_xor: np.logical_xor,
    graph.maximum: np.maximum,
}
"""Maps a binary op in the graph domain to the corresponding numpy operation.

These operations take two arrays as input and produce a single array output,
following NumPy's broadcasting rules for shape compatibility.

Add entries to this table to support more binary operations.
"""


_unary_operations: dict[type[graph.UnaryOp], _UnaryOpFunc] = {
    graph.negate: np.negative,
    graph.logical_not: np.logical_not,
    graph.exp: np.exp,
    graph.log: np.log,
}
"""Maps a unary op in the graph domain to the corresponding numpy operation.

These operations take a single array as input and apply the function
element-wise, producing an output of the same shape.

Add entries to this table to support more unary operations.
"""


def _reduce_sum(x: np.ndarray, axis: int) -> np.ndarray:
    """Sum along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to perform the sum

    Returns
    -------
        Array with the specified axis reduced by summation (keepdims=True)
    """
    return np.sum(x, axis=axis, keepdims=True)


def _reduce_mean(x: np.ndarray, axis: int) -> np.ndarray:
    """Average along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the mean

    Returns
    -------
        Array with the specified axis reduced by averaging (keepdims=True)
    """
    return np.mean(x, axis=axis, keepdims=True)


def _reduce_min(x: np.ndarray, axis: int) -> np.ndarray:
    """Minimum along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the minimum

    Returns
    -------
        Array with the specified axis reduced by minimum (keepdims=True)
    """
    return np.min(x, axis=axis, keepdims=True)


def _reduce_max(x: np.ndarray, axis: int) -> np.ndarray:
    """Maximum along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the maximum

    Returns
    -------
        Array with the specified axis reduced by maximum (keepdims=True)
    """
    return np.max(x, axis=axis, keepdims=True)


def _reduce_std(x: np.ndarray, axis: int) -> np.ndarray:
    """Compute standard deviation along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the standard deviation

    Returns
    -------
        Array with the specified axis reduced by standard deviation (keepdims=True)
    """
    return np.std(x, axis=axis, keepdims=True)


def _reduce_var(x: np.ndarray, axis: int) -> np.ndarray:
    """Variance along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the variance

    Returns
    -------
        Array with the specified axis reduced by variance (keepdims=True)
    """
    return np.var(x, axis=axis, keepdims=True)


def _reduce_median(x: np.ndarray, axis: int) -> np.ndarray:
    """Median along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the median

    Returns
    -------
        Array with the specified axis reduced by median (keepdims=True)
    """
    return np.median(x, axis=axis, keepdims=True)


def _reduce_prod(x: np.ndarray, axis: int) -> np.ndarray:
    """Product along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the product

    Returns
    -------
        Array with the specified axis reduced by product (keepdims=True)
    """
    return np.prod(x, axis=axis, keepdims=True)


def _reduce_any(x: np.ndarray, axis: int) -> np.ndarray:
    """Logical OR along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the logical OR

    Returns
    -------
        Array with the specified axis reduced by logical OR (keepdims=True)
    """
    return np.any(x, axis=axis, keepdims=True)


def _reduce_all(x: np.ndarray, axis: int) -> np.ndarray:
    """Logical AND along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the logical AND

    Returns
    -------
        Array with the specified axis reduced by logical AND (keepdims=True)
    """
    return np.all(x, axis=axis, keepdims=True)


def _reduce_count_nonzero(x: np.ndarray, axis: int) -> np.ndarray:
    """Count non-zero elements along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to count non-zero elements

    Returns
    -------
        Array with the specified axis reduced by counting non-zero (keepdims=True)
    """
    return np.count_nonzero(x, axis=axis, keepdims=True)


def _reduce_quantile(x: np.ndarray, axis: int, q: float) -> np.ndarray:
    """Quantile along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the quantile
        q: The quantile level in [0, 1]

    Returns
    -------
        Array with the specified axis reduced by quantile (keepdims=True)
    """
    return np.quantile(x, q=q, axis=axis, keepdims=True)


_projection_operations: dict[type[graph.Node], _ProjectionOpFunc] = {
    graph.project_using_sum: _reduce_sum,
    graph.project_using_mean: _reduce_mean,
    graph.project_using_min: _reduce_min,
    graph.project_using_max: _reduce_max,
    graph.project_using_std: _reduce_std,
    graph.project_using_var: _reduce_var,
    graph.project_using_median: _reduce_median,
    graph.project_using_prod: _reduce_prod,
    graph.project_using_any: _reduce_any,
    graph.project_using_all: _reduce_all,
    graph.project_using_count_nonzero: _reduce_count_nonzero,
}
"""Maps a projection op in the graph domain to the corresponding numpy operation.

Each operation takes the operand and the numpy dimension to reduce.  That
dimension is resolved per node from the op's *named* axis (see
:func:`_eval_projection_op`), not hard-coded.

Add entries to this table to support more projection operations."""


def _print_graph_node(node: graph.Node, domain_axes: tuple[Axis, ...] = ()) -> None:
    """Print a node before evaluation."""
    # 1. print the original DAG node as a comment so we can always
    # understand what is the specific node leading to this.
    print(f"# {str(node)}")

    # 2. print the numpy equivalent for non-immediate nodes such
    # that we can round-trip the representation.
    if not isinstance(node, (graph.constant, graph.placeholder, graph.array_constant, graph.array_placeholder)):
        print(numpy_ast.graph_node_to_numpy_code(node, domain_axes=domain_axes))


def _print_evaluated_node(node: graph.Node, value: np.ndarray) -> None:
    """Print a node after evaluation."""
    # Throughout this function we try to be very defensive with respect
    # to the node operations. Sometimes, numba returns bare floats rather
    # than `np.ndarray` and this only happens at runtime. This paranoia
    # does not apply to placeholders and constants, for which we provide
    # direct and correct `np.asarray()` initial value assignments.

    # 1. for nodes that are not evaluated, we print their actual
    # value so the representation can round trip.
    if isinstance(node, graph.placeholder):
        print(numpy_ast.graph_node_to_numpy_code(node, value))
    elif isinstance(node, graph.constant):
        print(numpy_ast.graph_node_to_numpy_code(node))

    # 2. print the shape and dtype, which are invaluable when debugging
    if hasattr(value, "shape"):
        print(f"# shape: {value.shape}")
    if hasattr(value, "dtype"):
        print(f"# dtype: {value.dtype}")

    # 3. give the user a sense of the node value for debugging purposes
    print("# value:")
    print("\n".join("# " + line for line in str(value).splitlines()))

    # 4. add an empty line, which is always nice to separate things
    print("")


class NodeValueNotFound(Exception):
    """Raised when a node value is not found in the state."""


class FunctionNotFound(Exception):
    """Raised when a user-defined function is not found in the state."""


class UnsupportedNodeType(Exception):
    """Raised when the executor encounters an unsupported node type."""


class UnsupportedOperation(Exception):
    """Raised when the executor encounters an unsupported operation."""


class PlaceholderValueNotProvided(Exception):
    """Raised when a required placeholder value is not provided in the state."""


@runtime_checkable
class Functor(Protocol):
    """A user-defined callable integrated into the DAG.

    ``output_axes``/``input_axes`` are an optional declared axis signature,
    set via :meth:`NumpyBackend.adapt`. Passing the functor to
    :class:`~..frontend.graph.function_call` as ``functor=`` makes
    ``output_axes`` replace that node's conservative axis-union inference, and
    verifies ``input_axes`` against the actual call arguments at graph-build
    time. Both are ``None`` for an unsigned functor — the common case, which
    keeps the conservative-union behaviour.
    """

    output_axes: tuple[Axis, ...] | None
    input_axes: tuple[tuple[Axis, ...], ...] | None

    def __call__(self, *args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """Execute the user defined function."""
        ...  # pragma: no cover


class _NumpyFunctor:
    """A callable bound to the numpy array convention (internal implementation)."""

    def __init__(
        self,
        fn: Callable[..., np.ndarray],
        *,
        output_axes: tuple[Axis, ...] | None = None,
        input_axes: tuple[tuple[Axis, ...], ...] | None = None,
    ) -> None:
        self._fn = fn
        self.output_axes = output_axes
        self.input_axes = input_axes

    def __call__(self, *args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        return self._fn(*args, **kwargs)


class NumpyBackend:
    """The numpy computation backend.

    Binds user-defined callables to the numpy array convention for use
    with :meth:`~civic_digital_twins.dt_model.simulation.evaluation.Evaluation.evaluate`.

    Example::

        from civic_digital_twins.dt_model import NumpyBackend

        result = evaluation.evaluate(
            ensemble=ens,
            functions={"ts_solve": NumpyBackend.adapt(_ts_solve)},
            backend=NumpyBackend,
        )
    """

    @staticmethod
    def adapt(
        fn: Callable[..., np.ndarray],
        *,
        output_axes: tuple[Axis, ...] | None = None,
        input_axes: tuple[tuple[Axis, ...], ...] | None = None,
    ) -> Functor:
        """Bind *fn* to the numpy array convention.

        The callable must accept and return :class:`numpy.ndarray` values.
        Returns a :class:`Functor` wrapping *fn*.

        *output_axes*/*input_axes* declare an optional axis signature (see
        :class:`Functor`); pass the returned functor to
        ``graph.function_call(..., functor=...)`` to make use of it.
        """
        return _NumpyFunctor(fn, output_axes=output_axes, input_axes=input_axes)


# Belt-and-suspenders: assert at import time that _NumpyFunctor satisfies Functor.
# Pyright catches this statically; the assignment also catches it at runtime if the
# protocol drifts (e.g. signature change) without a corresponding type-checker run.
_: Functor = _NumpyFunctor(lambda *, a, b: np.add(a, b))


@dataclass(frozen=True)
class State:
    """
    The graph executor state.

    Make sure to provide values for placeholder nodes ahead of the evaluation
    by initializing the `values` dictionary accordingly.

    Note that, if compileflags.TRACE is set, the State will print the
    nodes provided to the constructor in its __post_init__ method using
    the `=== begin/end placeholder ===' markers.

    Attributes
    ----------
        values: A dictionary caching the result of the computation.
        flags: Bitmask containing debug flags (e.g., compileflags.BREAK) set
            by default using the `DTMODEL_ENGINE_FLAGS` environement
            variable as documented by the `compileflags` package docs.
        functions: name-keyed user-defined function assignments (implicit,
            evaluate-time binding).
        node_functions: node-identity-keyed function assignments (explicit,
            construction-time binding via ``@functions`` contract).  Checked
            before ``functions`` so that two sub-models sharing the same
            function name each receive their own functor.
        domain_axes: the DOMAIN axes this evaluation carries, in canonical
            (layout) order.  They occupy the trailing numpy dimensions, and
            projections resolve their named axis against this tuple.  Defaults
            to empty (no DOMAIN axes); a time-only model passes ``(TIME_AXIS,)``.

    Notes
    -----
    ``frozen=True`` prevents *attribute reassignment* (e.g.
    ``state.values = new_dict`` raises ``FrozenInstanceError``), giving an
    identity-stability guarantee: any caller that holds a reference to
    ``state.values`` can rely on that reference remaining valid for the
    lifetime of the ``State`` object.  It does **not** make the dict
    itself immutable — ``state.values[node] = result`` works freely and
    is exactly what the executor does throughout evaluation.  This is
    intentional.
    """

    values: dict[graph.Node, np.ndarray]
    flags: int = compileflags.defaults
    functions: dict[str, Functor] = field(default_factory=dict)
    node_functions: dict[graph.Node, Functor] = field(default_factory=dict)
    domain_axes: tuple[Axis, ...] = ()

    def __post_init__(self):
        """Print the placeholder values provided to the constructor."""
        if self.flags & compileflags.TRACE != 0:
            nodes = sorted(self.values.keys(), key=lambda n: n.id)
            for node in nodes:
                _print_graph_node(node, self.domain_axes)
                _print_evaluated_node(node, self.values[node])

    def get_node_value(self, node: graph.Node) -> np.ndarray:
        """Access the value associated with a node.

        Args:
            node: The node whose value to retrieve.

        Returns
        -------
            The value associated with the node.

        Raises
        ------
            NodeValueNotFound: If the node has not been evaluated.
        """
        try:
            return self.values[node]
        except KeyError:
            raise NodeValueNotFound(f"executor: node '{node.name}' has not been evaluated")

    def set_node_value(self, node: graph.Node, value: np.ndarray) -> None:
        """Set the value associated with the given node."""
        self.values[node] = value


def evaluate_nodes(state: State, *nodes: graph.Node) -> np.ndarray | None:
    """Evaluate a list of `graph.Node` using the current `State`.

    This function is syntactic sugar for calling `evaluate_single_node` for each
    node in the given input and then returning the final value.

    This function returns `None` if you do not supply any input node.
    """
    # Honor the DUMP flag when requested to do so
    if state.flags & compileflags.DUMP != 0:
        for node in nodes:
            print(str(node))
        print("")

    # Defer to the internal nodes evaluator
    return _evaluate_nodes(state, *nodes)


def _evaluate_nodes(state: State, *nodes: graph.Node) -> np.ndarray | None:
    rv: np.ndarray | None = None
    for node in nodes:
        rv = evaluate_single_node(state, node)
    return rv


def evaluate_single_node(state: State, node: graph.Node) -> np.ndarray:
    """Evaluate a node given the current state.

    This function assumes you have already linearized the graph. If this
    is not the case, evaluation will fail. Use the `linearize.forest`
    module to ensure the graph is topologically sorted.

    Args:
        state: The current executor state.
        node: The node to evaluate.

    Raises
    ------
        NodeValueNotFound: If a dependent node has not been evaluated
            and therefore its value cannot be found in the state.
        UnsupportedNodeType: If the executor does not support the given node type.
        UnsupportedOperation: If the executor does not support a specific operation.
        PlaceholderValueNotProvided: If a placeholder node has no value provided
            and no default value.
    """
    # 1. check whether node has been already evaluated (note that this
    # covers the case of placeholders provided via the state)
    if node in state.values:
        return state.values[node]

    # 2. check whether we need to trace this node
    flags = node.flags | state.flags
    tracing = flags & compileflags.TRACE
    if tracing:
        _print_graph_node(node, state.domain_axes)

    # 3. evaluate the node
    result = _evaluate(state, node)

    # 4. check whether we need to print the computation result
    if tracing:
        _print_evaluated_node(node, result)

    # 5. check whether we need to stop after evaluating this node
    if flags & compileflags.BREAK != 0:
        input("# executor: press any key to continue...")
        print("")

    # 6. store the node result in the state
    state.values[node] = result

    # 7. return the result
    return result


evaluate = evaluate_single_node
"""Backward-compatible name for evaluate_node."""


def align_to_domain_block(
    arr: np.ndarray,
    node_axes: tuple[Axis, ...],
    domain_axes: tuple[Axis, ...],
) -> np.ndarray:
    """Reshape *arr* so its DOMAIN axes sit at their canonical trailing positions.

    Mid-evaluation, arrays have heterogeneous ranks and numpy aligns them from
    the right.  With a single DOMAIN axis that is harmless — everything that
    carries it has it last.  With several, an array carrying only the *first*
    domain axis would right-align onto the *last* one and silently compute
    nonsense.  So each value is padded to the full domain block, size 1 for the
    axes it does not carry, and permuted into canonical order for those it does.

    Only leaves need this: once every leaf is aligned, broadcasting preserves
    dimension positions, so every computed node is canonically ordered too.
    That is why ``output_axes`` on a computed node is read as a *set* — the
    layout, not the traversal, decides positions.

    Parameters
    ----------
    arr:
        Value whose trailing dimensions are its own DOMAIN axes, in
        *node_axes* order.  Leading dimensions are preserved untouched.
    node_axes:
        The node's own axes; non-DOMAIN entries are ignored.
    domain_axes:
        The evaluation's DOMAIN axes in canonical order.

    Returns
    -------
    *arr* reshaped so its trailing ``len(domain_axes)`` dimensions correspond
    to *domain_axes* positionally.
    """
    own = [ax for ax in node_axes if ax.role == DOMAIN]
    unknown = [ax for ax in own if ax not in domain_axes]
    if unknown:
        raise UnsupportedOperation(
            f"executor: node carries DOMAIN axes {[ax.name for ax in unknown]} that are not among "
            f"this evaluation's axes {[ax.name for ax in domain_axes]} — the layout cannot place them"
        )
    if not domain_axes:
        return arr
    n = len(own)
    leading = arr.shape[: arr.ndim - n]
    ordered = [ax for ax in domain_axes if ax in own]
    if ordered != own:
        base = arr.ndim - n
        arr = np.moveaxis(arr, [base + own.index(ax) for ax in ordered], [base + i for i in range(n)])
    present = arr.shape[arr.ndim - n :] if n else ()
    target: list[int] = []
    consumed = 0
    for ax in domain_axes:
        if consumed < n and ordered[consumed] == ax:
            target.append(present[consumed])
            consumed += 1
        else:
            target.append(1)
    return arr.reshape(leading + tuple(target))


def _eval_array_constant(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.array_constant, node)
    return align_to_domain_block(np.asarray(node.values), node.output_axes, state.domain_axes)


def _eval_array_placeholder_default(_: State, node: graph.Node) -> np.ndarray:
    # Reached only when the state carries no value for this placeholder.
    node = cast(graph.array_placeholder, node)
    raise PlaceholderValueNotProvided(
        f"executor: no value provided for array placeholder '{node.name}' "
        f"over axes {[ax.name for ax in node.output_axes]}"
    )


def _eval_constant_op(_: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.constant, node)
    return np.asarray(node.value)


def _eval_placeholder_default(_: State, node: graph.Node) -> np.ndarray:
    # Note: placeholders are part of the state, so, if we end up
    # here it means we didn't find anything in the state.
    node = cast(graph.placeholder, node)
    if node.default_value is not None:
        return np.asarray(node.default_value)
    raise PlaceholderValueNotProvided(
        f"executor: no value provided for placeholder '{node.name}' and no default value is set"
    )


def _eval_binary_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.BinaryOp, node)
    left = state.get_node_value(node.left)
    right = state.get_node_value(node.right)
    try:
        return _binary_operations[type(node)](left, right)
    except KeyError:
        raise UnsupportedOperation(f"executor: unsupported binary operation: {type(node)}")


def _eval_unary_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.UnaryOp, node)
    operand = state.get_node_value(node.node)
    try:
        return _unary_operations[type(node)](operand)
    except KeyError:
        raise UnsupportedOperation(f"executor: unsupported unary operation: {type(node)}")


def _eval_where_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.where, node)
    return np.where(
        state.get_node_value(node.condition),
        state.get_node_value(node.then),
        state.get_node_value(node.otherwise),
    )


def _eval_multi_clause_where_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.multi_clause_where, node)
    conditions = []
    values = []
    for cond, value in node.clauses:
        conditions.append(state.get_node_value(cond))
        values.append(state.get_node_value(value))
    default = state.get_node_value(node.default_value)
    return np.select(conditions, values, default=default)


def _eval_projection_op(state: State, node: graph.Node) -> np.ndarray:
    """Evaluate a ProjectionOp node, reducing along the node's named axis.

    The axis is resolved to a numpy dimension through
    :func:`~...axes.domain_axis_position`, against the DOMAIN axes this
    evaluation declares (:attr:`State.domain_axes`).  Reducing an axis the
    evaluation does not carry raises rather than silently reducing the wrong
    dimension.
    """
    node = cast(graph.ProjectionOp, node)
    try:
        position = domain_axis_position(state.domain_axes, node.axis)
    except ValueError:
        raise UnsupportedOperation(
            f"executor: numpybackend only supports projection along this evaluation's DOMAIN axes "
            f"{[ax.name for ax in state.domain_axes]}; got {node.axis!r}"
        ) from None
    operand = state.get_node_value(node.node)
    if isinstance(node, graph.project_using_quantile):
        return _reduce_quantile(operand, position, node.q)
    try:
        return _projection_operations[type(node)](operand, position)
    except KeyError:
        raise UnsupportedOperation(f"executor: unsupported projection operation: {type(node)}")


def _eval_axis_op(state: State, node: graph.Node) -> np.ndarray:
    """Evaluate an AxisOp node, operating along the node's named axis.

    Resolves the axis to a numpy dimension exactly as :func:`_eval_projection_op`
    does, but the operation is shape-preserving rather than reducing.
    """
    node = cast(graph.AxisOp, node)
    try:
        position = domain_axis_position(state.domain_axes, node.axis)
    except ValueError:
        raise UnsupportedOperation(
            f"executor: numpybackend only supports axis operations along this evaluation's DOMAIN axes "
            f"{[ax.name for ax in state.domain_axes]}; got {node.axis!r}"
        ) from None
    operand = state.get_node_value(node.node)
    if isinstance(node, graph.shift):
        return kernels.shift(operand, node.periods, axis=position, fill_value=node.fill_value)
    if isinstance(node, graph.roll):
        return np.roll(operand, node.periods, axis=position)
    if isinstance(node, graph.cumulative):
        return np.cumsum(operand, axis=position)
    if isinstance(node, graph.gradient):
        return np.gradient(operand, node.spacing, axis=position)
    raise UnsupportedOperation(f"executor: unsupported axis operation: {type(node)}")


def _eval_laplacian(state: State, node: graph.Node) -> np.ndarray:
    """Evaluate a laplacian node, summing second derivatives along the node's named axes.

    Each axis is resolved to a numpy dimension exactly as :func:`_eval_projection_op`
    does; the operation is shape-preserving, like :func:`_eval_axis_op`.
    """
    node = cast(graph.laplacian, node)
    positions: list[int] = []
    for axis in node.axes:
        try:
            positions.append(domain_axis_position(state.domain_axes, axis))
        except ValueError:
            raise UnsupportedOperation(
                f"executor: numpybackend only supports axis operations along this evaluation's DOMAIN axes "
                f"{[ax.name for ax in state.domain_axes]}; got {axis!r}"
            ) from None
    operand = state.get_node_value(node.node)
    return kernels.laplacian(operand, tuple(positions), node.spacings, node.boundaries)


def _eval_function(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.function_call, node)
    args: list[np.ndarray] = []
    kwargs: dict[str, np.ndarray] = {}
    for arg in node.args:
        args.append(state.get_node_value(arg))
    for key, value in node.kwargs.items():
        kwargs[key] = state.get_node_value(value)
    # Node-identity dispatch (explicit @functions binding) takes priority over name-based dispatch.
    function = state.node_functions.get(node)
    if function is None:
        function = state.functions.get(node.name)
    if function is None:
        raise FunctionNotFound(f"executor: cannot find functor for: {node.name}")
    return function(*args, **kwargs)


def _eval_variant_selector_noop(_state: State, _node: graph.Node) -> np.ndarray:
    """No-op evaluator for variant_selector nodes.

    variant_selector carries structural metadata for _build_plan() and
    produces no runtime value.  The executor stores a sentinel empty array
    so the node is marked as evaluated and skipped if encountered again.
    """
    return np.array([])


_EvaluatorFunc = Callable[[State, graph.Node], np.ndarray]

_evaluators: tuple[tuple[type[graph.Node], _EvaluatorFunc], ...] = (
    (graph.array_constant, _eval_array_constant),
    (graph.array_placeholder, _eval_array_placeholder_default),
    (graph.constant, _eval_constant_op),
    (graph.placeholder, _eval_placeholder_default),
    (graph.BinaryOp, _eval_binary_op),
    (graph.UnaryOp, _eval_unary_op),
    (graph.where, _eval_where_op),
    (graph.MultiClauseOp, _eval_multi_clause_where_op),
    (graph.variant_selector, _eval_variant_selector_noop),
    (graph.ProjectionOp, _eval_projection_op),
    (graph.AxisOp, _eval_axis_op),
    (graph.laplacian, _eval_laplacian),
    (graph.function_call, _eval_function),
)


def _evaluate(state: State, node: graph.Node) -> np.ndarray:
    # Attempt to match with every possible evaluator
    for node_type, evaluator in _evaluators:
        if isinstance(node, node_type):
            return evaluator(state, node)

    # Otherwise, just bail
    raise UnsupportedNodeType(f"executor: unsupported node type: {type(node)}")
