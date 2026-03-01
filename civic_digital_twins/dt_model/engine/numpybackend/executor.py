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

import types
from dataclasses import dataclass, field
from typing import (
    Callable,
    Protocol,
    TypeAlias,
    cast,
    runtime_checkable,
)

import numpy as np

from .. import compileflags
from ..frontend import forest, graph, ir
from . import numpy_ast

# Type aliases for operation function signatures
_BinaryOpFunc: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]
_UnaryOpFunc: TypeAlias = Callable[[np.ndarray], np.ndarray]
_AxisOpFunc: TypeAlias = Callable[[np.ndarray, graph.Axis], np.ndarray]

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


def _expand_dims(x: np.ndarray, axis: graph.Axis) -> np.ndarray:
    """Expand input array with a new axis at the specified position.

    Args:
        x: The input array to expand
        axis: The position where the new axis is placed

    Returns
    -------
        Array with the expanded dimension
    """
    return np.expand_dims(x, axis)


def _reduce_sum(x: np.ndarray, axis: graph.Axis) -> np.ndarray:
    """Sum along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to perform the sum

    Returns
    -------
        Array with the specified axis reduced by summation (keepdims=True)
    """
    return np.sum(x, axis=axis, keepdims=True)


def _reduce_mean(x: np.ndarray, axis: graph.Axis) -> np.ndarray:
    """Average along the specified axis, keeping the reduced axis as size 1.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the mean

    Returns
    -------
        Array with the specified axis reduced by averaging (keepdims=True)
    """
    return np.mean(x, axis=axis, keepdims=True)


_axes_operations: dict[type[graph.AxisOp], _AxisOpFunc] = {
    graph.expand_dims: _expand_dims,
    graph.project_using_sum: _reduce_sum,
    graph.project_using_mean: _reduce_mean,
}
"""Maps an axis op in the graph domain to the corresponding numpy operation.

These operations take an array and an axis parameter, performing
transformations that affect the array's dimensionality or reduce values
along the specified axis.

Add entries to this table to support more axis operations."""


def _print_graph_node(node: graph.Node) -> None:
    """Print a node before evaluation."""
    # 1. print the original DAG node as a comment so we can always
    # understand what is the specific node leading to this.
    print(f"# {str(node)}")

    # 2. print the numpy equivalent for non-immediate nodes such
    # that we can round-trip the representation.
    if not isinstance(
        node, (graph.constant, graph.placeholder, graph.timeseries_constant, graph.timeseries_placeholder)
    ):
        print(numpy_ast.graph_node_to_numpy_code(node))


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
    """A user-defined callable integrated into the DAG."""

    def __call__(self, *args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """Execute the user defined function."""
        ...  # pragma: no cover


class LambdaAdapter:
    """Adapter that transforms a Callable into a Functor."""

    def __init__(self, callable: Callable[..., np.ndarray]) -> None:
        self.callable = callable

    def __call__(self, *args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """Execute the wrapped callable with the given arguments."""
        return self.callable(*args, **kwargs)


_: Functor = LambdaAdapter(lambda *, a, b: np.add(a, b))


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
        functions: user-defined functions assignments.
        jitted: cache of JIT-compiled tree functions; populated lazily when
            ``compileflags.JIT`` is set.

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
    jitted: dict[forest.Tree, types.FunctionType] = field(default_factory=dict)

    def __post_init__(self):
        """Print the placeholder values provided to the constructor."""
        if self.flags & compileflags.TRACE != 0:
            nodes = sorted(self.values.keys(), key=lambda n: n.id)
            for node in nodes:
                _print_graph_node(node)
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
        _print_graph_node(node)

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


def _eval_timeseries_constant(_: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.timeseries_constant, node)
    return np.asarray(node.values)


def _eval_timeseries_placeholder_default(_: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.timeseries_placeholder, node)
    raise PlaceholderValueNotProvided(f"executor: no value provided for timeseries placeholder '{node.name}'")


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


def _eval_axis_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.AxisOp, node)
    operand = state.get_node_value(node.node)
    try:
        return _axes_operations[type(node)](operand, node.axis)
    except KeyError:
        raise UnsupportedOperation(f"executor: unsupported axis operation: {type(node)}")


def _eval_function(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.function_call, node)
    args: list[np.ndarray] = []
    kwargs: dict[str, np.ndarray] = {}
    for arg in node.args:
        args.append(state.get_node_value(arg))
    for key, value in node.kwargs.items():
        kwargs[key] = state.get_node_value(value)
    try:
        function = state.functions[node.name]
    except KeyError:
        raise FunctionNotFound(f"executor: cannot find functor for: {node.name}")
    return function(*args, **kwargs)


_EvaluatorFunc = Callable[[State, graph.Node], np.ndarray]

_evaluators: tuple[tuple[type[graph.Node], _EvaluatorFunc], ...] = (
    (graph.timeseries_constant, _eval_timeseries_constant),
    (graph.timeseries_placeholder, _eval_timeseries_placeholder_default),
    (graph.constant, _eval_constant_op),
    (graph.placeholder, _eval_placeholder_default),
    (graph.BinaryOp, _eval_binary_op),
    (graph.UnaryOp, _eval_unary_op),
    (graph.where, _eval_where_op),
    (graph.multi_clause_where, _eval_multi_clause_where_op),
    (graph.AxisOp, _eval_axis_op),
    (graph.function_call, _eval_function),
)


def _evaluate(state: State, node: graph.Node) -> np.ndarray:
    # Attempt to match with every possible evaluator
    for node_type, evaluator in _evaluators:
        if isinstance(node, node_type):
            return evaluator(state, node)

    # Otherwise, just bail
    raise UnsupportedNodeType(f"executor: unsupported node type: {type(node)}")


# === Tree / DAG evaluation (experimental JIT layer) ===


def _evaluate_single_tree_jit(state: State, tree: forest.Tree) -> np.ndarray:
    root = tree.root()
    if root in state.values:
        return state.values[root]

    flags = root.flags | state.flags
    tracing = flags & compileflags.TRACE
    if tracing:
        _print_graph_node(root)

    func = state.jitted.get(tree)
    if not func:
        func = numpy_ast.compile_function(tree)
        state.jitted[tree] = func

    result = numpy_ast.call_function(state, tree, func)

    if tracing:
        _print_evaluated_node(root, result)

    if flags & compileflags.BREAK != 0:
        input("# executor: press any key to continue...")
        print("")

    state.values[root] = result
    return result


def evaluate_single_tree(state: State, tree: forest.Tree) -> np.ndarray:
    """Evaluate a ``forest.Tree`` using the current ``State``.

    Constants embedded in the tree body are evaluated first so that their
    values are available to the JIT-compiled function.  If ``compileflags.JIT``
    is set *and* no user-defined functions are present, the tree is compiled
    with Numba and called directly; otherwise the body nodes are evaluated
    one-by-one via :func:`evaluate_single_node`.

    Parameters
    ----------
    state:
        Current executor state.
    tree:
        The tree to evaluate.

    Returns
    -------
        The value of the tree's root node.
    """
    for node in tree.body:
        if isinstance(node, graph.constant):
            evaluate_single_node(state, node)

    if not state.functions and state.flags & compileflags.JIT != 0:
        return _evaluate_single_tree_jit(state, tree)

    rv = _evaluate_nodes(state, *tree.body)
    assert rv is not None  # tree body is non-empty by invariant
    return rv


def evaluate_trees(state: State, *trees: forest.Tree) -> np.ndarray | None:
    """Evaluate a sequence of ``forest.Tree`` objects using the current ``State``.

    Parameters
    ----------
    state:
        Current executor state.
    trees:
        Trees to evaluate in topological order.

    Returns
    -------
        The value of the last tree's root node, or ``None`` if no trees
        were provided.
    """
    rv: np.ndarray | None = None
    for tree in trees:
        rv = evaluate_single_tree(state, tree)
    return rv


def evaluate_dag(state: State, dag: ir.DAG) -> np.ndarray | None:
    """Evaluate an ``ir.DAG`` using the current ``State``.

    Placeholders and constants in the DAG are evaluated first; the DAG
    body (trees *or* flat nodes, never both) is evaluated next.

    Parameters
    ----------
    state:
        Current executor state.  Placeholder values must already be set.
    dag:
        The IR DAG to evaluate.

    Returns
    -------
        The value of the last root node, or ``None`` for an empty DAG.
    """
    for node in dag.placeholders:
        evaluate_single_node(state, node)
    for node in dag.constants:
        evaluate_single_node(state, node)

    if dag.trees:
        return evaluate_trees(state, *dag.trees)
    return evaluate_nodes(state, *dag.nodes)
