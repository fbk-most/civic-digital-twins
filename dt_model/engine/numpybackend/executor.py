"""
Topologically-Sorted-Graph Executor
===================================

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

from dataclasses import dataclass
from typing import (
    Callable,
    cast,
)

import numpy as np

from ..frontend import graph
from . import debug, dispatch


class NodeValueNotFound(Exception):
    """Raised when a node value is not found in the state."""


class UnsupportedNodeType(Exception):
    """Raised when the executor encounters an unsupported node type."""


class UnsupportedOperation(Exception):
    """Raised when the executor encounters an unsupported operation."""


class PlaceholderValueNotProvided(Exception):
    """Raised when a required placeholder value is not provided in the state."""


@dataclass(frozen=True)
class State:
    """
    The graph executor state.

    Make sure to provide values for placeholder nodes ahead of the evaluation
    by initializing the `values` dictionary accordingly.

    Note that, if graph.NODE_FLAG_TRACE is set, the State will print the
    nodes provided to the constructor in its __post_init__ method.

    Attributes:
        values: A dictionary caching the result of the computation.
        flags: Bitmask containing debug flags (e.g., graph.NODE_FLAG_BREAK).
    """

    values: dict[graph.Node, np.ndarray]
    flags: int = 0

    def __post_init__(self):
        if self.flags & graph.NODE_FLAG_TRACE != 0:
            nodes = sorted(self.values.keys(), key=lambda n: n.id)
            # TODO(bassosimone): make it clear that those nodes are cached
            for node in nodes:
                debug.print_graph_node(node)
                debug.print_evaluated_node(self.values[node], cached=True)

    def get_node_value(self, node: graph.Node) -> np.ndarray:
        """Helper function to access the value associated with a node.

        Args:
            node: The node whose value to retrieve.

        Returns:
            The value associated with the node.

        Raises:
            NodeValueNotFound: If the node has not been evaluated.
        """
        try:
            return self.values[node]
        except KeyError:
            raise NodeValueNotFound(f"executor: node '{node.name}' has not been evaluated")


def evaluate(state: State, node: graph.Node) -> np.ndarray:
    """
    Evaluates a node assuming that all dependent nodes have already
    been evaluated and cached in the state. In other words, this
    function assumes you have already linearized the graph. If this
    is not the case, evaluation will fail. Use the `frontend.linearize`
    module to ensure the graph is topologically sorted.

    Args:
        state: The current executor state.
        node: The node to evaluate.

    Raises:
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
    tracing = flags & graph.NODE_FLAG_TRACE
    if tracing:
        debug.print_graph_node(node)

    # 3. evaluate the node proper
    result = _evaluate(state, node)

    # 4. check whether we need to print the computation result
    if tracing:
        debug.print_evaluated_node(result, cached=False)

    # 5. check whether we need to stop after evaluating this node
    if flags & graph.NODE_FLAG_BREAK != 0:
        input("executor: press any key to continue...")
        print("")

    # 6. store the node result in the state
    state.values[node] = result

    # 7. return the result
    return result


def _eval_constant_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.constant, node)
    return np.asarray(node.value)


def _eval_placeholder_default(state: State, node: graph.Node) -> np.ndarray:
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
        return dispatch.binary_operations[type(node)](left, right)
    except KeyError:
        raise UnsupportedOperation(f"executor: unsupported binary operation: {type(node)}")


def _eval_unary_op(state: State, node: graph.Node) -> np.ndarray:
    node = cast(graph.UnaryOp, node)
    operand = state.get_node_value(node.node)
    try:
        return dispatch.unary_operations[type(node)](operand)
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
        return dispatch.axes_operations[type(node)](operand, node.axis)
    except KeyError:
        raise UnsupportedOperation(f"executor: unsupported axis operation: {type(node)}")


_EvaluatorFunc = Callable[[State, graph.Node], np.ndarray]

_evaluators: tuple[tuple[type[graph.Node], _EvaluatorFunc], ...] = (
    (graph.constant, _eval_constant_op),
    (graph.placeholder, _eval_placeholder_default),
    (graph.BinaryOp, _eval_binary_op),
    (graph.UnaryOp, _eval_unary_op),
    (graph.where, _eval_where_op),
    (graph.multi_clause_where, _eval_multi_clause_where_op),
    (graph.AxisOp, _eval_axis_op),
)


def _evaluate(state: State, node: graph.Node) -> np.ndarray:
    # Attempt to match with every possible evaluator
    for node_type, evaluator in _evaluators:
        if isinstance(node, node_type):
            return evaluator(state, node)

    # Otherwise, just bail
    raise UnsupportedNodeType(f"executor: unsupported node type: {type(node)}")
