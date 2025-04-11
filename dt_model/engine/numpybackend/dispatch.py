"""Dispatch tables mapping frontend graph operations to numpy operations.

These tables are used by the evaluator to map symbolic operations in the graph
to their concrete NumPy implementations. To extend the system with new operations,
simply add entries to the appropriate dispatch table.
"""

from typing import Callable, TypeAlias

import numpy as np

from ..frontend import graph

# Type aliases for operation function signatures
BinaryOpFunc: TypeAlias = Callable[[np.ndarray, np.ndarray], np.ndarray]
UnaryOpFunc: TypeAlias = Callable[[np.ndarray], np.ndarray]
AxisOpFunc: TypeAlias = Callable[[np.ndarray, graph.Axis], np.ndarray]

binary_operations: dict[type[graph.BinaryOp], BinaryOpFunc] = {
    graph.add: np.add,
    graph.subtract: np.subtract,
    graph.multiply: np.multiply,
    graph.divide: np.divide,
    graph.equal: np.equal,
    graph.not_equal: np.not_equal,
    graph.less: np.less,
    graph.less_equal: np.less_equal,
    graph.greater: np.greater,
    graph.greater_equal: np.greater_equal,
    graph.logical_and: np.logical_and,
    graph.logical_or: np.logical_or,
    graph.logical_xor: np.logical_xor,
    graph.power: np.power,
    graph.maximum: np.maximum,
}
"""Maps a binary op in the graph domain to the corresponding numpy operation.

These operations take two arrays as input and produce a single array output,
following NumPy's broadcasting rules for shape compatibility.

Add entries to this table to support more binary operations.
"""


unary_operations: dict[type[graph.UnaryOp], UnaryOpFunc] = {
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
    """Reduce an array by summing along the specified axis.

    Args:
        x: The input array to reduce
        axis: The axis along which to perform the sum

    Returns
    -------
        Array with the specified axis reduced by summation
    """
    return np.sum(x, axis=axis)


def _reduce_mean(x: np.ndarray, axis: graph.Axis) -> np.ndarray:
    """Reduce an array by computing the mean along the specified axis.

    Args:
        x: The input array to reduce
        axis: The axis along which to compute the mean

    Returns
    -------
        Array with the specified axis reduced by averaging
    """
    return np.mean(x, axis=axis)


axes_operations: dict[type[graph.AxisOp], AxisOpFunc] = {
    graph.expand_dims: _expand_dims,
    graph.reduce_sum: _reduce_sum,
    graph.reduce_mean: _reduce_mean,
}
"""Maps an axis op in the graph domain to the corresponding numpy operation.

These operations take an array and an axis parameter, performing
transformations that affect the array's dimensionality or reduce values
along the specified axis.

Add entries to this table to support more axis operations."""
