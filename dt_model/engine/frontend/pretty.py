"""
Pretty Printing for Computation Graphs
======================================

This module will provide facilities for converting computation graphs into
readable string representations. It is currently a stub.
"""

# SPDX-License-Identifier: Apache-2.0

from . import graph


def format(node: graph.Node) -> str:
    """Format a computation graph node as a string.

    Args:
        node: The node to format

    Returns:
        A string representation with appropriate parentheses
        and operator precedence.

    Examples:
        >>> x = graph.placeholder("x")
        >>> y = graph.add(graph.multiply(x, 2), 1)
        >>> print(pretty.format(y))
        x * 2 + 1
    """
    # TODO(bassosimone): add yakof code here
    return ""
