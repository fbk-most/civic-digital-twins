"""
Debugging Code
==============

This module contains debugging code.
"""

import numpy as np

from ..frontend import graph, pretty


def print_graph_node(node: graph.Node) -> None:
    """Print a node within the computation graph."""
    print("=== begin tracepoint ===")
    print(f"name: {node.name}")
    print(f"id: {node.id}")
    print(f"type: {node.__class__}")
    print(f"formula: {pretty.format(node)}")


def print_evaluated_node(value: np.ndarray, cached: bool = False) -> None:
    """Print a node after evaluation."""
    print(f"shape: {value.shape}")
    print(f"cached: {cached}")
    print(f"value:\n{value}")
    print("=== end tracepoint ===")
    print("")
