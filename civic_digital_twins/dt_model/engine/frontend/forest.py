"""Partition the DAG into a forest of trees.

Given a set of root nodes, this module overlays a forest of trees
on top of the DAG and returns the trees in topological order.

Note that this module uses `linearize` internally. You don't need
to use `linearize` if you are already using this module.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from . import graph, linearize


def _node_type_repr(node: graph.Node) -> str:
    """Return one of graph.placeholder, graph.constant, and graph.Node."""
    return (
        "graph.placeholder"
        if isinstance(node, graph.placeholder)
        else "graph.constant"
        if isinstance(node, graph.constant)
        else "graph.Node"
    )


@dataclass(frozen=True)
class Tree:
    """Tree representing the computation of a root node.

    Tree Identity and Equality
    --------------------------

    A tree uses as hashing identity the identity of its root node and the
    equality operator checks for root node identity equality.

    Attributes
    ----------
    inputs: input nodes required to compute the root node value
        guaranteed to be sorted by increasing node ID.

    body: topologically sorted nodes to compute the root node value.
    """

    inputs: list[graph.Node]
    body: list[graph.Node]

    def __post_init__(self) -> None:
        """Ensure that invariants are respected."""
        # Make sure the tree is well formed
        assert len(self.body) > 0

        # Make sure inputs are sorted by increasing node ID
        assert sorted(self.inputs, key=lambda x: x.id) == self.inputs

    def root(self) -> graph.Node:
        """Return the root node of the tree."""
        return self.body[-1]

    def __hash__(self) -> int:
        """Override hashing to use the root node hash."""
        return hash(self.root())

    def __eq__(self, other: Any) -> bool:
        """Implement equality operator using hashing equality."""
        if not isinstance(other, Tree):
            return NotImplemented
        return self.root() is other.root()

    def __repr__(self) -> str:
        """Return a round-trippable Python function representation of the tree."""
        # Be prepared for assembling lines
        lines: list[str] = []

        # Format the function name
        root = self.root()
        inputs = ", ".join(f"n{input.id}: {_node_type_repr(input)}" for input in self.inputs)
        lines.append(f"def t{root.id}({inputs}) -> graph.Node:")

        # Format the function body
        for node in self.body:
            lines.append("    " + str(node))

        # Format the return statement
        lines.append(f"    return n{root.id}")

        # Add an empty line
        lines.append("")

        # Assemble the whole function code
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a round-trippable Python function representation of the tree."""
        return repr(self)


def partition(*roots: graph.Node) -> list[Tree]:
    """Partition the graph into a set of independent trees.

    Returns one tree per root node, where each tree contains the minimal
    subgraph required to compute its root. The trees are topologically
    sorted such that all dependencies are satisfied at evaluation time.
    """
    # 1. partition without topological sorting
    unsorted = _partition(*roots)

    # 2. be prepared for returning topologically sorted trees
    trees: list[Tree] = []

    # 3. be prepared to know the trees we've processed
    processed: set[Tree] = set()

    # 4. track the trees that we're working on
    work = deque(unsorted)

    # 5. iterate until we're out of trees
    while work:
        # 5.1. extract the tree currently at the front of the list
        tree = work.popleft()

        # 5.2. check whether all dependencies are satisfied skipping the
        # placeholders since they're satisfied by definition
        satisfied = all(
            dep in processed if not isinstance(dep, (graph.placeholder, graph.constant)) else True
            for dep in tree.inputs
        )

        # 5.3. if not, put the tree at the back.
        #
        # Note: in principle this algorithm loops forever but the input
        # is a DAG so the algorithm will always converge.
        if not satisfied:
            work.append(tree)
            continue

        # 5.4. finish processing the tree
        processed.add(tree)
        trees.append(tree)

    # 6. Make sure invariants hold after processing
    assert len(processed) == len(unsorted)

    # 7. return the topologically sorted trees
    return trees


def _partition(*roots: graph.Node) -> list[Tree]:
    # 1. be prepared for collecting all the trees
    trees: list[Tree] = []

    # 2. ensure that we know all the roots in advance
    rootset: set[graph.Node] = set(roots)

    # 3. process each root independently
    for root in roots:
        # 3.1. compute the boundary where to stop visiting
        boundary = rootset - {root}

        # 3.2. get the topological sorting for the current root
        allnodes = linearize.forest(root, boundary=boundary)

        # 3.3. prepare to collect unique inputs
        unique_inputs: set[graph.Node] = set()

        # 3.4. prepare the collect the "body"
        body: list[graph.Node] = []

        # 3.5. distinguish between inputs and "body" ensuring that
        # placeholders and constants are considered inputs.
        for node in allnodes:
            if node in boundary or isinstance(node, (graph.placeholder, graph.constant)):
                unique_inputs.add(node)
                continue
            body.append(node)

        # 3.6. ensure that inputs are sorted by increasing ID
        sorted_inputs = sorted(unique_inputs, key=lambda x: x.id)

        # 3.7. check some invariants; note that nodes must
        # always contain at least the root node
        assert len(body) > 0 and body[-1] is root

        # 3.8. create the tree for the node
        trees.append(Tree(inputs=sorted_inputs, body=body))

    # 4. return the forest
    return trees
