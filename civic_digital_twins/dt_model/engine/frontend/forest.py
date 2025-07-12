"""
Generates a partitioned forest from a graph and graph leaf nodes.

A graph is a DAG representing a computation. You construct is using the
`graph` module in this package or other higher-level modules.

The graph leaf nodes is the set of leaf nodes to evaluate.

Given a graph and the leaf nodes, this module transforms it into a
forest (i.e., a list of `Tree`). Each leaf node becomes a `Tree`
returned in output. Each `Tree` contains:

    1. inputs: the set of input nodes used by the tree

    2. nodes: the topologically sorted list of nodes to compute the root node

The `Tree.nodes` attribute always contains at least the leaf node, which
you can access using the `Tree.root` method.

Note that this module uses the `linearize` module internally. You don't need
to use linearize if you are already using this module.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from . import graph as g0
from . import linearize, pretty


@dataclass(frozen=True)
class Tree:
    """Tree in the compiled graph.

    Contains the root node, the inputs required to compute the value of
    the root node, and the topologically sorted nodes to evaluate to obtain
    the value of the root node.
    """

    # Inputs required to compute the root
    inputs: set[g0.Node]

    # Topologically sorted nodes.
    nodes: list[g0.Node]

    def root(self) -> g0.Node:
        """Return the root node of the tree."""
        assert len(self.nodes) > 0
        return self.nodes[-1]

    def __hash__(self) -> int:
        """Override hashing to use the root node hash."""
        return hash(self.root())

    def format(self) -> str:
        """Format returns a string representation of the tree."""
        lines: list[str] = []
        root = self.root()
        for node in self.nodes:
            line: list[str] = []
            line.append(f"#{node.id} ")
            if node in self.inputs:
                line.append("<INPUT> ")
            elif node is root:
                line.append("<OUTPUT> ")
            else:
                line.append("<INTERNAL> ")
            line.append(pretty.format(node))
            lines.append("".join(line))
        return "\n".join(lines)


def partition(*leaves_tuple: g0.Node) -> list[Tree]:
    """Transform a list of graph leaves into a forest.

    The leaves are the nodes to compute. Each leave becomes the root of
    a tree that contains the topologically nodes to evaluate the leave
    value. Nodes computed by other trees will not be computed by a tree
    and are listed as required inputs to evaluate the tree. The relevant
    placeholders are also listed as required inputs.
    """
    # 1. ensure that we use the insertion order sorting
    leaves = sorted(list(leaves_tuple), key=lambda x: x.id)

    # 2. be prepared for collecting all the trees
    trees: list[Tree] = []

    # 3. ensure that we know all the leaves in advance
    leaves_set: set[g0.Node] = set(leaves)

    # 4. process each leave independently
    for leaf in leaves:
        # 4.1. get the topological sorting for the leave
        nodes = linearize.forest(leaf, stopat=leaves_set - set([leaf]))

        # 4.2. prepare to collect inputs
        inputs: set[g0.Node] = set()

        # 4.3. add as inputs all placeholders and all the
        # leaves in the sorting except the current one
        for node in nodes:
            if isinstance(node, g0.placeholder):
                inputs.add(node)
                continue
            if node is not leaf and node in leaves_set:
                inputs.add(node)
                continue

        # 4.4. check some invariants; note that nodes must
        # always contain at least the leaf node
        assert len(nodes) > 0 and nodes[-1] is leaf

        # 4.4. create the tree for the node
        trees.append(Tree(inputs, nodes))

    # 5. return the forest
    return trees
