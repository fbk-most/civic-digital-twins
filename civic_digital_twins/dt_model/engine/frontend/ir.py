"""Intermediate DAG representation.

The intermediate representation (IR) of the DAG consists of:

1. a possibly-empty list of placeholders;

2. a possibly-empty list of constants;

3. a possibly-empty list of trees;

4. a possibly-empty linear list of nodes to evaluate.

The intermediate DAG representation unifies the two possible
representations you can obtain with the frontend:

1. the topologically sorted list of nodes;

2. the topologically sorted list of trees.

As such, this data structure abstract over the mechanism
used to lower the data and provides a uniform representation
for downstream backends.

What's more, because we explicitly single out placeholders
and constants, we simplify the job of a backend that does not
support strings and needs to remap unique strings to unique
integers. NumPy does not have this issue, however, for
example, it seems PyTorch may not be able to deal with strings.

See https://github.com/fbk-most/civic-digital-twins/issues/84#issuecomment-3129629098
for a mode detail discussion explaining how we could go about
implementing string->int remapping for backends that don't have
support for dealing with strings directly.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from . import forest, graph


@dataclass(frozen=True)
class DAG:
    """Intermediate DAG representation.

    Attributes
    ----------
    placeholders: list of placeholder dependencies for the DAG.
    constants: list of constant dependencies for the DAG.
    trees: topologically sorted list of trees.
    nodes: topologically sorted list of nodes.

    Note that, depending on how the original graph DAG was lowered,
    either trees or nodes are empty. This invariant is checked
    by the __post_init__ hook to ensure it is true.

    Also note that placeholders and constants are sorted by their
    `.id` attribute, which depends on the order in which these
    nodes were introduced inside the original source DAG.
    """

    placeholders: list[graph.placeholder]
    constants: list[graph.constant]
    trees: list[forest.Tree]
    nodes: list[graph.Node]

    def __post_init__(self) -> None:
        """Check invariants after initialization."""
        if self.trees and self.nodes:
            raise RuntimeError("self.trees and self.nodes cannot both be non-empty")

    def __repr__(self) -> str:
        """Return a round-trippable representation of the DAG IR."""
        lines = []

        if self.placeholders:
            lines.append("# === placeholders ===")
            for placeholder in self.placeholders:
                lines.append(repr(placeholder))
            lines.append("")

        if self.constants:
            lines.append("# === constants ===")
            for constant in self.constants:
                lines.append(repr(constant))
            lines.append("")

        if self.trees:
            lines.append("# === trees ===")
            for tree in self.trees:
                lines.append(repr(tree))
            lines.append("")

        if self.nodes:
            lines.append("# === nodes ===")
            for node in self.nodes:
                lines.append(repr(node))
            lines.append("")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a round-trippable representation of the DAG IR."""
        return repr(self)


def compile_nodes(*input_nodes: graph.Node) -> DAG:
    """Transform topologically-sorted nodes into a DAG IR.

    This function classifies the input nodes by their type obtaining:

    1. placeholders
    2. constants
    3. other nodes (end up in the nodes)

    The input is a topologically sorted list of nodes obtained
    by calling the `linearize.forest` function.

    The placeholders and constants are sorted by their `.id`.
    """
    # 1. create the variables where to store nodes according to their type
    placeholders: set[graph.placeholder] = set()
    constants: set[graph.constant] = set()
    nodes: list[graph.Node] = []

    # 2. classify nodes according to their type while preserving order
    for node in input_nodes:
        if isinstance(node, graph.placeholder):
            placeholders.add(node)
            continue

        if isinstance(node, graph.constant):
            constants.add(node)
            continue

        if isinstance(node, graph.Node):
            nodes.append(node)
            continue

        # should not happen; let's test this is actually true
        raise RuntimeError(f"Expected graph.Node, got {type(node)}")

    # 3. prepare the result DAG IR making sure that we sort the
    # constants and placeholders by their insertion order to
    # ensure that the program output is stable.
    return DAG(
        placeholders=sorted(placeholders, key=lambda x: x.id),
        constants=sorted(constants, key=lambda x: x.id),
        trees=list(),
        nodes=nodes,
    )


def compile_trees(*trees: forest.Tree) -> DAG:
    """Transform topologically sorted trees into a DAG IR.

    This function classifies the input trees obtaining:

    1. all the placeholders
    2. all the constants
    3. topologically sorted trees

    The input is a topologically sorted list of trees obtained
    by calling the `forest.partition` function.

    The placeholders and constants are sorted by their `.id`.
    """
    # 1. create the variables where to store nodes according to their type
    placeholders: set[graph.placeholder] = set()
    constants: set[graph.constant] = set()

    # 2. fill the placeholders and the constants making sure
    # we're not counting any of them more than once.
    for tree in trees:
        for node in tree.inputs:
            if isinstance(node, graph.placeholder):
                placeholders.add(node)
                continue

            if isinstance(node, graph.constant):
                constants.add(node)
                continue

            if isinstance(node, graph.Node):  # dependency on another tree
                continue

            # Note: forest.Tree has a very good post init, so it's not
            # so easy to sneak in wrong types. Hence, for now, there does
            # not seem a need to add assetions or exceptions here.

    # 3. prepare the result DAG IR making sure that we sort the
    # constants and placeholders by their insertion order to
    # ensure that the program output is stable.
    return DAG(
        placeholders=sorted(placeholders, key=lambda x: x.id),
        constants=sorted(constants, key=lambda x: x.id),
        trees=list(trees),
        nodes=list(),
    )
