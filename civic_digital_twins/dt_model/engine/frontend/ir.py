"""Intermediate DAG representation.

The intermediate representation of the DAG consists of:

1. a possibly-empty list of placeholders;

2. a possibly-empty list of constants;

3. a possibly-empty list of trees;

4. a possibly-empty linear program to evaluate.

The input for producing intermediate representation is the
topological sorting of the graph nodes.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from . import graph, forest


@dataclass(frozen=True)
class DAG:
    """Intermediate DAG representation."""

    placeholders: list[graph.placeholder]
    constants: list[graph.constant]
    trees: list[forest.Tree]
    program: list[graph.Node]

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

        if self.program:
            lines.append("# === main ===")
            for node in self.program:
                lines.append(repr(node))
            lines.append("")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a round-trippable representation of the DAG IR."""
        return repr(self)


def compile_linear(*nodes: graph.Node) -> DAG:
    """Transform topologically-sorted nodes into a DAG IR.

    This function does not reorganize the code in trees and just
    ensures that we classify them by their type:

    1. placeholders
    2. constants
    3. other nodes
    """
    # 1. create the variables where to store nodes according to their type
    placeholders: list[graph.placeholder] = []
    constants: list[graph.constant] = []
    program: list[graph.Node] = []

    # 2. classify nodes according to their type while preserving order
    for node in nodes:
        if isinstance(node, graph.placeholder):
            placeholders.append(node)
            continue
        if isinstance(node, graph.constant):
            constants.append(node)
            continue
        if isinstance(node, graph.Node):
            program.append(node)
            continue
        raise RuntimeError(f"Expected graph.Node, got {type(node)}")

    # 3. prepare the result DAG IR
    return DAG(
        placeholders=placeholders,
        constants=constants,
        trees=list(),
        program=program,
    )


def compile_trees(*roots: graph.Node) -> DAG:
    """Create an IR DAG computing the given roots.

    The algorithm is roughly the following:

    1. obtain topologically sorted trees using the forest module

    2. extract constants and placeholders so that they do not
    appear inside the code of each tree

    The resulting DAG program will feature a tree for each root
    provided in input and consist of a very small program in which
    we just evaluate the trees as functions and combine their
    results to compute the desired root values.
    """
    # 1. create the variables where to store nodes according to their type
    placeholders: set[graph.placeholder] = set()
    constants: set[graph.constant] = set()
    trees: list[forest.Tree] = []
    program: list[graph.Node] = []

    # 2. obtain the trees respecting their topological sorting
    trees.extend(forest.partition(*roots))

    # 3. fill the placeholders and the constants making sure
    # we're not counting any of them more than once. Note that
    # constants and placeholders are inputs, so their order
    # is not particularly relevant. Yet, we will want to sort
    # later on to make the output predictable.
    for tree in trees:
        for input in tree.inputs:
            if isinstance(input, graph.placeholder):
                placeholders.add(input)
            if isinstance(input, graph.constant):
                constants.add(input)

    # 4. construct the program
    # TODO(bassosimone): figure out how to do this
    _ = program

    # 5. prepare the result DAG IR making sure that we sort the
    # constants and placeholders by their insertion order.
    return DAG(
        placeholders=sorted(placeholders, key=lambda x: x.id),
        constants=sorted(constants, key=lambda x: x.id),
        trees=trees,
        program=program,
    )
