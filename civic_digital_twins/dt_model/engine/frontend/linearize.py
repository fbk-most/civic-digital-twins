"""Provide functions to linearize computation graphs into execution plans.

This module performs topological sorting of graph nodes, ensuring dependencies
are evaluated before the nodes that depend on them.

While we could directly rely on the node's creation ID to perform sorting on
the graph, relying on topological sorting makes the code slightly more robust
because it allows us to detect loops at sorting time.

The linearization process:
1. Starts from output nodes and traverses the graph
2. Ensures all dependencies are scheduled before their dependents
3. Handles common graph structures (conditionals, etc.)

Note that output nodes are also called root nodes because they are seen
as the roots of evaluation trees, while the DAG is a forest of trees.

This is useful for:
- Creating efficient execution plans for evaluators
- Visualizing the computation flow in order
- Debugging models by inspecting operations in a logical sequence

Note on Node Identity
---------------------
Because nodes override equality operators to build computation graphs,
finding nodes within the execution plan must use object identity (`is`),
not equality comparison (`==`).
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from . import graph


def forest(*roots: graph.Node, boundary: set[graph.Node] | None = None) -> list[graph.Node]:
    """
    Linearize a computation forest (multiple output nodes) into a sequence.

    The computation forest is an overlay abstraction created over a DAG where the
    given roots identify each tree within the forest itself.

    While the roots are typically the outputs computed by the DAG, they
    can actually be any node within the DAG itself.

    Evaluating the returned nodes in order ensures that you always evaluate
    node A before node B when node B depends on node A.

    Args:
        *roots: the nodes to start the linearization process from. Use the
            unpacking operator `*` to pass a list of nodes.

        boundary: stop visiting when reaching nodes in the boundary set.

    Returns
    -------
        Topologically sorted list of nodes forming an execution plan

    Raises
    ------
        ValueError: If a cycle is detected in the graph
        TypeError: If an unknown node type is encountered

    Examples
    --------
        >>> # Single output
        >>> plan = linearize.forest(output_node)
        >>>
        >>> # Multiple outputs
        >>> plan = linearize.forest(output1, output2, output3)
        >>>
        >>> # List of outputs
        >>> plan = linearize.forest(*output_list)

    Note on equality
    ----------------

    Because nodes override their equality operators to build computation
    graphs, finding nodes within the execution plan must use object identity, as
    documented more extensively in the `graph` and in this module's docstring.
    """
    # plan contains the linearized output
    plan: list[graph.Node] = []

    # visiting allows to detect cycles when visiting nodes
    visiting: set[graph.Node] = set()

    # visited caches the nodes we've already visited
    visited: set[graph.Node] = set()

    # ensure the boundary is not none to simplify the check below
    boundary = boundary if boundary is not None else set()

    def _visit(node: graph.Node) -> None:
        # Ensure we only visit a node at most once
        if node in visited:
            return

        # Ensure there are no cycles (the input should be a DAG anyway)
        if node in visiting:
            raise ValueError(
                f"linearize: cycle detected in computation graph at node {node.name or f'<unnamed node {node.id}>'}"
            )

        # Register that we're visiting this node
        visiting.add(node)

        # Stop recursing when we reach the boundary
        if node not in boundary:
            # Get dependent nodes based on this node's type
            deps = _get_dependencies(node)

            # Visit all dependencies first
            for dep in deps:
                _visit(dep)

        # We are not visiting this node anymore
        visiting.remove(node)

        # We have visited this node
        visited.add(node)

        # We can append this node to the final plan
        plan.append(node)

    # Start visiting from the root nodes
    for node in roots:
        _visit(node)

    # Return the linearized plan to the caller
    return plan


def _get_dependencies(node: graph.Node) -> list[graph.Node]:
    """
    Get the direct dependencies of a node.

    Args:
        node: The node to get dependencies for

    Returns
    -------
        List of nodes that are direct dependencies

    Raises
    ------
        TypeError: If the node type is unknown
    """
    if isinstance(node, graph.BinaryOp):
        return [node.left, node.right]

    if isinstance(node, graph.UnaryOp):
        return [node.node]

    if isinstance(node, graph.where):
        return [node.condition, node.then, node.otherwise]

    if isinstance(node, graph.multi_clause_where):
        deps: list[graph.Node] = []
        for cond, value in node.clauses:
            deps.append(cond)
            deps.append(value)
        deps.append(node.default_value)
        return deps

    if isinstance(node, graph.AxisOp):
        return [node.node]

    if isinstance(node, (graph.constant, graph.placeholder)):
        return []

    if isinstance(node, graph.function):
        deps: list[graph.Node] = []
        deps.extend(node.args)
        deps.extend(node.kwargs.values())
        return deps

    raise TypeError(f"linearize: unknown node type: {type(node)}")
