"""
Node Replacement for Computation Graphs
=======================================

This module provides facilities for transforming a computation graph
by replacing a set of nodes with another set of nodes. The transformation
creates a new graph without modifying the original nodes, preserving the
immutability of the computation graph.

The replacement process works recursively, traversing the entire graph
starting from the target node and applying replacements to matching nodes
and their descendants. For each node, it first checks if the node itself
should be replaced, then recursively processes its children.
"""

from . import graph


def nodes(
    target: graph.Node,
    replacements: dict[graph.Node, graph.Node],
) -> graph.Node:
    """Replace nodes in a computation graph with their specified replacements.

    This function creates a new graph by recursively traversing the original graph
    starting from the target node and replacing nodes according to the provided
    replacements dictionary. The function preserves the structure of the original
    graph for parts that don't need replacement.

    The replacement is performed using identity (is) comparison, not equality (==).
    If the target node is in the replacements dictionary, it is directly replaced.
    Otherwise, the function recursively processes its children, and if any child
    is replaced, a new node of the same type is created with the replaced children.

    Args:
        target: The root node of the graph or subgraph to transform.
        replacements: A dictionary mapping original nodes to their replacement nodes.
                     The keys are the nodes to be replaced, and the values are their
                     replacements.

    Returns:
        A new graph node representing the transformed graph. If no replacements
        were applied, the original target node is returned.

    Raises:
        TypeError: If an unsupported node type is encountered during traversal.

    Examples:
        >>> x = graph.placeholder("x")
        >>> y = graph.placeholder("y")
        >>> z = x + y * 2
        >>> # Replace y with a constant 5
        >>> new_z = nodes(z, {y: graph.constant(5)})
        >>> # Now new_z represents x + 5 * 2
    """

    # Direct replacement of the target node
    if target in replacements:
        return replacements[target]

    # Replacement for placeholders and constants
    if isinstance(target, graph.constant):
        return target
    if isinstance(target, graph.placeholder):
        return target

    # Replacement for binary operators
    if isinstance(target, graph.BinaryOp):
        # Attempt to replace the left and right children
        rleft, rright = nodes(target.left, replacements), nodes(target.right, replacements)

        # Handle the no-replacement case
        if rleft is target.left and rright is target.right:
            return target

        # Use copy_with method to create a new node with the replaced children
        return target.copy_with(rleft, rright)

    # Replacement for unary operators
    if isinstance(target, graph.UnaryOp):
        # Attempt to replace the child
        rnode = nodes(target.node, replacements)

        # Handle the no-replacement case
        if rnode is target.node:
            return target

        # Use copy_with method to create a new node with the replaced child
        return target.copy_with(rnode)

    # Replacement for axis operators
    if isinstance(target, graph.AxisOp):
        # Attempt to replace the child node
        rnode = nodes(target.node, replacements)

        # Handle the no-replacement case
        if rnode is target.node:
            return target

        # Use copy_with method to create a new node with the replaced child
        return target.copy_with(rnode)

    # Replacement for where operations
    if isinstance(target, graph.where):
        # Attempt to replace all three children
        rcondition = nodes(target.condition, replacements)
        rthen = nodes(target.then, replacements)
        rotherwise = nodes(target.otherwise, replacements)

        # Handle the no-replacement case
        if rcondition is target.condition and rthen is target.then and rotherwise is target.otherwise:
            return target

        # Use copy_with method to create a new node with the replaced children
        return target.copy_with(rcondition, rthen, rotherwise)

    # Replacement for multi_clause_where operations
    if isinstance(target, graph.multi_clause_where):
        # Attempt to replace each clause and the default value
        new_clauses = []
        clauses_changed = False

        for cond, value in target.clauses:
            rcond = nodes(cond, replacements)
            rvalue = nodes(value, replacements)
            new_clauses.append((rcond, rvalue))

            if rcond is not cond or rvalue is not value:
                clauses_changed = True

        rdefault = nodes(target.default_value, replacements)

        # Handle the no-replacement case
        if not clauses_changed and rdefault is target.default_value:
            return target

        # Use copy_with method to create a new node with the replaced children
        return target.copy_with(new_clauses, rdefault)

    # Handle the case of an unsupported node type
    raise TypeError(f"Unsupported node type for replacement: {type(target)}")
