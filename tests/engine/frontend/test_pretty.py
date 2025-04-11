"""Tests for the yakof.frontend.pretty module."""

# SPDX-License-Identifier: Apache-2.0

from dt_model.engine.frontend import graph, pretty


def test_basic_pretty_printing():
    """Test basic pretty printing of computation graphs."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.add(x, y)

    result = pretty.format(z)
    assert result == "x + 2.0"


def test_precedence_pretty_printing():
    """Test pretty printing with operator precedence."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.constant(3.0)

    expr1 = graph.add(x, graph.multiply(y, z))
    result1 = pretty.format(expr1)
    assert result1 == "x + 2.0 * 3.0"

    expr2 = graph.multiply(graph.add(x, y), z)
    result2 = pretty.format(expr2)
    assert result2 == "(x + 2.0) * 3.0"


def test_unary_operations_pretty_printing():
    """Test pretty printing of unary operations."""
    x = graph.placeholder("x")

    expr1 = graph.exp(x)
    result1 = pretty.format(expr1)
    assert result1 == "exp(x)"

    expr2 = graph.log(x)
    result2 = pretty.format(expr2)
    assert result2 == "log(x)"

    expr3 = graph.logical_not(x)
    result3 = pretty.format(expr3)
    assert result3 == "~x"


def test_comparison_operations_pretty_printing():
    """Test pretty printing of comparison operations."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)

    expr1 = graph.less(x, y)
    result1 = pretty.format(expr1)
    assert result1 == "x < 2.0"

    expr2 = graph.greater_equal(x, y)
    result2 = pretty.format(expr2)
    assert result2 == "x >= 2.0"


def test_logical_operations_pretty_printing():
    """Test pretty printing of logical operations."""
    x = graph.placeholder("x")
    y = graph.constant(True)

    expr1 = graph.logical_and(x, y)
    result1 = pretty.format(expr1)
    assert result1 == "x & True"

    expr2 = graph.logical_or(x, y)
    result2 = pretty.format(expr2)
    assert result2 == "x | True"

    expr3 = graph.logical_xor(x, y)
    result3 = pretty.format(expr3)
    assert result3 == "x ^ True"


def test_named_expressions_pretty_printing():
    """Test pretty printing of named expressions."""
    x = graph.placeholder("x")
    y = graph.constant(2.0, name="const_y")
    z = graph.add(x, y)

    result = pretty.format(z)
    assert result == "x + const_y"


def test_subtract_pretty_printing():
    """Test pretty printing of subtract operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.subtract(x, y)

    result = pretty.format(z)
    assert result == "x - 2.0"


def test_divide_pretty_printing():
    """Test pretty printing of divide operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.divide(x, y)

    result = pretty.format(z)
    assert result == "x / 2.0"


def test_power_pretty_printing():
    """Test pretty printing of power operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.power(x, y)

    result = pretty.format(z)
    assert result == "x ** 2.0"


def test_less_equal_pretty_printing():
    """Test pretty printing of less_equal operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.less_equal(x, y)

    result = pretty.format(z)
    assert result == "x <= 2.0"


def test_greater_pretty_printing():
    """Test pretty printing of greater operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.greater(x, y)

    result = pretty.format(z)
    assert result == "x > 2.0"


def test_equal_pretty_printing():
    """Test pretty printing of equal operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.equal(x, y)

    result = pretty.format(z)
    assert result == "x == 2.0"


def test_not_equal_pretty_printing():
    """Test pretty printing of not_equal operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.not_equal(x, y)

    result = pretty.format(z)
    assert result == "x != 2.0"


def test_where_pretty_printing():
    """Test pretty printing of where operation."""
    cond = graph.placeholder("cond")
    x = graph.constant(1.0)
    y = graph.constant(2.0)
    z = graph.where(cond, x, y)

    result = pretty.format(z)
    assert result == "where(cond, 1.0, 2.0)"


def test_multi_clause_where_pretty_printing():
    """Test pretty printing of multi_clause_where operation."""
    cond1 = graph.placeholder("cond1")
    cond2 = graph.placeholder("cond2")
    x = graph.constant(1.0)
    y = graph.constant(2.0)
    z = graph.constant(3.0)

    expr = graph.multi_clause_where([(cond1, x), (cond2, y)], z)
    result = pretty.format(expr)
    assert result == "multi_clause_where([(cond1, 1.0), (cond2, 2.0)], 3.0)"


def test_expand_dims_pretty_printing():
    """Test pretty printing of expand_dims operation."""
    x = graph.placeholder("x")
    y = graph.expand_dims(x, 0)

    result = pretty.format(y)
    assert result == "expand_dims(x, 0)"

    # Test with tuple axis
    z = graph.expand_dims(x, (0, 1))
    result = pretty.format(z)
    assert result == "expand_dims(x, (0, 1))"


def test_squeeze_pretty_printing():
    """Test pretty printing of squeeze operation."""
    x = graph.placeholder("x")
    y = graph.squeeze(x, 0)

    result = pretty.format(y)
    assert result == "squeeze(x, 0)"

    # Test with tuple axis
    z = graph.squeeze(x, (0, 1))
    result = pretty.format(z)
    assert result == "squeeze(x, (0, 1))"


def test_reduce_sum_pretty_printing():
    """Test pretty printing of reduce_sum operation."""
    x = graph.placeholder("x")
    y = graph.reduce_sum(x, 0)

    result = pretty.format(y)
    assert result == "reduce_sum(x, 0)"

    # Test with tuple axis
    z = graph.reduce_sum(x, (0, 1))
    result = pretty.format(z)
    assert result == "reduce_sum(x, (0, 1))"


def test_reduce_mean_pretty_printing():
    """Test pretty printing of reduce_mean operation."""
    x = graph.placeholder("x")
    y = graph.reduce_mean(x, 0)

    result = pretty.format(y)
    assert result == "reduce_mean(x, 0)"

    # Test with tuple axis
    z = graph.reduce_mean(x, (0, 1))
    result = pretty.format(z)
    assert result == "reduce_mean(x, (0, 1))"


def test_maximum_pretty_printing():
    """Test pretty printing of maximum operation."""
    x = graph.placeholder("x")
    y = graph.constant(2.0)
    z = graph.maximum(x, y)

    result = pretty.format(z)
    assert result == "maximum(x, 2.0)"


def test_unhandled_node_type():
    """Test pretty printing of an unhandled node type."""

    class UnhandledNode(graph.Node):
        pass

    x = UnhandledNode()
    result = pretty.format(x)
    assert result == "<unknown:UnhandledNode>"


def test_named_subexpressions_not_expanded():
    """Test that named subexpressions aren't expanded in pretty printing."""
    # Create a named subexpression
    x = graph.placeholder("x")
    y = graph.add(x, graph.constant(1.0))
    y.name = "y"  # Give the subexpression a name

    # Use the named subexpression in a larger expression
    z = graph.multiply(y, graph.constant(2.0))

    # The pretty-printed result should use the name rather than expanding
    result = pretty.format(z)
    assert result == "y * 2.0"  # Should use y's name, not expand it to "(x + 1.0) * 2.0"

    # Verify with more complex nested expressions
    a = graph.exp(y)
    a.name = "a"
    b = graph.where(graph.placeholder("cond"), a, graph.constant(3.0))

    result = pretty.format(b)
    assert result == "where(cond, a, 3.0)"  # Should use a's name, not expand it

    # Test with multiple levels of named expressions
    c = graph.add(b, graph.constant(4.0))
    c.name = "c"
    d = graph.subtract(c, graph.constant(5.0))

    result = pretty.format(d)
    assert result == "c - 5.0"  # Should use c's name, not expand further


def test_nested_logical_precedence():
    """Test precedence handling with nested logical operations that would expose the closure bug."""
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    z = graph.placeholder("z")
    w = graph.placeholder("w")

    # Test case 1: (x | y) & z
    # The OR has lower precedence than AND, so parentheses are required
    expr1 = graph.logical_and(graph.logical_or(x, y), z)
    result1 = pretty.format(expr1)
    assert result1 == "(x | y) & z"

    # Test case 2: x & (y | z)
    # No parentheses needed around the x & part, but needed for y | z
    expr2 = graph.logical_and(x, graph.logical_or(y, z))
    result2 = pretty.format(expr2)
    assert result2 == "x & (y | z)"

    # Test case 3: (x & y) | (z & w)
    # No parentheses needed due to operator precedence
    expr3 = graph.logical_or(
        graph.logical_and(x, y),
        graph.logical_and(z, w)
    )
    result3 = pretty.format(expr3)
    assert result3 == "x & y | z & w"

    # Test case 4: Complex nested case with mixed precedence
    # ((x | y) & z) | (w & y)
    # The bug would likely produce incorrect parentheses
    expr4 = graph.logical_or(
        graph.logical_and(graph.logical_or(x, y), z),
        graph.logical_and(w, y)
    )
    result4 = pretty.format(expr4)
    assert result4 == "(x | y) & z | w & y"

    # Test case 5: Test with three levels of different precedence
    # ~(x | y) & z
    # This should add parentheses around x | y, but not around the NOT operation
    expr5 = graph.logical_and(graph.logical_not(graph.logical_or(x, y)), z)
    result5 = pretty.format(expr5)
    assert result5 == "~(x | y) & z"

    # Test case 6: Test with reversed precedence: (x | y) & (z | w)
    expr6 = graph.logical_and(
        graph.logical_or(x, y),
        graph.logical_or(z, w)
    )
    result6 = pretty.format(expr6)
    assert result6 == "(x | y) & (z | w)"
