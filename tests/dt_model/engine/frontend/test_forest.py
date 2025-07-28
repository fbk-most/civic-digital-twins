"""Tests for the civic_digital_twins.dt_model.engine.frontend.forest module."""

# SPDX-License-Identifier: Apache-2.0

from civic_digital_twins.dt_model.engine.frontend import forest, graph


def test_partition():
    """Test that we correctly partition the forest."""
    x = graph.placeholder("x")
    y = graph.placeholder("y")

    k0 = graph.constant(2.0, name="k0")
    k1 = graph.constant(1.0, name="k1")

    # c := (x + k0) * (y - k1)
    a = graph.add(x, k0, name="a")
    b = graph.subtract(y, k1, name="b")
    c = graph.multiply(a, b, name="c")

    # g := exp(y) / log(c + k1)
    d = graph.exp(y, name="d")
    e = graph.add(c, k1, name="e")
    f = graph.log(e, name="f")
    g = graph.divide(d, f, name="g")

    # h := max(c, g)
    h = graph.maximum(c, g, name="h")

    # Partition the DAG using out of order roots to ensure that
    # we are correctly reordering the trees
    trees = forest.partition(h, g, c)

    # Ensure there are three trees in output
    assert len(trees) == 3

    # Ensure that the first tree is about node c
    assert trees[0].root() is c

    # Ensure that the second tree is about node g
    assert trees[1].root() is g

    # Ensure that the third tree is about node h
    assert trees[2].root() is h

    # Ensure that the string representation is correct
    expect = f"""def t{g.id}(n{y.id}: graph.Node, n{k1.id}: graph.Node, n{c.id}: graph.Node) -> graph.Node:
    n{d.id} = graph.exp(node=n{y.id}, name='d')
    n{e.id} = graph.add(left=n{c.id}, right=n{k1.id}, name='e')
    n{f.id} = graph.log(node=n{e.id}, name='f')
    n{g.id} = graph.divide(left=n{d.id}, right=n{f.id}, name='g')
    return n{g.id}
"""

    got = str(trees[1])

    assert expect == got

    # Tests for tree equality and inequality
    assert trees[0] == trees[0]
    assert trees[1] != trees[0]
