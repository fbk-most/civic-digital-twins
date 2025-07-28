"""Tests for the civic_digital_twins.dt_model.engine.frontend.ir module."""

# SPDX-License-Identifier: Apache-2.0

import collections

import pytest

from civic_digital_twins.dt_model.engine.frontend import forest, graph, ir, linearize


def test_compile_trees_good():
    """Test that we correctly lower a forest to a DAG IR."""
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

    # Compile the trees to a DAG IR
    dag = ir.compile_trees(*trees)

    # Ensure that there are two placeholders
    assert len(dag.placeholders) == 2

    # Ensure that there are two constants
    assert len(dag.constants) == 2

    # Ensure there are three trees in output
    assert len(dag.trees) == 3

    # Ensure that the first tree is about node c
    assert dag.trees[0].root() is c

    # Ensure that the second tree is about node g
    assert dag.trees[1].root() is g

    # Ensure that the third tree is about node h
    assert dag.trees[2].root() is h

    # Ensure that the nodes are empty
    assert not dag.nodes

    # Ensure that the string representation is correct
    expect = f"""# === placeholders ===
n{x.id} = graph.placeholder(name='x', default_value=None)
n{y.id} = graph.placeholder(name='y', default_value=None)

# === constants ===
n{k0.id} = graph.constant(value=2.0, name='k0')
n{k1.id} = graph.constant(value=1.0, name='k1')

# === trees ===
def t{c.id}(n{x.id}: graph.placeholder, n{y.id}: graph.placeholder, n{k0.id}: graph.constant, n{k1.id}: graph.constant) -> graph.Node:
    n{a.id} = graph.add(left=n{x.id}, right=n{k0.id}, name='a')
    n{b.id} = graph.subtract(left=n{y.id}, right=n{k1.id}, name='b')
    n{c.id} = graph.multiply(left=n{a.id}, right=n{b.id}, name='c')
    return n{c.id}

def t{g.id}(n{y.id}: graph.placeholder, n{k1.id}: graph.constant, n{c.id}: graph.Node) -> graph.Node:
    n{d.id} = graph.exp(node=n{y.id}, name='d')
    n{e.id} = graph.add(left=n{c.id}, right=n{k1.id}, name='e')
    n{f.id} = graph.log(node=n{e.id}, name='f')
    n{g.id} = graph.divide(left=n{d.id}, right=n{f.id}, name='g')
    return n{g.id}

def t{h.id}(n{c.id}: graph.Node, n{g.id}: graph.Node) -> graph.Node:
    n{h.id} = graph.maximum(left=n{c.id}, right=n{g.id}, name='h')
    return n{h.id}

"""

    got = str(dag)

    assert expect == got


def test_compile_nodes_good():
    """Test that we correctly lower topologically-sorted nodes to a DAG IR."""
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

    # Create a linear plan for evaluating the DAG.
    plan = linearize.forest(h, g, c)

    # Compile the nodes to a DAG IR
    dag = ir.compile_nodes(*plan)

    # Ensure that there are two placeholders
    assert len(dag.placeholders) == 2

    # Ensure that there are two constants
    assert len(dag.constants) == 2

    # Ensure that the trees are empty
    assert not dag.trees

    # Ensure that we have the expected number of nodes
    assert len(dag.nodes) == 8

    # Ensure that the string representation is correct
    expect = f"""# === placeholders ===
n{x.id} = graph.placeholder(name='x', default_value=None)
n{y.id} = graph.placeholder(name='y', default_value=None)

# === constants ===
n{k0.id} = graph.constant(value=2.0, name='k0')
n{k1.id} = graph.constant(value=1.0, name='k1')

# === nodes ===
n{a.id} = graph.add(left=n{x.id}, right=n{k0.id}, name='a')
n{b.id} = graph.subtract(left=n{y.id}, right=n{k1.id}, name='b')
n{c.id} = graph.multiply(left=n{a.id}, right=n{b.id}, name='c')
n{d.id} = graph.exp(node=n{y.id}, name='d')
n{e.id} = graph.add(left=n{c.id}, right=n{k1.id}, name='e')
n{f.id} = graph.log(node=n{e.id}, name='f')
n{g.id} = graph.divide(left=n{d.id}, right=n{f.id}, name='g')
n{h.id} = graph.maximum(left=n{c.id}, right=n{g.id}, name='h')
"""

    got = str(dag)

    assert expect == got


def test_compile_nodes_raises():
    """Ensure that compile nodes raises on invalid types."""
    # We need to disable the type checker to run this tests w/o static type errors
    # Raises because `collections.deque()` is not a graph.Node
    with pytest.raises(RuntimeError):
        _ = ir.compile_nodes(collections.deque())  # type: ignore


def test_dag_post_init_invariant():
    """Ensure that the __post_init__ checks invariants."""
    # The specific kind of nodes and body is not that important and what matters
    # is the number of entries so let's use constant, which is simpler.
    with pytest.raises(RuntimeError):
        _ = ir.DAG(
            placeholders=list(),
            constants=list(),
            trees=[forest.Tree(inputs=list(), body=[graph.constant(1.0)])],
            nodes=[graph.constant(1.0)],
        )
