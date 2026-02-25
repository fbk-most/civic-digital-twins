"""Tests for the civic_digital_twins.dt_model.engine.numpybackend.jit module."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import jit


def test_round_trip():
    """Create a complex DAG and verify the resulting code is correct."""
    k = graph.constant(10)
    assert jit.graph_node_to_numpy_code(k) == f"n{k.id} = np.asarray(10)"

    p = graph.placeholder("p")
    pvalue = np.asarray([[10, 11, 12]])
    assert jit.graph_node_to_numpy_code(p, pvalue) == f"n{p.id} = np.asarray([[10, 11, 12]])"

    node = graph.add(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.add(n{k.id}, n{p.id})"

    node = graph.subtract(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.subtract(n{k.id}, n{p.id})"

    node = graph.multiply(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.multiply(n{k.id}, n{p.id})"

    node = graph.divide(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.divide(n{k.id}, n{p.id})"

    node = graph.equal(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.equal(n{k.id}, n{p.id})"

    node = graph.not_equal(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.not_equal(n{k.id}, n{p.id})"

    node = graph.less(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.less(n{k.id}, n{p.id})"

    node = graph.less_equal(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.less_equal(n{k.id}, n{p.id})"

    node = graph.greater(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.greater(n{k.id}, n{p.id})"

    node = graph.greater_equal(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.greater_equal(n{k.id}, n{p.id})"

    node = graph.logical_and(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.logical_and(n{k.id}, n{p.id})"

    node = graph.logical_or(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.logical_or(n{k.id}, n{p.id})"

    node = graph.logical_xor(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.logical_xor(n{k.id}, n{p.id})"

    node = graph.exp(k)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.exp(n{k.id})"

    node = graph.power(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.power(n{k.id}, n{p.id})"

    node = graph.log(k)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.log(n{k.id})"

    node = graph.maximum(k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.maximum(n{k.id}, n{p.id})"

    node = graph.where(k, k, p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.where(n{k.id}, n{k.id}, n{p.id})"

    node = graph.multi_clause_where([(k, p)], p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.select([n{k.id}], [n{p.id}], n{p.id})"

    node = graph.expand_dims(k, axis=(1, 2))
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.expand_dims(n{k.id}, axis=(1, 2))"

    node = graph.squeeze(k, axis=(2,))
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.squeeze(n{k.id}, axis=(2,))"

    node = graph.project_using_sum(k, axis=(2,))
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.sum(n{k.id}, axis=(2,), keepdims=True)"

    node = graph.project_using_mean(k, axis=(2,))
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = np.mean(n{k.id}, axis=(2,), keepdims=True)"

    node = graph.function_call("foo", k, p, k=k, p=p)
    assert jit.graph_node_to_numpy_code(node) == f"n{node.id} = foo(n{k.id}, n{p.id}, k=n{k.id}, p=n{p.id})"


class UnsupportedNode(graph.Node):
    """An unsupported node to verify it causes an exception."""


def test_no_operation_for_node():
    """Ensure the code throws when there is no operation for a node."""
    with pytest.raises(jit.UnsupportedNodeType):
        jit.graph_node_to_ast_stmt(UnsupportedNode())


def test_no_arguments_handling_for_node():
    """Same as above but tests the case when there's no arguments handling code."""
    with pytest.raises(jit.UnsupportedNodeArguments):
        jit.graph_node_to_ast_stmt(jit._InternalTestingNode())
