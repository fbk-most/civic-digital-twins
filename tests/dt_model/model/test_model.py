"""Tests for civic_digital_twins.dt_model.model.Model."""

# SPDX-License-Identifier: Apache-2.0

from typing import cast

from scipy import stats

from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.model.index import Distribution, Index, TimeseriesIndex
import numpy as np


c1 = cast(Distribution, stats.norm(loc=2.0, scale=1.0))


def test_model_stores_name_and_indexes():
    """Model stores its name and index list."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", [a, b])
    assert m.name == "test"
    assert m.indexes == [a, b]


def test_model_abstract_indexes_empty_when_all_concrete():
    """abstract_indexes() is empty when all indexes have concrete values."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", [a, b])
    assert m.abstract_indexes() == []
    assert m.is_instantiated()


def test_model_abstract_indexes_includes_none_value():
    """abstract_indexes() includes placeholder (value=None) indexes."""
    a = Index("a", 1.0)
    p = Index("p", None)
    m = Model("test", [a, p])
    assert m.abstract_indexes() == [p]
    assert not m.is_instantiated()


def test_model_abstract_indexes_includes_distribution_value():
    """abstract_indexes() includes distribution-backed indexes."""
    a = Index("a", 1.0)
    d = Index("d", c1)
    m = Model("test", [a, d])
    assert m.abstract_indexes() == [d]
    assert not m.is_instantiated()


def test_model_abstract_indexes_includes_timeseries_placeholder():
    """abstract_indexes() includes TimeseriesIndex placeholders."""
    ts = TimeseriesIndex("ts")  # placeholder — values=None
    tf = TimeseriesIndex("tf", np.array([1.0, 2.0]))  # concrete
    m = Model("test", [ts, tf])
    assert m.abstract_indexes() == [ts]
    assert not m.is_instantiated()


def test_model_formula_index_is_concrete():
    """A formula-based Index (node wrapping) is not abstract."""
    from civic_digital_twins.dt_model.engine.frontend import graph

    n = graph.constant(3.0)
    idx = Index("formula", n)
    m = Model("test", [idx])
    assert m.abstract_indexes() == []
    assert m.is_instantiated()
