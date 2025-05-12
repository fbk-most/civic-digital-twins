"""Tests for InstantiatedModel class."""
# SPDX-License-Identifier: Apache-2.0

from typing import cast

from scipy import stats

from civic_digital_twins.dt_model import Index
from civic_digital_twins.dt_model.model.abstract_model import AbstractModel
from civic_digital_twins.dt_model.model.instantiated_model import InstantiatedModel
from civic_digital_twins.dt_model.symbols.index import Distribution

c1 = cast(Distribution, stats.norm(loc=2.0, scale=1.0))
c2 = cast(Distribution, stats.norm(loc=4.0, scale=1.0))
d1 = cast(Distribution, stats.uniform(loc=10.0, scale=5.0))

abstract_model = AbstractModel("M", [], [], [Index("a", 1), Index("b", 2), Index("c", c1), Index("d", d1)], [], [])


def test_base_instantiated_model():
    """Test on values of an unmodified instantiated model."""
    inst = InstantiatedModel(abstract_model)
    modified_values = inst.get_values(all=False)
    all_values = inst.get_values(all=True)
    assert modified_values == {}
    assert all_values == {"a": 1, "b": 2, "c": c1, "d": d1}


def test_modified_instantiated_model():
    """Test on values of a modified instantiated model."""
    inst = InstantiatedModel(abstract_model, values={"a": -1, "c": c2})
    modified_values = inst.get_values(all=False)
    all_values = inst.get_values(all=True)
    assert modified_values == {"a": -1, "c": c2}
    assert modified_values["a"] == -1
    assert modified_values["c"].dist.name == "norm"
    assert modified_values["c"].kwds == {"loc": 4.0, "scale": 1.0}
    assert all_values == {"a": -1, "b": 2, "c": c2, "d": d1}
