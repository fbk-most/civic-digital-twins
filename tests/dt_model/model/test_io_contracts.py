"""Tests for the @inputs, @outputs, and @expose contract decorators."""

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import warnings

import pytest

from civic_digital_twins.dt_model import expose, inputs, outputs
from civic_digital_twins.dt_model.model.index import Index, TimeseriesIndex
from civic_digital_twins.dt_model.model.model import Model

# ---------------------------------------------------------------------------
# @inputs
# ---------------------------------------------------------------------------


def test_inputs_decorator_applies_dataclass():
    """@inputs applies @dataclass so the class becomes a dataclass."""

    @inputs
    class Inputs:
        x: Index

    assert dataclasses.is_dataclass(Inputs)


def test_inputs_decorator_stamps_marker():
    """@inputs sets _is_inputs = True on the decorated class."""

    @inputs
    class Inputs:
        x: Index

    assert getattr(Inputs, "_is_inputs", False) is True


def test_inputs_accepts_scalar_index():
    """A scalar GenericIndex field is accepted without error."""

    @inputs
    class Inputs:
        x: Index

    idx = Index("x", 1.0)
    inst = Inputs(x=idx)
    assert inst.x is idx


def test_inputs_accepts_list_of_indexes():
    """A list[GenericIndex] field is accepted without error."""

    @inputs
    class Inputs:
        xs: list

    idxs = [Index("a", 1.0), Index("b", 2.0)]
    inst = Inputs(xs=idxs)
    assert inst.xs is idxs


def test_inputs_accepts_dict_of_indexes():
    """A dict[str, GenericIndex] field is accepted without error."""

    @inputs
    class Inputs:
        xs: dict

    d = {"a": Index("a", 1.0)}
    inst = Inputs(xs=d)
    assert inst.xs is d


def test_inputs_rejects_scalar_non_index():
    """A non-GenericIndex scalar value raises TypeError."""

    @inputs
    class Inputs:
        x: Index

    with pytest.raises(TypeError, match="expected GenericIndex"):
        Inputs(x="not_an_index")  # type: ignore[arg-type]


def test_inputs_rejects_list_with_non_index_element():
    """A list containing a non-GenericIndex element raises TypeError with field[i] context."""

    @inputs
    class Inputs:
        xs: list

    with pytest.raises(TypeError, match=r"xs\[0\].*expected GenericIndex"):
        Inputs(xs=["bad"])  # type: ignore[arg-type]


def test_inputs_rejects_dict_with_non_index_value():
    """A dict with a non-GenericIndex value raises TypeError with field['key'] context."""

    @inputs
    class Inputs:
        xs: dict

    with pytest.raises(TypeError, match=r"xs\['k'\].*expected GenericIndex"):
        Inputs(xs={"k": 42})  # type: ignore[arg-type]


def test_inputs_empty_list_is_valid():
    """An empty list field passes validation."""

    @inputs
    class Inputs:
        xs: list

    assert Inputs(xs=[]).xs == []


def test_inputs_empty_dict_is_valid():
    """An empty dict field passes validation."""

    @inputs
    class Inputs:
        xs: dict

    assert Inputs(xs={}).xs == {}


def test_inputs_timeseries_index_accepted():
    """TimeseriesIndex (a GenericIndex subclass) is accepted by @inputs."""

    @inputs
    class Inputs:
        ts: TimeseriesIndex

    ts = TimeseriesIndex("ts")
    inst = Inputs(ts=ts)
    assert inst.ts is ts


def test_inputs_supports_both_call_forms():
    """@inputs and @inputs() both work."""

    @inputs
    class A:
        x: Index

    @inputs()
    class B:
        x: Index

    idx = Index("x", 1.0)
    assert A(x=idx).x is idx
    assert B(x=idx).x is idx


# ---------------------------------------------------------------------------
# @outputs
# ---------------------------------------------------------------------------


def test_outputs_decorator_stamps_marker():
    """@outputs sets _is_outputs = True on the decorated class."""

    @outputs
    class Outputs:
        y: Index

    assert getattr(Outputs, "_is_outputs", False) is True


def test_outputs_validates_fields():
    """@outputs validates that field values are GenericIndex instances."""

    @outputs
    class Outputs:
        y: Index

    with pytest.raises(TypeError, match="expected GenericIndex"):
        Outputs(y=99)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# @expose
# ---------------------------------------------------------------------------


def test_expose_decorator_stamps_marker():
    """@expose sets _is_expose = True on the decorated class."""

    @expose
    class Expose:
        z: Index

    assert getattr(Expose, "_is_expose", False) is True


def test_expose_validates_fields():
    """@expose validates that field values are GenericIndex instances."""

    @expose
    class Expose:
        z: Index

    with pytest.raises(TypeError, match="expected GenericIndex"):
        Expose(z="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Deprecation warning for plain @dataclass
# ---------------------------------------------------------------------------


def test_plain_dataclass_inputs_emits_deprecation_warning():
    """Passing a plain @dataclass as inputs= emits a DeprecationWarning pointing at @inputs."""

    @dataclasses.dataclass
    class Inputs:
        x: Index

    x = Index("x", 1.0)

    class M(Model):
        def __init__(self) -> None:
            super().__init__("M", inputs=Inputs(x=x), outputs=None)

    with pytest.warns(DeprecationWarning, match="@inputs"):
        M()


def test_plain_dataclass_outputs_emits_deprecation_warning():
    """Passing a plain @dataclass as outputs= emits a DeprecationWarning pointing at @outputs."""

    @dataclasses.dataclass
    class Outputs:
        y: Index

    y = Index("y", 2.0)

    class M(Model):
        def __init__(self) -> None:
            super().__init__("M", outputs=Outputs(y=y))

    with pytest.warns(DeprecationWarning, match="@outputs"):
        M()


def test_inputs_decorator_no_deprecation_warning():
    """Using @inputs does not trigger any DeprecationWarning."""

    @inputs
    class Inputs:
        x: Index

    x = Index("x", 1.0)

    class M(Model):
        def __init__(self) -> None:
            super().__init__("M", inputs=Inputs(x=x))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        M()  # must not raise


# ---------------------------------------------------------------------------
# Integration: @inputs / @outputs / @expose with Model
# ---------------------------------------------------------------------------


def test_model_uses_inputs_proxy_from_decorated_class():
    """model.inputs.x and model.outputs.y are accessible after construction."""

    @inputs
    class Inputs:
        x: Index

    @outputs
    class Outputs:
        y: Index

    x = Index("x", 1.0)
    y = Index("y", x.node + x.node)

    class M(Model):
        def __init__(self) -> None:
            super().__init__(
                "M",
                inputs=Inputs(x=x),
                outputs=Outputs(y=y),
            )

    m = M()
    assert m.inputs.x is x
    assert m.outputs.y is y


def test_model_expose_decorated_class():
    """model.expose.internal is accessible when @expose is used."""

    @expose
    class Expose:
        internal: Index

    internal = Index("internal", 3.14)

    class M(Model):
        def __init__(self) -> None:
            super().__init__("M", expose=Expose(internal=internal))

    m = M()
    assert m.expose.internal is internal
