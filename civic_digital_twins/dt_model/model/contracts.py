"""Contract decorators for :class:`~.model.Model` subclasses.

``@define``, ``@functions``, ``@inputs``, ``@outputs``, and ``@expose`` replace
the bare ``@dataclass`` convention with purpose-specific decorators that make
intent explicit and validate field types at construction time.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import functools
import sys
import typing
from collections.abc import Callable, Iterator
from typing import Any, Literal, TypeVar, dataclass_transform, overload

from ..axes import Axis
from .index import GenericIndex, _verify_declared_axes

__all__ = ["define", "expose", "functions", "inputs", "outputs"]

_T = TypeVar("_T")

_MISSING = object()


# ---------------------------------------------------------------------------
# @functions
# ---------------------------------------------------------------------------


def functions(
    _cls: type | None = None,
    *,
    extra: Literal["allow", "forbid"] = "allow",
) -> Any:
    """Declare an explicit function contract on a :class:`~.model.Model` subclass.

    Annotated fields are **explicit** (declared) functions: the caller must
    supply a :class:`~..engine.numpybackend.executor.Functor` for each one.
    Extra keyword arguments passed to the decorated class constructor go into
    ``_extra`` and are treated as **implicit** functions promoted by the
    caller — they are claimed by :class:`~.model.Model` at construction time
    for any matching :class:`~..engine.frontend.graph.function_call` node in
    the model's own subgraph.

    Parameters
    ----------
    extra:
        ``"allow"`` (default) — undeclared keyword arguments are stored in
        ``_extra`` and used for ancestor-level implicit-function promotion.
        ``"forbid"`` — undeclared keyword arguments raise :class:`TypeError`
        at construction time, enforcing a strict contract.

    Examples
    --------
    Declare and wire an explicit function::

        from civic_digital_twins.dt_model import Model, functions, inputs, outputs, NumpyBackend
        from civic_digital_twins.dt_model.engine.frontend import graph

        class TrafficModel(Model):

            @inputs
            class Inputs:
                ts_inflow: TimeseriesIndex

            @functions
            class Functions:
                ts_solve: Functor     # required explicit function

            def __init__(self, inputs: Inputs, *, fns: Functions) -> None:
                traffic = TimeseriesIndex(
                    "traffic",
                    graph.function_call("ts_solve", inputs.ts_inflow.node),
                )
                super().__init__("Traffic", inputs=inputs, functions=fns)

        model = TrafficModel(
            TrafficModel.Inputs(ts_inflow=ts_inflow),
            fns=TrafficModel.Functions(ts_solve=NumpyBackend.adapt(_ts_solve)),
        )

    Promote an implicit function from a parent::

        class MobilityModel(Model):
            def __init__(self, inputs: TrafficModel.Inputs) -> None:
                self.traffic = TrafficModel(
                    inputs,
                    fns=TrafficModel.Functions(
                        ts_solve=NumpyBackend.adapt(_ts_solve),
                        smooth=NumpyBackend.adapt(_smooth),   # implicit, goes into _extra
                    ),
                )
                super().__init__("Mobility", inputs=inputs)
    """

    def decorator(cls: type) -> type:
        declared: dict[str, Any] = {name: ann for name, ann in cls.__annotations__.items() if not name.startswith("_")}
        extra_mode = extra

        def __init__(self: Any, **kwargs: Any) -> None:
            extra_kw: dict[str, Any] = {}
            for k, v in kwargs.items():
                if k in declared:
                    setattr(self, k, v)
                else:
                    extra_kw[k] = v

            for name in declared:
                if name not in self.__dict__:
                    default = getattr(type(self), name, _MISSING)
                    if default is _MISSING:
                        raise TypeError(f"{cls.__name__}() missing required argument: {name!r}")
                    setattr(self, name, default)

            if extra_mode == "forbid" and extra_kw:
                raise TypeError(f"{cls.__name__}() got unexpected keyword arguments: {sorted(extra_kw)}")
            self._extra = extra_kw

        def items(self: Any) -> Iterator[tuple[str, Any]]:
            """Yield ``(name, functor)`` pairs for declared fields then extra fields."""
            for name in declared:
                yield name, getattr(self, name)
            yield from self._extra.items()

        def __repr__(self: Any) -> str:
            declared_parts = [f"{name}={getattr(self, name)!r}" for name in declared]
            extra_parts = [f"{k}={v!r}" for k, v in self._extra.items()]
            return f"{type(self).__name__}({', '.join(declared_parts + extra_parts)})"

        cls.__init__ = __init__  # type: ignore[assignment]
        cls.items = items  # type: ignore[assignment]
        cls.__repr__ = __repr__  # type: ignore[assignment]
        cls._declared = frozenset(declared)
        cls._extra_mode = extra_mode
        cls._is_functions = True

        return cls

    if _cls is not None:
        return decorator(_cls)
    return decorator


# ---------------------------------------------------------------------------
# @inputs / @outputs / @expose — shared implementation
# ---------------------------------------------------------------------------


def _validate_index_field(cls_name: str, field_name: str, val: Any) -> None:
    """Raise :class:`TypeError` if *val* is not a valid IO contract field value.

    Valid shapes: a single :class:`~.index.GenericIndex`, a ``list`` of them,
    a ``dict`` mapping strings to them, a nested ``@expose``-decorated dataclass
    instance, or an ``IOProxy`` wrapping an ``@expose`` or ``@outputs`` dataclass
    (for surfacing sub-model diagnostics and outputs for inspection).
    """
    if isinstance(val, GenericIndex):
        return
    if getattr(type(val), "_is_expose", False) or getattr(type(val), "_is_outputs", False):
        return
    # IOProxy wrapping an @expose or @outputs dataclass (model.expose / model.outputs
    # both return IOProxy, not the raw dataclass)
    _dc = getattr(val, "_dc", None)
    if _dc is not None and getattr(type(_dc), "_is_expose", False):
        return
    if _dc is not None and getattr(type(_dc), "_is_outputs", False):
        return
    if isinstance(val, list):
        for i, item in enumerate(val):
            if not isinstance(item, GenericIndex):
                raise TypeError(f"{cls_name}.{field_name}[{i}]: expected GenericIndex, got {type(item).__name__}")
        return
    if isinstance(val, dict):
        for k, item in val.items():
            if not isinstance(item, GenericIndex):
                raise TypeError(f"{cls_name}.{field_name}[{k!r}]: expected GenericIndex, got {type(item).__name__}")
        return
    raise TypeError(f"{cls_name}.{field_name}: expected GenericIndex (or list/dict thereof), got {type(val).__name__}")


def _declared_axes(annotation: Any) -> tuple[Axis, ...] | None:
    """Return the ``FIXED_AXES`` a field's resolved annotation declares, or ``None``.

    Recognizes a bare class carrying ``FIXED_AXES`` (e.g. ``TimeseriesIndex``,
    or any user-defined shape following the same pattern), or a
    ``list[...]``/``dict[str, ...]`` whose element type does. Returns
    ``None`` for anything else — a plain ``Index``/``ConstIndex``/
    ``GenericIndex`` annotation, or an annotation that :func:`typing.get_type_hints`
    could not resolve (left as a string) — so no shape check runs there,
    same as today.
    """
    origin = typing.get_origin(annotation)
    if origin in (list, dict):
        args = typing.get_args(annotation)
        return _declared_axes(args[-1]) if args else None
    return getattr(annotation, "FIXED_AXES", None)


def _verify_index_field_shape(cls_name: str, field_name: str, declared: tuple[Axis, ...], val: Any) -> None:
    """Verify *val* (or each of its elements) carries exactly *declared* axes.

    Structural, never nominal (see :func:`~.index._verify_declared_axes`):
    compares ``output_axes`` as a set, regardless of which ``Index``
    subclass produced the value. Only applies where *val* is (or contains) a
    :class:`~.index.GenericIndex` — nested ``@expose``/``@outputs``
    dataclasses and ``IOProxy`` fields carry no ``output_axes`` and are left
    to :func:`_validate_index_field`'s own checks.
    """
    if isinstance(val, GenericIndex):
        _verify_declared_axes(f"{cls_name}.{field_name}", declared, val.output_axes)
    elif isinstance(val, list):
        for i, item in enumerate(val):
            if isinstance(item, GenericIndex):
                _verify_declared_axes(f"{cls_name}.{field_name}[{i}]", declared, item.output_axes)
    elif isinstance(val, dict):
        for k, item in val.items():
            if isinstance(item, GenericIndex):
                _verify_declared_axes(f"{cls_name}.{field_name}[{k!r}]", declared, item.output_axes)


def _make_io_decorator(marker: str):
    """Return a decorator that wraps ``@dataclass`` and validates ``GenericIndex`` fields.

    The return type is intentionally left to inference so that the overloaded,
    ``@dataclass_transform``-marked ``entry_point`` type is preserved for callers
    (annotating it ``Any`` would erase the decorated class to ``Any``).
    """

    def decorator(cls: type) -> type:
        cls = dataclasses.dataclass(cls)
        original_init = cls.__init__

        # Resolve field annotations once, at decoration time, rather than on
        # every instantiation.  typing.get_type_hints() (rather than
        # dataclasses.fields()[i].type, a string under `from __future__ import
        # annotations`) is what lets us actually look at the annotated class
        # and check it for FIXED_AXES; on failure (an annotation that cannot
        # be resolved), fields default to no declared shape, same as today.
        try:
            hints = typing.get_type_hints(cls, globalns=vars(sys.modules[cls.__module__]), localns=vars(cls))
        except Exception:
            hints = {}
        field_axes = {name: _declared_axes(hint) for name, hint in hints.items()}

        @functools.wraps(original_init)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            cls_name = type(self).__name__
            for f in dataclasses.fields(self):  # type: ignore[arg-type]
                val = getattr(self, f.name)
                _validate_index_field(cls_name, f.name, val)
                declared = field_axes.get(f.name)
                if declared is not None:
                    _verify_index_field_shape(cls_name, f.name, declared, val)

        cls.__init__ = __init__  # type: ignore[assignment]
        setattr(cls, marker, True)
        return cls

    # @dataclass_transform lets Pyright synthesize the wrapped @dataclass
    # __init__ from the decorated class's own fields, so that constructing an
    # @inputs/@outputs/@expose class (e.g. ``Model.Inputs(x=...)``) is fully
    # type-checked at call sites without ``# type: ignore[call-arg]``.  It is a
    # static-only marker: ``entry_point`` is returned unchanged at runtime.  The
    # overloads keep the identity of the decorated class (``type[_T]``) so that
    # accessing it through a typed model does not collapse to ``Any``.
    @overload
    def entry_point(_cls: type[_T]) -> type[_T]: ...
    @overload
    def entry_point(_cls: None = ...) -> Callable[[type[_T]], type[_T]]: ...
    @dataclass_transform(field_specifiers=(dataclasses.field, dataclasses.Field))
    def entry_point(_cls: Any = None) -> Any:
        if _cls is not None:
            return decorator(_cls)
        return decorator

    return entry_point


inputs = _make_io_decorator("_is_inputs")
inputs.__name__ = "inputs"
inputs.__doc__ = """Decorator for declaring the ``Inputs`` contract on a :class:`~.model.Model` subclass.

Wraps ``@dataclass`` and validates at construction time that every field
holds a :class:`~.index.GenericIndex`, a ``list`` of them, or a ``dict``
mapping strings to them.

Passing a plain ``@dataclass`` instance as ``inputs=`` to
:class:`~.model.Model` is deprecated; use ``@inputs`` instead.

Examples
--------
::

    from civic_digital_twins.dt_model import Model, inputs, outputs

    class MyModel(Model):

        @inputs
        class Inputs:
            inflow: TimeseriesIndex

        @outputs
        class Outputs:
            traffic: TimeseriesIndex

        def __init__(self, inputs: Inputs) -> None:
            ...
            super().__init__("MyModel", inputs=inputs, outputs=...)
"""

outputs = _make_io_decorator("_is_outputs")
outputs.__name__ = "outputs"
outputs.__doc__ = """Decorator for declaring the ``Outputs`` contract on a :class:`~.model.Model` subclass.

Wraps ``@dataclass`` and validates at construction time that every field
holds a :class:`~.index.GenericIndex`, a ``list`` of them, or a ``dict``
mapping strings to them.

Passing a plain ``@dataclass`` instance as ``outputs=`` to
:class:`~.model.Model` is deprecated; use ``@outputs`` instead.
"""

expose = _make_io_decorator("_is_expose")
expose.__name__ = "expose"
expose.__doc__ = """Decorator for declaring inspectable non-contractual indexes on a :class:`~.model.Model` subclass.

Wraps ``@dataclass`` and validates at construction time that every field holds one of:

* a :class:`~.index.GenericIndex` (scalar, list, or dict thereof), or
* an ``IOProxy`` wrapping an ``@expose``-decorated dataclass — to surface a sub-model's
  diagnostics in bulk (``inflow=_inflow.expose``), or
* an ``IOProxy`` wrapping an ``@outputs``-decorated dataclass — to surface a sub-model's
  outputs for inspection in bulk (``inflow_out=_inflow.outputs``).

Passing a plain ``@dataclass`` instance as ``expose=`` to
:class:`~.model.Model` is deprecated; use ``@expose`` instead.
"""


# ---------------------------------------------------------------------------
# @define
# ---------------------------------------------------------------------------


def define(name: str) -> Callable[[type[_T]], type[_T]]:
    """Declare a leaf :class:`~.model.Model` subclass via a ``compute()`` method.

    Generates a typed ``__init__(self, inputs: Inputs)`` (plus ``fns: Functions``
    when a ``@functions`` inner class is declared) and wires the result of
    :meth:`compute` into ``super().__init__()`` automatically.

    Parameters
    ----------
    name:
        Human-readable model name forwarded to
        :class:`~.model.Model.__init__`.

    Usage
    -----
    Leaf model without ``Expose``::

        @define("Parking")
        class ParkingModel(Model):

            @inputs
            class Inputs:
                pv_tourists: ConditionalDistributionIndex

            @outputs
            class Outputs:
                i_u_parking: Index

            def compute(self, inputs: Inputs) -> Outputs:
                i_u_parking = Index("parking_usage", inputs.pv_tourists * ...)
                return ParkingModel.Outputs(i_u_parking=i_u_parking)

    With ``@functions`` and ``Expose``::

        @define("Traffic")
        class TrafficModel(Model):

            @inputs
            class Inputs:
                ts_inflow: TimeseriesIndex

            @functions
            class Functions:
                ts_solve: Functor

            @outputs
            class Outputs:
                ts_traffic: TimeseriesIndex

            @expose
            class Expose:
                ts_raw: TimeseriesIndex

            def compute(self, inputs: Inputs, *, fns: Functions) -> tuple[Outputs, Expose]:
                ts_raw     = TimeseriesIndex("raw", graph.function_call("ts_solve", inputs.ts_inflow))
                ts_traffic = TimeseriesIndex("traffic", ts_raw * ...)
                return (
                    TrafficModel.Outputs(ts_traffic=ts_traffic),
                    TrafficModel.Expose(ts_raw=ts_raw),
                )

    Composite / root models that assign sub-model attributes before calling
    ``super().__init__()`` can remain on the direct-``__init__`` path by
    declaring ``legacy=True``::

        class CompositeModel(Model, legacy=True):
            def __init__(self) -> None:
                self.leaf = LeafModel(...)
                ...
                super().__init__("composite", ...)

    Raises
    ------
    TypeError
        At decoration time if ``compute()`` is absent, if both ``compute``
        and ``__init__`` are defined, or if an ``@expose Expose`` class is
        declared but the ``compute`` return annotation does not include
        ``Expose``.
    """

    def decorator(cls: Any) -> Any:
        # Validate class structure at decoration time.
        if "compute" not in cls.__dict__:
            raise TypeError(f"@define({name!r}) requires {cls.__name__} to define a compute() method.")
        if "__init__" in cls.__dict__:
            raise TypeError(f"@define class {cls.__name__} must not define __init__. Implement compute() instead.")

        # Detect @functions inner class via its role marker.
        has_functions = "Functions" in cls.__dict__ and getattr(cls.__dict__["Functions"], "_is_functions", False)

        # Detect @expose Expose inner class declared directly on this class.
        has_expose_cls = "Expose" in cls.__dict__ and getattr(cls.__dict__["Expose"], "_is_expose", False)

        # Resolve compute()'s return annotation.  Using typing.get_type_hints()
        # rather than .__annotations__ handles `from __future__ import annotations`,
        # which otherwise stores all annotations as strings.
        try:
            hints = typing.get_type_hints(
                cls.compute,
                globalns=vars(sys.modules[cls.__module__]),
                localns=vars(cls),
            )
        except Exception:
            hints = getattr(cls.compute, "__annotations__", {})

        return_hint = hints.get("return", None)
        returns_expose = return_hint is not None and getattr(return_hint, "__origin__", None) is tuple

        # Consistency check: @expose declared → must appear in return annotation.
        if has_expose_cls and not returns_expose:
            raise TypeError(
                f"@define class {cls.__name__} declares an @expose Expose inner class "
                f"but compute() return annotation is not tuple[Outputs, Expose]. "
                f"Update the return annotation to include Expose, or remove the Expose declaration."
            )

        # Capture for closure so the generated __init__ uses the right class in
        # super() even if the @define class is later subclassed.
        _cls = cls
        _name = name
        _returns_expose = returns_expose

        # If Inputs has no declared fields it can be auto-constructed, so
        # make the `inputs` parameter optional (default None → Inputs()).
        _inputs_cls = cls.__dict__.get("Inputs")
        _inputs_is_empty = (
            _inputs_cls is not None
            and dataclasses.is_dataclass(_inputs_cls)
            and len(dataclasses.fields(_inputs_cls)) == 0  # type: ignore[arg-type]
        )

        # Use distinct names to avoid Pyright reportRedeclaration in the if/else.
        if has_functions:

            def _init_with_fns(self: Any, inputs: Any = None, *, fns: Any) -> None:  # type: ignore[misc]
                if _inputs_is_empty and inputs is None and _inputs_cls is not None:
                    inputs = _inputs_cls()  # type: ignore[operator]
                if _returns_expose:
                    out, exp = self.compute(inputs, fns=fns)
                    super(_cls, self).__init__(_name, inputs=inputs, outputs=out, expose=exp, functions=fns)  # type: ignore[misc]
                else:
                    out = self.compute(inputs, fns=fns)
                    super(_cls, self).__init__(_name, inputs=inputs, outputs=out, functions=fns)  # type: ignore[misc]

            cls.__init__ = _init_with_fns  # type: ignore[assignment]
        else:

            def _init_no_fns(self: Any, inputs: Any = None) -> None:
                if _inputs_is_empty and inputs is None and _inputs_cls is not None:
                    inputs = _inputs_cls()  # type: ignore[operator]
                if _returns_expose:
                    out, exp = self.compute(inputs)
                    super(_cls, self).__init__(_name, inputs=inputs, outputs=out, expose=exp)  # type: ignore[misc]
                else:
                    out = self.compute(inputs)
                    super(_cls, self).__init__(_name, inputs=inputs, outputs=out)  # type: ignore[misc]

            cls.__init__ = _init_no_fns  # type: ignore[assignment]

        cls._is_model_decorated = True
        return cls

    return decorator
