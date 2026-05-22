"""``@functions`` decorator for declaring explicit function contracts on :class:`~.model.Model` subclasses."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

__all__ = ["functions"]

_MISSING = object()


def functions(
    _cls: type | None = None,
    *,
    extra: Literal["allow", "forbid"] = "allow",
) -> Any:
    """Decorator for declaring an explicit function contract on a :class:`~.model.Model` subclass.

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

        from dataclasses import dataclass
        from civic_digital_twins.dt_model import Model, functions, NumpyBackend
        from civic_digital_twins.dt_model.engine.frontend import graph

        class TrafficModel(Model):

            @dataclass
            class Inputs:
                ts_inflow: TimeseriesIndex

            @functions
            class Functions:
                ts_solve: Functor     # required explicit function

            def __init__(self, inputs: Inputs, *, functions: Functions) -> None:
                traffic = TimeseriesIndex(
                    "traffic",
                    graph.function_call("ts_solve", inputs.ts_inflow),
                )
                super().__init__("Traffic", inputs=inputs, functions=functions)

        model = TrafficModel(
            inputs,
            functions=TrafficModel.Functions(ts_solve=NumpyBackend.adapt(_ts_solve)),
        )

    Promote an implicit function from a parent::

        class MobilityModel(Model):
            def __init__(self, inputs: Inputs) -> None:
                self.traffic = TrafficModel(
                    inputs,
                    functions=TrafficModel.Functions(
                        ts_solve=NumpyBackend.adapt(_ts_solve),
                        smooth=NumpyBackend.adapt(_smooth),   # implicit, goes into _extra
                    ),
                )
                super().__init__("Mobility", inputs=inputs)
    """

    def decorator(cls: type) -> type:
        # Collect declared field names from class-level annotations, skipping private names.
        declared: dict[str, Any] = {
            name: ann
            for name, ann in cls.__annotations__.items()
            if not name.startswith("_")
        }
        extra_mode = extra

        def __init__(self: Any, **kwargs: Any) -> None:
            extra_kw: dict[str, Any] = {}
            for k, v in kwargs.items():
                if k in declared:
                    setattr(self, k, v)
                else:
                    extra_kw[k] = v

            # Ensure all required declared fields are set (no class-level default = required).
            for name in declared:
                if not hasattr(self, name):
                    default = getattr(type(self), name, _MISSING)
                    if default is _MISSING:
                        raise TypeError(f"{cls.__name__}() missing required argument: {name!r}")
                    setattr(self, name, default)

            if extra_mode == "forbid" and extra_kw:
                raise TypeError(
                    f"{cls.__name__}() got unexpected keyword arguments: {sorted(extra_kw)}"
                )
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
        # Called as @functions (no parentheses).
        return decorator(_cls)
    # Called as @functions(...) with keyword arguments.
    return decorator
