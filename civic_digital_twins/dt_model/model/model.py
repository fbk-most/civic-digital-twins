"""Core model definition."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from .index import Distribution, GenericIndex, Index

_DC = TypeVar("_DC")

# ---------------------------------------------------------------------------
# IOProxy value types
# ---------------------------------------------------------------------------

# A single proxy slot can hold a scalar index, a list of indexes, or a
# dict mapping strings to indexes.
_ProxyValue = GenericIndex | list[GenericIndex] | dict[str, GenericIndex]


def _iter_scalars(value: _ProxyValue) -> Iterator[GenericIndex]:
    """Yield all scalar :class:`~.index.GenericIndex` items from *value*.

    Parameters
    ----------
    value:
        A single :class:`~.index.GenericIndex`, a ``list`` of them, or a
        ``dict`` mapping strings to them.

    Yields
    ------
    GenericIndex
        Each scalar index in declaration order.
    """
    if isinstance(value, dict):
        yield from value.values()
    elif isinstance(value, list):
        yield from value
    else:
        yield value


# ---------------------------------------------------------------------------
# IOProxy
# ---------------------------------------------------------------------------


class IOProxy(Generic[_DC]):
    """Read-only attribute-access proxy over a declared inputs, outputs, or expose mapping.

    The class is generic over the dataclass type *_DC* it wraps.  When
    constructed from a dataclass instance (the normal path), field access
    ``proxy.field`` is typed as :data:`~typing.Any` by the type checker, which
    means the declared field type on *_DC* flows through without any
    :func:`~typing.cast` calls at the call site.

    Each slot is accessible by the field name used to register it.  The slot
    value may be a single :class:`~.index.GenericIndex`, a ``list`` of them,
    or a ``dict`` mapping strings to them.

    Supports:

    * Attribute access — ``proxy.field``  (returns the raw value: scalar, list, or dict)
    * Iteration        — ``for idx in proxy`` — yields scalar indexes only
      (lists and dict values are flattened).
    * ``len(proxy)``   — counts scalar entries (same flattening).
    * ``idx in proxy`` — identity-based membership test across all scalar entries.
    * ``repr(proxy)``  — lists declared field names.
    """

    def __init__(self, entries: list[tuple[str, _ProxyValue]], dc: _DC | None = None) -> None:
        # entries is an ordered list of (field_name, value) pairs.
        # dc is the original dataclass instance (if any) — stored so that
        # __getattr__ can delegate to it and return Any, giving callers precise
        # field types without requiring cast().
        # We use object.__setattr__ throughout to avoid triggering our own
        # __setattr__ override.
        object.__setattr__(self, "_entries", entries)
        object.__setattr__(self, "_map", {key: val for key, val in entries})
        object.__setattr__(self, "_dc", dc)

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Return the value registered under *name*.

        When the proxy was built from a dataclass instance the return type is
        :data:`~typing.Any`, which allows the declared field type on the
        dataclass to flow through at the call site without requiring
        :func:`~typing.cast`.

        Parameters
        ----------
        name:
            The field name to look up.

        Returns
        -------
        Any
            The registered value (a scalar index, list, or dict in practice).

        Raises
        ------
        AttributeError
            If *name* is not a registered field.
        """
        mapping: dict[str, _ProxyValue] = object.__getattribute__(self, "_map")
        if name in mapping:
            return mapping[name]
        raise AttributeError(f"No input/output with attribute name {name!r}. Available: {list(mapping)}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Raise :class:`AttributeError` — :class:`IOProxy` is read-only."""
        raise AttributeError("IOProxy is read-only.")

    # ------------------------------------------------------------------
    # Iteration / containment / sizing — operate on flattened scalars
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[GenericIndex]:
        """Iterate over scalar :class:`~.index.GenericIndex` entries in declaration order.

        Lists and dict values are flattened; only scalar indexes are yielded.
        """
        entries: list[tuple[str, _ProxyValue]] = object.__getattribute__(self, "_entries")
        for _, val in entries:
            yield from _iter_scalars(val)

    def __len__(self) -> int:
        """Return the total count of scalar :class:`~.index.GenericIndex` entries.

        Lists and dict values contribute their individual elements to the count.
        """
        entries: list[tuple[str, _ProxyValue]] = object.__getattribute__(self, "_entries")
        return sum(1 for _, val in entries for _ in _iter_scalars(val))

    def __contains__(self, item: object) -> bool:
        """Return ``True`` if *item* is one of the scalar entries (identity check).

        Parameters
        ----------
        item:
            Object to test for membership.
        """
        entries: list[tuple[str, _ProxyValue]] = object.__getattribute__(self, "_entries")
        return any(idx is item for _, val in entries for idx in _iter_scalars(val))

    def __repr__(self) -> str:
        """Return a string representation listing the declared field names."""
        entries: list[tuple[str, _ProxyValue]] = object.__getattribute__(self, "_entries")
        return f"IOProxy({[key for key, _ in entries]})"


# ---------------------------------------------------------------------------
# Proxy builders
# ---------------------------------------------------------------------------


def _proxy_from_dataclass(dc_instance: _DC) -> IOProxy[_DC]:
    """Build an :class:`IOProxy` from a dataclass instance.

    Each dataclass field becomes one slot in the proxy; its value may be a
    scalar :class:`~.index.GenericIndex`, a ``list`` of them, or a ``dict``
    mapping strings to them.

    The original dataclass instance is stored on the proxy so that
    :meth:`IOProxy.__getattr__` can delegate to it, allowing the precise field
    type to flow through to callers without requiring :func:`~typing.cast`.

    Parameters
    ----------
    dc_instance:
        An instance of any dataclass.

    Returns
    -------
    IOProxy[_DC]
        Proxy whose attribute keys are the dataclass field names.
    """
    entries: list[tuple[str, _ProxyValue]] = []
    for field in dataclasses.fields(dc_instance):  # type: ignore[arg-type]
        val: _ProxyValue = getattr(dc_instance, field.name)
        entries.append((field.name, val))
    return IOProxy(entries, dc=dc_instance)


def _collect_indexes(
    inputs: Any | None,
    outputs: Any | None,
    expose: Any | None,
) -> list[GenericIndex]:
    """Collect and deduplicate all scalar indexes from dataclass instances.

    Iterates over all fields of *inputs*, *outputs*, and *expose* (each may be
    ``None`` or a dataclass instance), flattens list/dict values, and returns a
    deduplicated list preserving first-seen order.

    Parameters
    ----------
    inputs:
        Dataclass instance or ``None``.
    outputs:
        Dataclass instance or ``None``.
    expose:
        Dataclass instance or ``None``.

    Returns
    -------
    list[GenericIndex]
        Deduplicated flat index list.
    """
    seen: set[int] = set()
    result: list[GenericIndex] = []
    for dc in (inputs, outputs, expose):
        if dc is None:
            continue
        for field in dataclasses.fields(dc):  # type: ignore[arg-type]
            val: _ProxyValue = getattr(dc, field.name)
            for idx in _iter_scalars(val):
                if id(idx) not in seen:
                    seen.add(id(idx))
                    result.append(idx)
    return result


def _build_proxy(
    label: str,
    declarations: list[GenericIndex],
    instance_dict: dict[str, Any],
    indexes: list[GenericIndex],
    model_name: str,
) -> IOProxy[Any]:
    """Build an :class:`IOProxy` for a legacy flat inputs/outputs declaration.

    The attribute name used for proxy access is determined by looking up each
    declared index in the model instance's ``__dict__``: whichever attribute
    name holds that exact object (identity check) becomes the access key.
    ``index.name`` is not used here — it is a display label only.

    Parameters
    ----------
    label:
        ``"inputs"`` or ``"outputs"`` — used in error messages only.
    declarations:
        Ordered list of :class:`~.index.GenericIndex` objects declared as
        inputs or outputs.
    instance_dict:
        ``self.__dict__`` of the model instance at the time
        ``super().__init__()`` is called, i.e. after all ``self.* =``
        assignments have been made.
    indexes:
        The full flat index list of the model.  Every declared index must
        appear here (identity check).
    model_name:
        Used in error messages.

    Returns
    -------
    IOProxy
        Proxy whose attribute keys are the Python attribute names found in
        *instance_dict*.

    Raises
    ------
    ValueError
        If a declared index is not in *indexes*, or if no attribute holding
        that index can be found in *instance_dict*, or if two declared indexes
        share the same attribute name.
    """
    index_ids = {id(idx) for idx in indexes}

    # Build a reverse map: id(index) -> attr_name from the instance dict.
    # When multiple attributes point to the same object we take the first one
    # found (dict insertion order, i.e. assignment order).
    id_to_attr: dict[int, str] = {}
    for attr, val in instance_dict.items():
        if isinstance(val, GenericIndex) and id(val) not in id_to_attr:
            id_to_attr[id(val)] = attr

    entries: list[tuple[str, _ProxyValue]] = []
    seen_attrs: dict[str, int] = {}  # attr_name -> position, for collision detection

    for idx in declarations:
        # 1. Must be in indexes.
        if id(idx) not in index_ids:
            raise ValueError(
                f"Model {model_name!r}: {label} entry "
                f"{getattr(idx, 'name', repr(idx))!r} "
                f"is not in the model's indexes list."
            )

        # 2. Must be findable as an instance attribute.
        attr = id_to_attr.get(id(idx))
        if attr is None:
            raise ValueError(
                f"Model {model_name!r}: {label} entry "
                f"{getattr(idx, 'name', repr(idx))!r} "
                f"is not assigned to any attribute of the model instance. "
                f"Assign it to 'self.<name>' before calling super().__init__()."
            )

        # 3. No two declared entries may share the same attribute name.
        if attr in seen_attrs:
            raise ValueError(
                f"Model {model_name!r}: {label} attribute name {attr!r} "
                f"collision — two declared entries map to the same attribute."
            )
        seen_attrs[attr] = len(entries)
        entries.append((attr, idx))

    return IOProxy(entries, dc=None)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Model:
    """A named collection of :class:`~.index.GenericIndex` objects with an optional I/O contract.

    Two APIs are supported:

    **New dataclass-based API** (recommended)
        Declare ``Inputs``, ``Outputs``, and/or ``Expose`` as inner
        ``@dataclass`` classes on the subclass.  Construct instances of them
        and pass to ``super().__init__()``::

            from dataclasses import dataclass

            class MyModel(Model):

                @dataclass
                class Inputs:
                    inflow: TimeseriesIndex

                @dataclass
                class Outputs:
                    traffic: TimeseriesIndex
                    total:   Index

                @dataclass
                class Expose:
                    ratio: Index   # inspectable but not contractual

                def __init__(self, inflow: TimeseriesIndex) -> None:
                    Inputs  = MyModel.Inputs
                    Outputs = MyModel.Outputs
                    Expose  = MyModel.Expose

                    traffic = TimeseriesIndex("traffic", ...)
                    total   = Index("total", traffic.sum())
                    ratio   = Index("ratio", ...)

                    super().__init__(
                        "My Model",
                        inputs=Inputs(inflow=inflow),
                        outputs=Outputs(traffic=traffic, total=total),
                        expose=Expose(ratio=ratio),
                    )

            m = MyModel(inflow_ts)
            m.inputs.inflow    # the wired inflow index
            m.outputs.traffic  # contractual output
            m.expose.ratio     # inspectable, non-contractual

        The local aliases (``Inputs = MyModel.Inputs``, etc.) avoid the
        fully-qualified form inside ``__init__`` and keep the construction
        calls concise.

        ``Inputs`` and ``Outputs`` are plural because they name a *collection*
        of fields.  ``Expose`` is a verb-derived noun (like ``Meta`` or
        ``Config``) — "Exposes" would be grammatically wrong — so it stays
        singular by design.

        Dataclass fields may hold a single :class:`~.index.GenericIndex`, a
        ``list`` of them, or a ``dict`` mapping strings to them.  The flat
        ``indexes`` list is derived automatically by collecting and
        deduplicating all scalar indexes from ``inputs``, ``outputs``, and
        ``expose``.

    **Legacy API** (deprecated)
        Pass a flat ``indexes`` list directly, along with optional
        ``inputs`` and ``outputs`` as plain ``list[GenericIndex]``::

            super().__init__("My Model", indexes=[a, b, c], outputs=[b])

        Every entry in ``inputs``/``outputs`` must be assigned to a
        ``self.*`` attribute before ``super().__init__()`` is called.
        A :class:`DeprecationWarning` is emitted when ``indexes`` is
        provided explicitly.

    Parameters
    ----------
    name:
        Human-readable name for the model.
    indexes:
        *Deprecated.* Explicit flat index list.  If omitted, the list is
        derived from the dataclass arguments.
    inputs:
        Dataclass instance (new API) or ``list[GenericIndex]`` (legacy).
    outputs:
        Dataclass instance (new API) or ``list[GenericIndex]`` (legacy).
    expose:
        Dataclass instance (new API) or ``None``.  Ignored in legacy mode.

    Notes
    -----
    **Three access levels** (new API)

    1. ``model.outputs.<field>`` / ``model.inputs.<field>`` — declared
       public interface.  Stable and contractual.
    2. ``model.expose.<field>`` — inspectable but not contracted.
    3. Purely local variables inside ``__init__`` — engine-internal only;
       not accessible from outside.
    """

    def __init__(
        self,
        name: str,
        indexes: list[GenericIndex] | None = None,
        *,
        inputs: Any | None = None,
        outputs: Any | None = None,
        expose: Any | None = None,
    ) -> None:
        self.name = name

        if indexes is not None:
            # ------------------------------------------------------------------
            # Legacy path — explicit flat index list provided
            # ------------------------------------------------------------------
            warnings.warn(
                "Passing 'indexes' explicitly is deprecated and will be removed in a future version. "
                "Use the dataclass-based inputs/outputs/expose API instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            self.indexes: list[GenericIndex] = indexes

            # Capture instance __dict__ before we assign self.inputs / self.outputs
            # so those names do not pollute the attr lookup.
            instance_dict = dict(self.__dict__)
            instance_dict.pop("name", None)
            instance_dict.pop("indexes", None)

            # In legacy mode inputs/outputs are expected to be list[GenericIndex] | None.
            legacy_inputs: list[GenericIndex] = inputs if isinstance(inputs, list) else []
            legacy_outputs: list[GenericIndex] = outputs if isinstance(outputs, list) else []

            self.inputs: IOProxy[Any] = _build_proxy("inputs", legacy_inputs, instance_dict, indexes, name)
            self.outputs: IOProxy[Any] = _build_proxy("outputs", legacy_outputs, instance_dict, indexes, name)
            self.expose: IOProxy[Any] = IOProxy([])

        else:
            # ------------------------------------------------------------------
            # New dataclass-based path
            # ------------------------------------------------------------------
            self.indexes = _collect_indexes(inputs, outputs, expose)

            self.inputs = _proxy_from_dataclass(inputs) if inputs is not None else IOProxy([])  # type: ignore[assignment]
            self.outputs = _proxy_from_dataclass(outputs) if outputs is not None else IOProxy([])  # type: ignore[assignment]
            self.expose = _proxy_from_dataclass(expose) if expose is not None else IOProxy([])  # type: ignore[assignment]

    def abstract_indexes(self) -> list[GenericIndex]:
        """Return indexes that require external values before evaluation.

        An index is abstract when its ``value`` is ``None`` (explicit
        placeholder) or a :class:`~.index.Distribution` (needs sampling).
        Constant and formula-based indexes are concrete and are not returned.

        Returns
        -------
        list[GenericIndex]
            All abstract indexes belonging to this model.

        Notes
        -----
        ``inputs`` may include concrete indexes (e.g. a data timeseries wired
        in from a parent model), and not every abstract index need be declared
        as an input (e.g. a distribution-backed behavioural parameter sampled
        internally by the ensemble).
        """
        result = []
        for index in self.indexes:
            if isinstance(index, Index):
                if index.value is None or isinstance(index.value, Distribution):
                    result.append(index)
        return result

    def is_instantiated(self) -> bool:
        """Return ``True`` when all indexes have concrete, evaluable values.

        Returns
        -------
        bool
            ``True`` if :meth:`abstract_indexes` is empty.
        """
        return len(self.abstract_indexes()) == 0
