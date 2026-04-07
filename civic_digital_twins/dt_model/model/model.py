"""Core model definition."""

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import inspect
import warnings
from collections.abc import Iterator
from typing import Any

from .index import Distribution, GenericIndex, Index


class ModelContractWarning(UserWarning):
    """Base class for all :class:`Model` I/O contract warnings.

    Subclass this to introduce new contract-violation categories.  Using a
    common base makes it easy to turn *all* contract warnings into errors in a
    test suite with a single filter::

        warnings.filterwarnings("error", category=ModelContractWarning)

    or to silence them all in a legacy codebase::

        warnings.filterwarnings("ignore", category=ModelContractWarning)

    Each subclass remains independently filterable for fine-grained control.
    """


class InputsContractWarning(ModelContractWarning):
    """Emitted when a :class:`Model` subclass receives an undeclared :class:`~.index.GenericIndex` parameter.

    Specifically, this warning fires when a constructor parameter holds a
    :class:`~.index.GenericIndex` value that is not declared in the ``Inputs``
    dataclass.

    The convention is that every :class:`~.index.GenericIndex` (or
    ``list`` / ``dict`` thereof) passed into a :class:`Model` subclass
    ``__init__`` must be stored in a field of the ``Inputs`` dataclass and
    forwarded to ``super().__init__(inputs=...)``.  This makes the data-flow
    contract explicit and enables the cross-variant consistency check performed
    by :class:`~.model_variant.ModelVariant`.

    ``Expose`` fields are intentionally excluded from this rule: they are
    meant to surface purely internal intermediates and are not part of the
    inter-model wiring contract.

    To silence this warning for a specific model, override ``__init__`` and
    suppress it with :func:`warnings.filterwarnings` before calling
    ``super().__init__()``.
    """


class AbstractIndexNotInInputsWarning(ModelContractWarning):
    """Emitted when an abstract index is not declared in a :class:`Model`'s ``Inputs``.

    Abstract indexes receive their values from outside the model (via the
    ensemble or a parent model's scenario assignments).  They are therefore
    inputs by definition and should be declared in the ``Inputs`` dataclass
    so that the data-flow contract is explicit and cross-variant consistency
    checks work correctly.

    This is a soft warning initially (not an error) for backwards
    compatibility with models that create abstract indexes internally and
    surface them via ``expose`` or flat ``indexes`` lists.  It is tracked
    for promotion to an error in a future release.

    The canonical fix is to declare the abstract index as a field of
    ``Inputs`` and wire it through ``super().__init__(inputs=Inputs(...))``.
    """


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


class IOProxy[DC]:
    """Read-only attribute-access proxy over a declared inputs, outputs, or expose mapping.

    The class is generic over the dataclass type *DC* it wraps.  When
    constructed from a dataclass instance (the normal path), field access
    ``proxy.field`` is typed as :data:`~typing.Any` by the type checker, which
    means the declared field type on *DC* flows through without any
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

    def __init__(self, entries: list[tuple[str, _ProxyValue]], dc: DC | None = None) -> None:
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


def _proxy_from_dataclass[DC](dc_instance: DC) -> IOProxy[DC]:
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
    IOProxy[DC]
        Proxy whose attribute keys are the dataclass field names.
    """
    entries: list[tuple[str, _ProxyValue]] = []
    for field in dataclasses.fields(dc_instance):  # type: ignore[arg-type]
        val: _ProxyValue = getattr(dc_instance, field.name)
        entries.append((field.name, val))
    return IOProxy(entries, dc=dc_instance)


def _check_inputs_contract(
    caller_frame: Any,
    caller_cls: type,
    inputs_proxy: IOProxy[Any],
) -> None:
    """Warn if any ``GenericIndex`` constructor parameter is absent from ``inputs``.

    Walks the parameter list of *caller_cls*``.__init__`` (excluding ``self``),
    looks up the corresponding value in *caller_frame*'s locals, and checks
    that every scalar :class:`~.index.GenericIndex` found there is also
    reachable via *inputs_proxy*.  Parameters whose values are not
    :class:`~.index.GenericIndex` objects (e.g. ``str``, ``np.ndarray``,
    ``pd.DataFrame``) are silently skipped.

    A :class:`InputsContractWarning` is emitted for each violating parameter.

    Parameters
    ----------
    caller_frame:
        The ``f_back`` frame of ``Model.__init__`` — i.e. the frame of the
        subclass ``__init__`` that called ``super().__init__()``.
    caller_cls:
        The concrete :class:`Model` subclass being constructed.
    inputs_proxy:
        The already-built ``self.inputs`` proxy to check against.
    """
    try:
        sig = inspect.signature(caller_cls.__init__)
    except (ValueError, TypeError):
        return

    local_vars: dict[str, Any] = caller_frame.f_locals

    # Build the set of all GenericIndex node ids reachable through inputs.
    inputs_ids: set[int] = {id(idx) for idx in inputs_proxy}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        value = local_vars.get(param_name, inspect.Parameter.empty)
        if value is inspect.Parameter.empty:
            continue

        # Collect all scalar GenericIndex objects from this parameter value.
        # Handles scalar, list[Index], and dict[str, Index] shapes.
        missing: list[str] = []
        if isinstance(value, GenericIndex):
            if id(value) not in inputs_ids:
                missing.append(param_name)
        elif isinstance(value, list) and value and isinstance(value[0], GenericIndex):
            for i, item in enumerate(value):
                if isinstance(item, GenericIndex) and id(item) not in inputs_ids:
                    missing.append(f"{param_name}[{i}]")
        elif isinstance(value, dict):
            for k, item in value.items():
                if isinstance(item, GenericIndex) and id(item) not in inputs_ids:
                    missing.append(f"{param_name}[{k!r}]")

        for entry in missing:
            warnings.warn(
                f"{caller_cls.__name__}: parameter {entry!r} holds a GenericIndex "
                f"that is not declared in Inputs.  "
                f"Add it as a field of {caller_cls.__name__}.Inputs and include it "
                f"in the inputs=... passed to super().__init__().",
                InputsContractWarning,
                stacklevel=4,
            )


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
       ``Expose`` is intended for purely diagnostic intermediates
       and must **not** be used to wire indexes into sibling or parent
       models.
    3. Purely local variables inside ``__init__`` — engine-internal only;
       not accessible from outside.

    **Inputs contract convention**

    Every :class:`~.index.GenericIndex` (or ``list`` / ``dict`` thereof)
    received as a constructor parameter must be declared as a field of the
    ``Inputs`` dataclass and forwarded via ``inputs=Inputs(...)`` to
    ``super().__init__()``.  This rule makes the inter-model data-flow
    contract explicit and enables the cross-variant consistency check
    performed by :class:`~.model_variant.ModelVariant`.

    At construction time, :class:`Model` checks this convention
    automatically: if a constructor parameter holds a
    :class:`~.index.GenericIndex` value that is absent from the declared
    ``Inputs``, an :class:`InputsContractWarning` is emitted.  The warning
    is a soft reminder rather than an error, so existing models continue to
    work while the contract is incrementally tightened.
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

            # Convention check: every GenericIndex constructor parameter should
            # be declared in Inputs.  We inspect the immediate caller's frame
            # (the subclass __init__ that called super().__init__()) and warn
            # for any parameter whose value is a GenericIndex not found in
            # self.inputs.  The check is skipped for Model itself.
            concrete_cls = type(self)
            if concrete_cls is not Model:
                frame = inspect.currentframe()
                caller_frame = frame.f_back if frame is not None else None
                if caller_frame is not None:
                    _check_inputs_contract(caller_frame, concrete_cls, self.inputs)

                for idx in self.abstract_indexes():
                    if idx not in self.inputs:
                        idx_name = getattr(idx, "name", repr(idx))
                        warnings.warn(
                            f"{concrete_cls.__name__}: abstract index {idx_name!r} is not "
                            f"declared in Inputs. Abstract indexes receive their values from "
                            f"outside the model and should be declared in Inputs.",
                            AbstractIndexNotInInputsWarning,
                            stacklevel=3,
                        )

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
