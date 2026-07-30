"""Core model definition."""

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import inspect
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from ..engine.frontend import graph
from ..engine.numpybackend.executor import Functor
from .index import GenericIndex, Index


class ModelContractViolation(Exception):
    """Common base for any :class:`Model` I/O contract violation, soft or hard.

    Subclassed by both :class:`ModelContractWarning` (soft, routed through
    :func:`warnings.warn` and therefore filterable) and
    :class:`ModelContractError` (hard, always raised directly).  Catch this
    to handle any contract violation regardless of severity::

        try:
            SomeModel(...)
        except ModelContractViolation:
            ...

    Catching this does not, by itself, turn soft warnings into exceptions —
    it only catches instances that are already being raised, either because
    they are hard errors or because a :mod:`warnings` filter escalated them.
    It inherits from :class:`Exception` (rather than being a bare mixin)
    solely so that it is itself a valid ``except``/``pytest.raises`` target;
    it is never raised or emitted directly.
    """


class ModelContractWarning(ModelContractViolation, UserWarning):
    """Base class for all soft :class:`Model` I/O contract warnings.

    Subclass this to introduce new soft contract-violation categories.  Using
    a common base lets the remaining soft warnings in the family be turned
    into errors in a test suite with a single filter::

        warnings.filterwarnings("error", category=ModelContractWarning)

    or silenced in a legacy codebase::

        warnings.filterwarnings("ignore", category=ModelContractWarning)

    Each subclass remains independently filterable for fine-grained control.
    """


class ModelContractError(ModelContractViolation):
    """Base class for all hard :class:`Model` I/O contract errors.

    Subclass this to introduce new hard contract-violation categories.
    Unlike :class:`ModelContractWarning`, instances are raised directly
    rather than routed through :func:`warnings.warn`, so they bypass the
    warnings-filter machinery entirely: ``warnings.filterwarnings`` has no
    effect on them.  Fix the offending code instead of trying to silence it.
    """


class InputsContractError(ModelContractError):
    """Raised when a :class:`Model` subclass receives an undeclared :class:`~.index.GenericIndex` parameter.

    Specifically, this is raised when a constructor parameter holds a
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
    """


class AbstractIndexNotInInputsError(ModelContractError):
    """Raised when an abstract index is not declared in a :class:`Model`'s ``Inputs``.

    Abstract indexes receive their values from outside the model (via the
    ensemble or a parent model's scenario assignments).  They are therefore
    inputs by definition and must be declared in the ``Inputs`` dataclass
    so that the data-flow contract is explicit and cross-variant consistency
    checks work correctly.

    The canonical fix is to declare the abstract index as a field of
    ``Inputs`` and wire it through ``super().__init__(inputs=Inputs(...))``.
    """


class InputsTypeMismatchError(ModelContractError):
    """Raised when the ``inputs`` argument is not an instance of the subclass's own ``Inputs``.

    Specifically, this is raised when the ``inputs`` value passed to a
    :class:`Model` subclass's constructor is not an instance of that
    subclass's own declared ``Inputs`` dataclass — most commonly, passing
    another model's ``Inputs`` by mistake (e.g. ``ParkingModel(inputs=
    OtherModel.Inputs(...))``).

    This check exists because two unrelated ``Inputs`` dataclasses can
    coincidentally share the same field names and types, in which case the
    mistake would not otherwise raise at all: the wrong data would be
    silently wired into the model.  Checking the type explicitly turns that
    silent miswiring (or a confusing :class:`AttributeError` deep inside
    ``compute()``, when the shapes differ) into an immediate, unambiguous
    error at the model boundary.
    """


class FunctionsTypeMismatchError(ModelContractError):
    """Raised when the ``functions`` argument is not an instance of the subclass's own ``Functions``.

    The :class:`Functions` analogue of :class:`InputsTypeMismatchError`:
    raised when a model that declares a ``@functions`` inner class is
    constructed with a ``fns``/``functions`` value that is not an instance of
    that declared ``Functions`` class — most commonly another model's
    ``Functions`` by mistake, or an unrelated object.

    Without this check the mistake is *silently* absorbed: a value lacking the
    ``_is_functions`` marker is dropped entirely (the model's own function map
    stays empty), and a different model's ``Functions`` is accepted and mapped
    by field name — either way the failure only surfaces much later, as a
    missing-function error at evaluation time, far from its cause.  Checking
    the type at construction turns that into an immediate, located error.
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
        A single :class:`~.index.GenericIndex`, a ``list`` of them, a
        ``dict`` mapping strings to them, or a nested ``@expose`` or
        ``@outputs``-decorated dataclass instance (recurses into its fields).

    Yields
    ------
    GenericIndex
        Each scalar index in declaration order.
    """
    _dc = getattr(value, "_dc", None)
    if _dc is not None and getattr(type(_dc), "_is_expose", False):
        # IOProxy wrapping an @expose dataclass — its __iter__ yields scalars
        yield from value  # type: ignore[misc]
    elif _dc is not None and getattr(type(_dc), "_is_outputs", False):
        # IOProxy wrapping an @outputs dataclass — surface sub-model outputs for inspection
        yield from value  # type: ignore[misc]
    elif getattr(type(value), "_is_expose", False) or getattr(type(value), "_is_outputs", False):
        # raw @expose or @outputs dataclass passed directly — recurse into fields
        for field in dataclasses.fields(value):  # type: ignore[arg-type]
            yield from _iter_scalars(getattr(value, field.name))
    elif isinstance(value, dict):
        yield from value.values()
    elif isinstance(value, list):
        yield from value
    else:
        yield value  # type: ignore[arg-type]


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
    """Raise if any ``GenericIndex`` constructor parameter is absent from ``inputs``.

    Walks the parameter list of *caller_cls*``.__init__`` (excluding ``self``),
    looks up the corresponding value in *caller_frame*'s locals, and checks
    that every scalar :class:`~.index.GenericIndex` found there is also
    reachable via *inputs_proxy*.  Parameters whose values are not
    :class:`~.index.GenericIndex` objects (e.g. ``str``, ``np.ndarray``,
    ``pd.DataFrame``) are silently skipped.

    :class:`InputsContractError` is raised, naming every violating
    parameter, if at least one is found.

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

    missing: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        value = local_vars.get(param_name, inspect.Parameter.empty)
        if value is inspect.Parameter.empty:
            continue

        # Collect all scalar GenericIndex objects from this parameter value.
        # Handles scalar, list[Index], and dict[str, Index] shapes.
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

    if missing:
        names = ", ".join(repr(entry) for entry in missing)
        raise InputsContractError(
            f"{caller_cls.__name__}: parameter(s) {names} hold a GenericIndex "
            f"that is not declared in Inputs.  "
            f"Add each to a field of {caller_cls.__name__}.Inputs and include it "
            f"in the inputs=... passed to super().__init__()."
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Model:
    """A named collection of :class:`~.index.GenericIndex` objects with an optional I/O contract.

    Three APIs are supported, from most to least recommended:

    **`@define` + `compute()` (recommended for leaf models)**
        Declare ``Inputs``, ``Outputs``, and optionally ``Expose`` and
        ``Functions`` as inner classes decorated with :func:`~.contracts.inputs`,
        :func:`~.contracts.outputs`, :func:`~.contracts.expose`, and
        :func:`~.contracts.functions`.  Implement ``compute()`` to return the
        outputs; :func:`~.contracts.define` generates the ``__init__``
        automatically::

            from civic_digital_twins.dt_model import Index, Model, define, inputs, outputs

            @define("My Model")
            class MyModel(Model):

                @inputs
                class Inputs:
                    inflow: TimeseriesIndex

                @outputs
                class Outputs:
                    traffic: TimeseriesIndex
                    total:   Index

                def compute(self, inputs: Inputs) -> Outputs:
                    traffic = TimeseriesIndex("traffic", ...)
                    total   = Index("total", traffic.sum())
                    return MyModel.Outputs(traffic=traffic, total=total)

            m = MyModel(inputs=MyModel.Inputs(inflow=inflow_ts))
            m.inputs.inflow    # the wired inflow index
            m.outputs.traffic  # contractual output

    **Contract decorators + `__init__` (composite / root models)**
        Use :func:`~.contracts.inputs`, :func:`~.contracts.outputs`, and
        :func:`~.contracts.expose` on the inner classes, then write ``__init__``
        manually and call ``super().__init__()``.  Required for models that
        assign sub-model attributes before ``super().__init__()`` is called.
        A subclass that defines ``__init__`` directly without ``legacy=True``
        raises ``TypeError`` at class-definition time.  Pass ``legacy=True``
        to opt in; this escape hatch itself emits a ``DeprecationWarning``
        and is staged for removal in a future milestone::

            from civic_digital_twins.dt_model import Model, inputs, outputs

            class RootModel(Model, legacy=True):

                @inputs
                class Inputs:
                    inflow: TimeseriesIndex

                @outputs
                class Outputs:
                    traffic: TimeseriesIndex

                def __init__(self) -> None:
                    self.leaf = LeafModel(...)
                    super().__init__(
                        "Root",
                        inputs=RootModel.Inputs(inflow=...),
                        outputs=RootModel.Outputs(traffic=...),
                    )

        Dataclass fields may hold a single :class:`~.index.GenericIndex`, a
        ``list`` of them, or a ``dict`` mapping strings to them.  The flat
        ``indexes`` list is derived automatically from ``inputs``, ``outputs``,
        and ``expose``.

    Parameters
    ----------
    name:
        Human-readable name for the model.
    inputs:
        Instance of an ``@inputs``-decorated dataclass, or ``None``.
    outputs:
        Instance of an ``@outputs``-decorated dataclass, or ``None``.
    expose:
        Instance of an ``@expose``-decorated dataclass, or ``None``.
    functions:
        Instance of a ``@functions``-decorated class declaring the custom
        functors this model requires.  At construction time the model
        performs a backward BFS from its output nodes to claim each
        matching ``function_call`` graph node; the resulting
        ``_node_functions`` map is injected into the executor ``State``
        at evaluation time.  Pass ``None`` (default) when the model uses
        no explicit function contract.

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
    ``Inputs``, :class:`InputsContractError` is raised.  Separately, if the
    ``inputs`` argument itself is not an instance of the subclass's own
    ``Inputs`` (e.g. a different model's ``Inputs`` passed by mistake),
    :class:`InputsTypeMismatchError` is raised; likewise, if the
    ``functions`` argument is not an instance of the subclass's own declared
    ``Functions``, :class:`FunctionsTypeMismatchError` is raised.
    """

    def __init_subclass__(cls, *, legacy: bool = False, **kwargs: Any) -> None:
        """Reject or warn when a subclass defines ``__init__`` directly instead of using ``@define``.

        Fired at class-definition time (before class decorators run), so
        ``@define``-decorated classes — which have no ``__init__`` at that point
        — are not affected.

        Parameters
        ----------
        legacy:
            Pass ``True`` to opt into hand-written ``__init__`` for models that
            cannot be expressed with :func:`~.contracts.define` + ``compute()``
            (e.g. composite models that assign sub-model attributes before
            calling ``super().__init__()``).  This escape hatch is itself
            deprecated and staged for removal in a future milestone: it emits
            a :class:`DeprecationWarning` rather than suppressing one.

        Raises
        ------
        TypeError
            If ``__init__`` is defined directly without ``legacy=True``.
        """
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            if not legacy:
                raise TypeError(
                    f"{cls.__name__} defines __init__ directly. "
                    "Use @define with compute() instead, or pass legacy=True to opt in "
                    "(deprecated; staged for removal in a future milestone)."
                )
            warnings.warn(
                f"{cls.__name__} uses legacy=True to define __init__ directly. "
                "This escape hatch is deprecated and will be removed in a future milestone. "
                "Use @define with compute() instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    def __init__(  # pyright: ignore[reportRedeclaration]
        self,
        name: str,
        *,
        inputs: Any | None = None,
        outputs: Any | None = None,
        expose: Any | None = None,
        functions: Any | None = None,
    ) -> None:
        self.name = name

        concrete_cls = type(self)
        if concrete_cls is not Model and inputs is not None:
            _declared_inputs_cls = concrete_cls.__dict__.get("Inputs")
            if _declared_inputs_cls is not None and not isinstance(inputs, _declared_inputs_cls):
                raise InputsTypeMismatchError(
                    f"{concrete_cls.__name__} expected inputs of type "
                    f"{_declared_inputs_cls.__qualname__}, got "
                    f"{type(inputs).__qualname__} instead."
                )

        if concrete_cls is not Model and functions is not None:
            _declared_functions_cls = concrete_cls.__dict__.get("Functions")
            if _declared_functions_cls is not None and not isinstance(functions, _declared_functions_cls):
                raise FunctionsTypeMismatchError(
                    f"{concrete_cls.__name__} expected functions of type "
                    f"{_declared_functions_cls.__qualname__}, got "
                    f"{type(functions).__qualname__} instead."
                )

        self.indexes = _collect_indexes(inputs, outputs, expose)

        # Typed as Any (not IOProxy): these are read-only, attribute-access
        # dynamic proxies whose field access already yields Any, and declaring
        # them Any lets an @expose contract surface a sub-model's outputs/expose
        # proxy into a field annotated with the wrapped dataclass type (e.g.
        # ``parking: ParkingModel.Outputs``) without leaking the internal
        # IOProxy type into user-facing annotations.
        self.inputs: Any = _proxy_from_dataclass(inputs) if inputs is not None else IOProxy([])
        self.outputs: Any = _proxy_from_dataclass(outputs) if outputs is not None else IOProxy([])
        self.expose: Any = _proxy_from_dataclass(expose) if expose is not None else IOProxy([])

        # Build node-function map: collect sub-model claims first, then this
        # model's own @functions declarations (closest-ancestor wins).
        _scan = {k: v for k, v in self.__dict__.items() if k not in ("name", "indexes", "inputs", "outputs", "expose")}
        _submodel_fns = _collect_submodel_node_functions(_scan)
        if functions is not None and getattr(type(functions), "_is_functions", False):
            self._node_functions = _build_node_functions_map(self.indexes, self.inputs, functions, _submodel_fns)
        else:
            self._node_functions = _submodel_fns

        # Convention check: every GenericIndex constructor parameter should
        # be declared in Inputs.  We inspect the immediate caller's frame
        # (the subclass __init__ that called super().__init__()) and warn
        # for any parameter whose value is a GenericIndex not found in
        # self.inputs.  The check is skipped for Model itself.
        if concrete_cls is not Model:
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame is not None else None
            if caller_frame is not None:
                _check_inputs_contract(caller_frame, concrete_cls, self.inputs)

            # Dropped-index check: any graph.placeholder or
            # graph.timeseries_placeholder node that is reachable from the
            # model's internally-built formula nodes but not itself covered
            # by a declared index will never receive a value at evaluation time.
            # This catches both sub-model concrete indexes and inline
            # Index(name, scalar) / TimeseriesIndex(name, array) created
            # inside compute() but not surfaced via Inputs/Outputs/Expose.
            # Formula-backed input nodes are excluded from the traversal
            # boundary: their placeholder dependencies belong to the model
            # that built them (the parent or a sibling sub-model).
            _input_formula_nodes: tuple[graph.Node, ...] = tuple(
                idx.node
                for idx in self.inputs
                if not isinstance(idx.node, (graph.placeholder, graph.timeseries_placeholder))
            )
            _orphaned = _find_orphaned_placeholder_nodes(self.indexes, _input_formula_nodes)
            if _orphaned:
                _names = ", ".join(repr(n.name) for n in _orphaned)
                raise ValueError(
                    f"{concrete_cls.__name__}: the following indexes appear in "
                    f"the model's formulas but are not declared in Inputs, Outputs, or Expose: "
                    f"{_names}. "
                    "Without a declaration the model cannot inject their values at "
                    "evaluation time and will fail with a missing-value error. "
                    "Add each index to Inputs (if it is supplied from outside the "
                    "model), Outputs or Expose (if it is computed inside), or "
                    "replace it with ConstIndex / ConstTimeseriesIndex if its "
                    "value is fixed and should never be overridden."
                )

            _missing_abstract = [
                getattr(idx, "name", repr(idx)) for idx in self.abstract_indexes() if idx not in self.inputs
            ]
            if _missing_abstract:
                _names = ", ".join(repr(n) for n in _missing_abstract)
                raise AbstractIndexNotInInputsError(
                    f"{concrete_cls.__name__}: abstract index(es) {_names} not declared "
                    f"in Inputs. Abstract indexes receive their values from outside the "
                    f"model and must be declared in Inputs."
                )

    if TYPE_CHECKING:  # pragma: no cover

        class _DataclassInstance(Protocol):
            """Structural match for *any* dataclass instance.

            ``@inputs``/``@outputs``/``@expose`` are ``@dataclass_transform``
            decorators, so Pyright sees their instances as dataclasses and
            hence as matching this protocol.  Using it (instead of ``Any``)
            for the floor's ``inputs``/``outputs``/``expose`` lets the checker
            reject non-dataclass garbage (``inputs=1``, ``inputs="x"``) by
            default, while still accepting every real ``Inputs`` regardless of
            which model it belongs to.
            """

            __dataclass_fields__: ClassVar[dict[str, Any]]

        # Permissive constructor "floor" for static type checking only.
        #
        # ``@define`` synthesizes each subclass's ``__init__`` at *runtime*
        # (from ``compute()``'s signature), so Pyright never sees it and falls
        # back to this base signature.  It intentionally accepts every real
        # call shape — the ``@define`` form ``Model(inputs=..., fns=...)`` and
        # the base/``legacy=True`` form ``Model(name=..., functions=...)`` — so
        # that *constructing a model is green by default, with no per-model
        # annotation required*.  Both keyword names are real: the base
        # ``Model.__init__`` above uses ``functions``, while ``@define``'s
        # synthesized ``__init__`` uses ``fns`` — a model built either way must
        # type-check here, so both are listed.
        #
        # ``inputs``/``outputs``/``expose`` must always be passed by keyword
        # (every real call site does this); the floor does not accept them
        # positionally, so ``name`` can stay ``str`` instead of ``Any`` and
        # reject obviously-wrong first arguments.
        #
        # It is only a *floor*: it rejects non-dataclass ``inputs`` and unknown
        # keywords/arity, but does not know *which* model's ``Inputs`` is
        # correct (an ``Outputs`` instance would also pass here). Two
        # mechanisms cover that finer check:
        #   * runtime — ``InputsTypeMismatchError`` (raised above) catches a
        #     mismatched/cross-model ``Inputs`` for every model, always;
        #   * static (opt-in) — a model that wants full constructor checking
        #     adds its own ``if TYPE_CHECKING: def __init__(self, inputs:
        #     Inputs) -> None: ...`` stub, which overrides this floor.
        #
        # It obscures the real ``__init__`` above (hence the scoped
        # ``reportRedeclaration`` ignore there); the real one still runs.
        def __init__(
            self,
            name: str = "",
            *,
            inputs: _DataclassInstance | None = ...,
            outputs: _DataclassInstance | None = ...,
            expose: _DataclassInstance | None = ...,
            # ``functions``/``fns`` stay ``Any``: the ``@functions`` class is
            # hand-built (not a dataclass), so it does not match the protocol.
            #
            # They are also kept as two separate parameters rather than
            # unified into one: they are both real, independently load-bearing
            # runtime keyword names, not an artifact of this stub. The base
            # ``Model.__init__`` above takes ``functions`` (used by every
            # legacy hand-written ``__init__`` that forwards to
            # ``super().__init__(..., functions=fns)``); ``@define``'s
            # synthesized ``__init__`` takes ``fns`` (used by every
            # ``@define`` + ``@functions`` construction call, including
            # production examples). Renaming either to unify them would be a
            # breaking change to one of those two real call shapes, for a
            # purely cosmetic gain here. The ``functions`` name only exists
            # because of the ``legacy=True`` escape hatch, which is already
            # deprecated and staged for removal — once it is gone, this
            # parameter (and this split) goes with it for free. Tracked as a
            # follow-up, together with ``legacy=True`` removal itself.
            functions: Any = ...,
            fns: Any = ...,
        ) -> None: ...

    def abstract_indexes(self) -> list[GenericIndex]:
        """Return indexes that require external values before evaluation.

        Delegates to each index's own :attr:`~.index.Index.is_abstract` /
        :attr:`~.index.TimeseriesIndex.is_abstract` classification. Constant
        and formula-based indexes are concrete and are not returned.

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
        return [index for index in self.indexes if isinstance(index, Index) and index.is_abstract]

    def is_instantiated(self) -> bool:
        """Return ``True`` when all indexes have concrete, evaluable values.

        Returns
        -------
        bool
            ``True`` if :meth:`abstract_indexes` is empty.
        """
        return len(self.abstract_indexes()) == 0


# ---------------------------------------------------------------------------
# Functions contract helpers
# (defined after Model so _collect_submodel_node_functions can reference Model
# directly without a forward-reference workaround)
# ---------------------------------------------------------------------------


def _iter_node_deps(node: graph.Node) -> list[graph.Node]:
    """Return direct graph dependencies of *node* for backward traversal.

    This function is exhaustive over all non-leaf node types.  Leaf nodes
    (``constant``, ``placeholder``, ``timeseries_constant``,
    ``timeseries_placeholder``) have no dependencies and fall through to the
    ``return []`` at the end.

    **Maintenance note**: every new non-leaf ``graph.Node`` subclass must be
    handled here; omitting one will silently stop BFS traversal at that node,
    causing ``function_call`` nodes reachable through it to go unclaimed.
    """
    if isinstance(node, graph.BinaryOp):
        return [node.left, node.right]
    if isinstance(node, graph.UnaryOp):
        return [node.node]
    if isinstance(node, graph.where):
        return [node.condition, node.then, node.otherwise]
    if isinstance(node, graph.exclusive_multi_clause_where):
        deps: list[graph.Node] = []
        for cond, val in node.clauses:
            deps.append(cond)
            deps.append(val)
        deps.append(node.default_value)
        deps.append(node.companion)
        return deps
    if isinstance(node, graph.MultiClauseOp):
        deps = []
        for cond, val in node.clauses:
            deps.append(cond)
            deps.append(val)
        deps.append(node.default_value)
        return deps
    if isinstance(node, graph.variant_selector):
        deps = [node.selector_node]
        for branch_nodes in node.branch_map.values():
            deps.extend(branch_nodes)
        return deps
    if isinstance(node, graph.ProjectionOp):
        return [node.node]
    if isinstance(node, graph.function_call):
        return list(node.args) + list(node.kwargs.values())
    # Leaf nodes: constant, placeholder, timeseries_constant, timeseries_placeholder.
    return []


def _collect_submodel_node_functions(instance_dict: dict[str, Any]) -> dict[graph.Node, Functor]:
    """Collect ``_node_functions`` from ``Model``-typed attributes (one level deep).

    Handles three value shapes: a direct ``Model`` attribute, a ``list`` of
    ``Model`` instances, or a ``dict`` whose values are ``Model`` instances.
    Deeper nesting (e.g. ``dict[str, list[Model]]``) is not flattened.
    """
    claimed: dict[graph.Node, Functor] = {}
    for val in instance_dict.values():
        if isinstance(val, Model):
            claimed.update(val._node_functions)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, Model):
                    claimed.update(item._node_functions)
        elif isinstance(val, dict):
            for item in val.values():
                if isinstance(item, Model):
                    claimed.update(item._node_functions)
    return claimed


def _find_orphaned_placeholder_nodes(
    indexes: list[GenericIndex],
    input_formula_nodes: tuple["graph.Node", ...],
) -> list["graph.Node"]:
    """Find placeholder nodes reachable from *indexes* but not covered by any index.

    Traverses the computation graph **backward** from the model's internally-
    built formula nodes (outputs and expose) using :func:`_iter_node_deps`.
    Any :class:`~engine.frontend.graph.placeholder` or
    :class:`~engine.frontend.graph.timeseries_placeholder` node that is
    reachable but whose identity is *not* among ``{idx.node for idx in indexes}``
    is returned as **orphaned**.

    The traversal is bounded by two stopping conditions:

    * **Covered nodes** — nodes that already have a declared index in
      *indexes* (they receive their values via
      :meth:`~simulation.scenario.Scenario.base_substitutions`).
    * **Formula-backed input nodes** (*input_formula_nodes*) — formula nodes
      that were built by *another* model and passed to this one as inputs.
      These nodes' placeholder dependencies belong to that other model;
      traversing into them would produce false positives.

    An orphaned placeholder node is always a latent bug:

    * If it was created by a concrete-valued ``Index(name, 0.2)``, the value
      is stored only in the ``Index`` object and is injected by
      :meth:`~simulation.scenario.Scenario.base_substitutions`, which
      iterates ``model.indexes``.  A node absent from ``model.indexes`` is
      never injected → ``PlaceholderValueNotProvided``.
    * If it was created by an abstract ``Index(name, None)``, it is absent
      from :meth:`~.model.Model.abstract_indexes` and therefore unknown to
      the ensemble; evaluation fails for the same reason.

    Parameters
    ----------
    indexes:
        The model's declared indexes (``model.indexes``).
    input_formula_nodes:
        Formula-backed ``.node`` values from the model's declared inputs.  The
        BFS stops when it reaches any of these nodes (their dependencies were
        built by a different model and are that model's responsibility).

    Returns
    -------
    list[graph.Node]
        Orphaned placeholder nodes in discovery order.  Empty when every
        reachable placeholder is covered by a declared index.
    """
    # Boundary/visited sets use id() (graph.Node overrides __eq__), as elsewhere in this module.
    covered_nodes: list[graph.Node] = [idx.node for idx in indexes]
    covered_ids: set[int] = {id(n) for n in covered_nodes}
    input_formula_ids: set[int] = {id(n) for n in input_formula_nodes}
    # Start only from formula-backed nodes that were *built by this model*:
    # i.e., nodes that are covered but are NOT formula-backed inputs from another model.
    internal_formula_starts: list[graph.Node] = [
        node
        for node in covered_nodes
        if not isinstance(node, (graph.placeholder, graph.timeseries_placeholder)) and id(node) not in input_formula_ids
    ]
    visited_ids: set[int] = set()
    to_visit: list[graph.Node] = list(internal_formula_starts)
    orphaned: list[graph.Node] = []
    while to_visit:
        node = to_visit.pop()
        if id(node) in visited_ids:
            continue
        visited_ids.add(id(node))
        for dep in _iter_node_deps(node):
            if id(dep) in visited_ids or id(dep) in covered_ids or id(dep) in input_formula_ids:
                continue  # already handled or belongs to another model
            if isinstance(dep, (graph.placeholder, graph.timeseries_placeholder)):
                # placeholder nodes with a default_value are self-contained:
                # the executor falls back to that value when the node is absent
                # from state.values, so they are not orphaned.
                if isinstance(dep, graph.placeholder) and dep.default_value is not None:
                    continue
                orphaned.append(dep)  # uncovered placeholder — always a bug
            else:
                to_visit.append(dep)  # uncovered formula node — traverse further
    return orphaned


def _build_node_functions_map(
    indexes: list[GenericIndex],
    inputs_proxy: "IOProxy[Any]",
    functions: Any,
    claimed: dict[graph.Node, Functor],
) -> dict[graph.Node, Functor]:
    """Backward BFS from own output/expose nodes to find and claim ``function_call`` nodes.

    Traversal starts from all non-input index nodes and walks backward through
    graph dependencies.  Two stopping conditions prevent over-claiming:

    1. **Input placeholder nodes** — the ``.node`` of each index in *inputs_proxy*.
       These are the construction-time boundary; their dependencies were created
       outside this model.
    2. **Already-claimed nodes** — nodes in *claimed* (sub-models' ``_node_functions``).
       Stopping here prevents traversal into a sub-model's internal graph.

    The "closest-ancestor wins" rule is enforced by *claimed*: sub-model
    declarations populate it before this function is called, so a parent model
    cannot re-claim a node that a child already owns.

    Parameters
    ----------
    indexes:
        All ``GenericIndex`` objects belonging to this model.
    inputs_proxy:
        Proxy over the model's declared ``Inputs``; provides the input
        boundary nodes.
    functions:
        Instance of a ``@functions``-decorated class; ``functions.items()``
        yields ``(name, Functor)`` pairs for declared and extra fields.
    claimed:
        Pre-populated ``{node: Functor}`` map from sub-models.  Updated
        in-place is avoided — a new dict is returned.

    Returns
    -------
    dict[graph.Node, Functor]
        Merged map: sub-model claims unioned with this model's own claims.
    """
    fn_map: dict[str, Functor] = dict(functions.items())
    if not fn_map:
        return dict(claimed)

    # Boundary sets use id() for O(1) membership tests.
    input_ids: set[int] = {id(idx.node) for idx in inputs_proxy}
    claimed_ids: set[int] = {id(n) for n in claimed}
    stop_ids: set[int] = input_ids | claimed_ids

    # Start the traversal from every non-input index node.
    start_nodes: list[graph.Node] = [idx.node for idx in indexes if id(idx.node) not in input_ids]

    result: dict[graph.Node, Functor] = dict(claimed)
    visited_ids: set[int] = set()
    queue = list(start_nodes)

    while queue:
        node = queue.pop()
        nid = id(node)
        if nid in visited_ids:
            continue
        visited_ids.add(nid)

        # Stop at input boundaries and already-claimed nodes; do not recurse.
        if nid in stop_ids:
            continue

        # Claim this function_call node if its name is declared and it is unclaimed.
        if isinstance(node, graph.function_call) and nid not in claimed_ids:
            functor = fn_map.get(node.name)
            if functor is not None:
                result[node] = functor

        for dep in _iter_node_deps(node):
            if id(dep) not in visited_ids:
                queue.append(dep)

    return result
