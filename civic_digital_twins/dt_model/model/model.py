"""Core model definition."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .index import Distribution, GenericIndex, Index


class IOProxy:
    """Read-only attribute-access proxy over a declared inputs or outputs list.

    Each index is accessible by the Python attribute name under which it was
    assigned on the model instance (e.g. ``self.inflow`` → ``proxy.inflow``).
    ``index.name`` is treated as a free-form display label and plays no role
    here.

    Supports:

    * Attribute access  — ``proxy.inflow``
    * Iteration         — ``for idx in proxy``
    * ``len(proxy)``
    * ``idx in proxy``  — identity-based membership test
    * ``repr(proxy)``
    """

    def __init__(self, entries: list[tuple[str, GenericIndex]]) -> None:
        # entries is an ordered list of (attr_name, index) pairs.
        # We use object.__setattr__ throughout to avoid triggering our own
        # __setattr__ override.
        object.__setattr__(self, "_entries", entries)
        object.__setattr__(self, "_map", {key: idx for key, idx in entries})

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> GenericIndex:
        """Return the index registered under *name*."""
        mapping: dict[str, GenericIndex] = object.__getattribute__(self, "_map")
        if name in mapping:
            return mapping[name]
        raise AttributeError(f"No input/output with attribute name {name!r}. Available: {list(mapping)}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Raise AttributeError — IOProxy is read-only."""
        raise AttributeError("IOProxy is read-only.")

    # ------------------------------------------------------------------
    # Iteration / containment
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[GenericIndex]:
        """Iterate over declared indexes in declaration order."""
        entries: list[tuple[str, GenericIndex]] = object.__getattribute__(self, "_entries")
        return (idx for _, idx in entries)

    def __len__(self) -> int:
        """Return the number of declared indexes."""
        entries: list[tuple[str, GenericIndex]] = object.__getattribute__(self, "_entries")
        return len(entries)

    def __contains__(self, item: object) -> bool:
        """Return True if *item* is one of the declared indexes (identity check)."""
        entries: list[tuple[str, GenericIndex]] = object.__getattribute__(self, "_entries")
        return any(idx is item for _, idx in entries)

    def __repr__(self) -> str:
        """Return a string representation listing the declared attribute names."""
        entries: list[tuple[str, GenericIndex]] = object.__getattribute__(self, "_entries")
        return f"IOProxy({[key for key, _ in entries]})"


def _build_proxy(
    label: str,
    declarations: list[GenericIndex],
    instance_dict: dict[str, Any],
    indexes: list[GenericIndex],
    model_name: str,
) -> IOProxy:
    """Build an :class:`IOProxy` for a declared inputs or outputs list.

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

    entries: list[tuple[str, GenericIndex]] = []
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

    return IOProxy(entries)


class Model:
    """A named collection of :class:`~.index.GenericIndex` objects.

    Optionally carries a declared input/output interface.

    A model is *abstract* if it contains indexes that need external values
    before evaluation — placeholder nodes (``value is None``) or
    distribution-backed indexes (``isinstance(value, Distribution)``).
    It is *instantiated* when every index has a concrete, evaluable value.

    Parameters
    ----------
    name:
        Human-readable name for the model.
    indexes:
        The complete flat list of indexes belonging to this model, used by
        the evaluation engine to resolve transitive dependencies.  Includes
        inputs, outputs, and all other indexes — whether assigned to
        ``self.*`` attributes or held as local variables inside ``__init__``.
    inputs:
        Optional declared subset of *indexes* that this model expects to be
        provided from outside — either wired from a parent model or assigned
        by the caller before evaluation.  Every entry must be assigned to a
        ``self.*`` attribute of this instance before ``super().__init__()``
        is called.  Accessible as ``model.inputs.<attr_name>``.
    outputs:
        Optional declared subset of *indexes* that this model exposes as its
        public results — the values a parent model or caller should consume.
        Same rules as *inputs*.  Accessible as ``model.outputs.<attr_name>``.

    Notes
    -----
    **Three access levels**

    CDT models follow a three-level access convention:

    1. ``model.outputs.<attr>`` / ``model.inputs.<attr>`` — the declared
       public interface.  Stable, contractual.  Parent models and callers
       should only rely on these.

    2. ``model.<attr>`` — any index assigned to a ``self.*`` attribute but
       not declared in *inputs* or *outputs*.  Inspectable from outside
       (useful for debugging, visualisation, exploration) but not part of
       the contractual interface.

    3. Local variables inside ``__init__`` — indexes known only to the
       evaluation engine via *indexes*.  Treated as fully internal
       implementation details; not accessible from outside the model.

    **Access keys**

    The key used to access an index via ``model.inputs`` or ``model.outputs``
    is the Python attribute name under which it was assigned on the model
    instance (``self.<attr> = Index(...)``).  ``index.name`` is a free-form
    display label (used in plots and reports) and plays no role here.

    Example::

        class MyModel(Model):
            def __init__(self):
                # Level 1 — declared output (contractual)
                self.traffic = Index("Reference traffic", 200.0)
                # Level 2 — named but not contracted (inspectable)
                self.inflow  = Index("Total vehicle inflow", 100.0)
                # Level 3 — anonymous (internal to engine only)
                _base = Index("base", 50.0)

                super().__init__(
                    "My Model",
                    indexes=[self.traffic, self.inflow, _base],
                    outputs=[self.traffic],
                )

        m = MyModel()
        m.outputs.traffic  # level 1 — contractual access
        m.inflow           # level 2 — inspectable, not contracted
        # _base is unreachable from outside

    **Validation**

    At construction time:

    * Every entry in *inputs* and *outputs* must appear in *indexes*
      (identity check).
    * Every entry must be assigned to a ``self.*`` attribute of the instance
      at the time ``super().__init__()`` is called.
    * No two entries in the same list may map to the same attribute name.
    """

    def __init__(
        self,
        name: str,
        indexes: list[GenericIndex],
        *,
        inputs: list[GenericIndex] | None = None,
        outputs: list[GenericIndex] | None = None,
    ) -> None:
        self.name = name
        self.indexes = indexes

        # Capture the instance __dict__ *before* we assign self.inputs /
        # self.outputs, so that those names do not pollute the attr lookup.
        instance_dict = dict(self.__dict__)
        # Remove keys we have just set that are not index attributes.
        instance_dict.pop("name", None)
        instance_dict.pop("indexes", None)

        self.inputs: IOProxy = _build_proxy("inputs", inputs or [], instance_dict, indexes, name)
        self.outputs: IOProxy = _build_proxy("outputs", outputs or [], instance_dict, indexes, name)

    def abstract_indexes(self) -> list[GenericIndex]:
        """Return indexes that require external values before evaluation.

        An index is abstract when its ``value`` is ``None`` (explicit
        placeholder) or a ``Distribution`` (needs sampling).  Constant and
        formula-based indexes are concrete and are not returned here.

        Note: ``inputs`` may include concrete indexes (e.g. a data timeseries
        that a parent model wires in), and not every abstract index need be
        declared as an input (e.g. a distribution-backed behavioral parameter
        sampled internally by the ensemble).
        """
        result = []
        for index in self.indexes:
            if isinstance(index, Index):
                if index.value is None or isinstance(index.value, Distribution):
                    result.append(index)
        return result

    def is_instantiated(self) -> bool:
        """Return True when all indexes have concrete, evaluable values."""
        return len(self.abstract_indexes()) == 0
