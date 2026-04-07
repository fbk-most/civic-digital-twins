"""ModelVariant — selects among Model subclasses sharing the same I/O contract."""

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from ..engine.frontend import graph
from .index import CategoricalIndex, GenericIndex, Index
from .model import IOProxy, Model

__all__ = ["ModelVariant"]


class ModelVariant:
    """Selects among :class:`Model` subclasses sharing the same I/O contract.

    Operates in two modes depending on the *selector* type:

    **Static mode** (``selector: str``) — resolves to exactly one variant at
    construction time and acts as a fully transparent proxy for it.  Zero
    overhead: inactive variants do not appear in the graph at all.

    **Runtime mode** (``selector: CategoricalIndex | graph.Node``) — all
    variants are preserved in the graph.  A :class:`~engine.frontend.graph.variant_selector`
    node and one :class:`~engine.frontend.graph.exclusive_multi_clause_where`
    node per output field are created at construction time.  The evaluation
    layer's ``_build_plan`` / ``_execute_plan`` path handles per-scenario
    dispatch efficiently.

    Parameters
    ----------
    name:
        Human-readable name for the variant group.
    variants:
        Mapping from string key to an already-constructed :class:`Model`
        instance.  All variants must share the same ``outputs`` field names.
        ``inputs`` may differ across variants.
    selector:
        * ``str`` — static: the named variant is activated immediately.
        * :class:`~.index.CategoricalIndex` — runtime: per-scenario
          probabilistic selection; the index is sampled by
          :class:`~simulation.ensemble.DistributionEnsemble`.
        * ``graph.Node`` — runtime: per-scenario deterministic selection
          derived from other model parameters; must evaluate to a string
          matching a variant key.  Use :meth:`guards_to_selector` as
          a convenience builder.

    Raises
    ------
    ValueError
        If *variants* is empty.
    ValueError
        If *selector* is a ``str`` that does not match any variant key.
    ValueError
        If *selector* is a :class:`~.index.CategoricalIndex` and any
        outcome key is not present in *variants*.
    ValueError
        If the ``outputs`` field names differ across variants.

    Notes
    -----
    **Static mode — transparency**

    Proxies the following attributes of the active variant:
    ``inputs``, ``outputs``, ``expose``, ``indexes``,
    ``abstract_indexes()``, ``is_instantiated()``, and any direct
    attribute lookup forwarded via ``__getattr__``.

    **Runtime mode — merged graph**

    ``mv.outputs.field`` returns a real :class:`~.index.Index` backed by an
    :class:`~engine.frontend.graph.exclusive_multi_clause_where` node.
    Parent model formulas can wire these outputs directly.

    ``mv.abstract_indexes()`` returns the union of all variants' abstract
    indexes plus the :class:`~.index.CategoricalIndex` selector (if
    applicable).

    ``mv.expose`` returns only fields whose names appear in **all** variants'
    expose proxies (intersection by name).

    ``mv._selector_index`` is a thin :class:`~.index.Index` wrapping
    ``_selector_node``; ``result[mv._selector_index]`` from an
    :class:`~simulation.evaluation.EvaluationResult` returns a ``(S, 1)``
    string array of the active variant key per scenario.

    **Architectural note**

    The static / runtime distinction is a deliberate v0.8.x simplification.
    Post-0.8.x, a constant-folding engine could represent static mode as a
    ``constant(key)`` selector feeding the same runtime structure, unifying
    both paths.

    Examples
    --------
    Static::

        mv = ModelVariant(
            "Transport",
            variants={"bike": BikeModel(), "train": TrainModel()},
            selector="bike",
        )
        mv.outputs.emissions   # delegates to BikeModel

    Runtime — categorical::

        mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
        mv = ModelVariant("Transport",
                          variants={"bike": BikeModel(), "train": TrainModel()},
                          selector=mode)

    Runtime — guard::

        mv = ModelVariant(
            "Transport",
            variants={"bike": BikeModel(), "train": TrainModel()},
            selector=ModelVariant.guards_to_selector([
                ("train", cost_threshold > 5.0),
                ("bike",  True),
            ]),
        )
    """

    def __init__(
        self,
        name: str,
        variants: Mapping[str, Model],
        selector: str | CategoricalIndex | graph.Node,
    ) -> None:
        if not variants:
            raise ValueError(f"ModelVariant {name!r}: 'variants' must not be empty.")

        if not isinstance(selector, (str, CategoricalIndex, graph.Node)):  # type: ignore[arg-type]
            raise ValueError(
                f"ModelVariant {name!r}: selector must be a str, CategoricalIndex, "
                f"or graph.Node; got {type(selector).__name__!r}."
            )

        _validate_io_contract(name, variants)

        variants_dict = dict(variants)

        # ---------------------------------------------------------------
        # Static mode
        # ---------------------------------------------------------------
        if isinstance(selector, str):
            if selector not in variants_dict:
                raise ValueError(
                    f"ModelVariant {name!r}: selector {selector!r} does not match any "
                    f"declared variant key.  Available keys: {list(variants_dict)}."
                )
            object.__setattr__(self, "name", name)
            object.__setattr__(self, "variants", variants_dict)
            object.__setattr__(self, "_is_static", True)
            object.__setattr__(self, "_active_key", selector)
            object.__setattr__(self, "_active", variants_dict[selector])
            return

        # ---------------------------------------------------------------
        # Runtime mode
        # ---------------------------------------------------------------

        # Construction-time validation for CategoricalIndex selectors.
        if isinstance(selector, CategoricalIndex):
            bad = [k for k in selector.support if k not in variants_dict]
            if bad:
                raise ValueError(
                    f"ModelVariant {name!r}: CategoricalIndex {selector.name!r} has "
                    f"outcome key(s) {bad} not present in variants.  "
                    f"Known keys: {list(variants_dict)}."
                )

        selector_node: graph.Node = selector.node if isinstance(selector, GenericIndex) else selector

        # Build branch_map: key → [output_field_node, ...] in field order.
        output_field_names = _io_field_names(next(iter(variants_dict.values())).outputs)
        branch_map: dict[str, list[graph.Node]] = {
            key: [getattr(model.outputs, field).node for field in output_field_names]
            for key, model in variants_dict.items()
        }

        # Create variant_selector with empty merge_nodes (populated below).
        vs = graph.variant_selector(
            selector_node=selector_node,
            branch_map=branch_map,
            merge_nodes=[],
            name=f"vs:{name}",
        )

        # Build one exclusive_mcw per output field + collect merged Index objects.
        merged_entries: list[tuple[str, GenericIndex]] = []
        merge_nodes: list[graph.Node] = []
        for field in output_field_names:
            clauses = [
                (selector_node == key, getattr(model.outputs, field).node) for key, model in variants_dict.items()
            ]
            mcw = graph.exclusive_multi_clause_where(
                clauses=clauses,
                default_value=graph.constant(float("nan")),
                companion=vs,
                name=f"mcw:{name}:{field}",
            )
            merged_idx = Index(f"merged:{name}:{field}", mcw)
            merged_entries.append((field, merged_idx))
            merge_nodes.append(mcw)

        vs.merge_nodes = merge_nodes  # complete the variant_selector

        merged_outputs = IOProxy(merged_entries)  # type: ignore[arg-type]
        selector_index = Index(f"selector:{name}", selector_node)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "variants", variants_dict)
        object.__setattr__(self, "_is_static", False)
        object.__setattr__(self, "_selector", selector)
        object.__setattr__(self, "_selector_node", selector_node)
        object.__setattr__(self, "_selector_index", selector_index)
        object.__setattr__(self, "_merged_outputs", merged_outputs)
        object.__setattr__(self, "_variant_selector", vs)

    # -------------------------------------------------------------------
    # Static helper
    # -------------------------------------------------------------------

    @staticmethod
    def guards_to_selector(
        guards: list[tuple[str, graph.Node | bool]],
    ) -> graph.Node:
        """Build a string-producing selector node from ``(key, predicate)`` pairs.

        Wraps :func:`~engine.frontend.graph.piecewise`.  Guards are evaluated
        left-to-right; the first matching condition selects its key.  The last
        entry should use ``True`` as its predicate (the unconditional fallback).

        **Guard ordering**: more specific guards must come before more general
        ones.  A general guard placed first will shadow everything after it.

        Parameters
        ----------
        guards:
            Ordered list of ``(variant_key, condition)`` pairs.  The
            condition may be a ``graph.Node`` (boolean-valued) or the literal
            ``True`` for an unconditional fallback.

        Returns
        -------
        graph.Node
            A node that evaluates to the matching variant key string.

        Examples
        --------
        ::

            selector = ModelVariant.guards_to_selector([
                ("metro", (cost > 5.0) & (hour >= 8.0)),  # more specific first
                ("train", cost > 5.0),
                ("bike",  True),                          # fallback
            ])
        """
        return graph.piecewise(*[(graph.constant(key), cond) for key, cond in guards])

    # -------------------------------------------------------------------
    # Core Model attributes — static and runtime paths
    # -------------------------------------------------------------------

    @property
    def inputs(self) -> IOProxy[Any]:
        """Inputs proxy.

        Static: proxies the active variant.
        Runtime: returns the union of all variants' input fields (deduplicated
        by field name, first-seen wins).  Because all branch graphs are live
        simultaneously, every variant's inputs must be reachable — consistent
        with :meth:`abstract_indexes` returning the union of all variants'
        abstract indexes.
        """
        if object.__getattribute__(self, "_is_static"):
            return object.__getattribute__(self, "_active").inputs
        variants: dict[str, Model] = object.__getattribute__(self, "variants")
        seen_fields: set[str] = set()
        entries: list[tuple[str, Any]] = []
        for v in variants.values():
            for field in _io_field_names(v.inputs):
                if field not in seen_fields:
                    seen_fields.add(field)
                    entries.append((field, getattr(v.inputs, field)))
        return IOProxy(entries)  # type: ignore[arg-type]

    @property
    def outputs(self) -> IOProxy[Any]:
        """Outputs proxy.

        Static: proxies the active variant.
        Runtime: returns the merged outputs proxy (one
        :class:`~engine.frontend.graph.exclusive_multi_clause_where`-backed
        :class:`~.index.Index` per output field).
        """
        if object.__getattribute__(self, "_is_static"):
            return object.__getattribute__(self, "_active").outputs
        return object.__getattribute__(self, "_merged_outputs")

    @property
    def expose(self) -> IOProxy[Any]:
        """Expose proxy.

        Static: proxies the active variant.
        Runtime: returns only fields whose names appear in **all** variants'
        expose proxies (intersection by name).  ``expose`` is not part of the
        I/O contract and must not be used for inter-model wiring.
        """
        if object.__getattribute__(self, "_is_static"):
            return object.__getattribute__(self, "_active").expose
        variants: dict[str, Model] = object.__getattribute__(self, "variants")
        variant_list = list(variants.values())
        common = set(_io_field_names(variant_list[0].expose))
        for v in variant_list[1:]:
            common &= set(_io_field_names(v.expose))
        first_expose = variant_list[0].expose
        entries = [(field, getattr(first_expose, field)) for field in _io_field_names(first_expose) if field in common]
        return IOProxy(entries)

    @property
    def indexes(self) -> list[GenericIndex]:
        """All indexes relevant to this ``ModelVariant``.

        Static: index list of the active variant only.
        Runtime: deduplicated union of all variants' indexes, plus the merged
        output indexes, plus the :class:`~.index.CategoricalIndex` selector
        and selector index (if applicable).
        """
        if object.__getattribute__(self, "_is_static"):
            return object.__getattribute__(self, "_active").indexes
        variants: dict[str, Model] = object.__getattribute__(self, "variants")
        seen: set[int] = set()
        result: list[GenericIndex] = []
        for v in variants.values():
            for idx in v.indexes:
                if id(idx) not in seen:
                    seen.add(id(idx))
                    result.append(idx)
        merged_outputs: IOProxy[Any] = object.__getattribute__(self, "_merged_outputs")
        for idx in merged_outputs:
            if id(idx) not in seen:
                seen.add(id(idx))
                result.append(idx)
        sel = object.__getattribute__(self, "_selector")
        if isinstance(sel, CategoricalIndex) and id(sel) not in seen:
            seen.add(id(sel))
            result.append(sel)
        sel_idx: Index = object.__getattribute__(self, "_selector_index")
        if id(sel_idx) not in seen:
            result.append(sel_idx)
        return result

    def abstract_indexes(self) -> list[GenericIndex]:
        """Abstract indexes that must be assigned before evaluation.

        Static: delegates to the active variant.
        Runtime: deduplicated union of all variants' abstract indexes, plus
        the :class:`~.index.CategoricalIndex` selector if applicable.

        The rule is *empty iff evaluation can proceed without any external
        assignments*: a ``graph.Node`` selector adds no new abstract index
        (its dependencies are already covered by the variants' indexes).
        """
        if object.__getattribute__(self, "_is_static"):
            return object.__getattribute__(self, "_active").abstract_indexes()
        variants: dict[str, Model] = object.__getattribute__(self, "variants")
        seen: set[int] = set()
        result: list[GenericIndex] = []
        for v in variants.values():
            for idx in v.abstract_indexes():
                if id(idx) not in seen:
                    seen.add(id(idx))
                    result.append(idx)
        sel = object.__getattribute__(self, "_selector")
        if isinstance(sel, CategoricalIndex) and id(sel) not in seen:
            result.append(sel)
        return result

    def is_instantiated(self) -> bool:
        """Return ``True`` iff there are no abstract indexes.

        Static: delegates to the active variant.
        Runtime: always ``False`` (runtime variants always have abstract
        indexes — at minimum the selector or the variants' own parameters).
        """
        if object.__getattribute__(self, "_is_static"):
            return object.__getattribute__(self, "_active").is_instantiated()
        return False

    # -------------------------------------------------------------------
    # Fall-through attribute access
    # -------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute lookups.

        Static: forwards to the active variant.
        Runtime: forwards to the first variant.
        """
        if object.__getattribute__(self, "_is_static"):
            active: Model = object.__getattribute__(self, "_active")
            return getattr(active, name)
        first: Model = next(iter(object.__getattribute__(self, "variants").values()))
        return getattr(first, name)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        name: str = object.__getattribute__(self, "name")
        variants: dict[str, Model] = object.__getattribute__(self, "variants")
        if object.__getattribute__(self, "_is_static"):
            active_key: str = object.__getattribute__(self, "_active_key")
            return f"ModelVariant({name!r}, active={active_key!r}, keys={list(variants)})"
        sel = object.__getattribute__(self, "_selector")
        return f"ModelVariant({name!r}, selector={sel!r}, keys={list(variants)})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _io_field_names(proxy: IOProxy[Any]) -> list[str]:
    """Return the ordered list of field names registered in *proxy*.

    Parameters
    ----------
    proxy:
        An :class:`~.model.IOProxy` instance.

    Returns
    -------
    list[str]
        Field names in declaration order.
    """
    entries: list[tuple[str, Any]] = object.__getattribute__(proxy, "_entries")
    return [key for key, _ in entries]


def _validate_io_contract(variant_group_name: str, variants: Mapping[str, Model]) -> None:
    """Ensure all variants share identical ``outputs`` field names.

    Inputs may differ across variants — in runtime mode all variant graphs
    are live simultaneously, so the union of their inputs is exposed via
    :attr:`ModelVariant.inputs`.

    Parameters
    ----------
    variant_group_name:
        Name of the :class:`ModelVariant` group — used in error messages.
    variants:
        Mapping of key → :class:`Model` instance.

    Raises
    ------
    ValueError
        If any variant's ``outputs`` field names differ from the first
        variant's.
    """
    items = list(variants.items())
    ref_key, ref_model = items[0]
    ref_output_names = _io_field_names(ref_model.outputs)

    for key, model in items[1:]:
        output_names = _io_field_names(model.outputs)
        if output_names != ref_output_names:
            raise ValueError(
                f"ModelVariant {variant_group_name!r}: 'outputs' field names differ "
                f"between variants {ref_key!r} and {key!r}.  "
                f"Expected {ref_output_names}, got {output_names}."
            )
