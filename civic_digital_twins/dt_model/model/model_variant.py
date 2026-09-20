"""ModelVariant — selects among Model subclasses sharing the same I/O contract."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..engine.frontend import graph
from ..engine.numpybackend.executor import Functor
from .index import CategoricalIndex, GenericIndex, Index
from .model import IOProxy, Model

__all__ = ["ModelVariant"]


class _MergedOutputsMarker:
    """``dc=`` marker for a synthesized (not dataclass-backed) ``outputs`` :class:`IOProxy`.

    ``mv.outputs``/``mv.expose`` are computed views over multiple variants, not backed by any
    single declared ``Outputs``/``Expose`` dataclass instance — there is nothing real to pass as
    ``IOProxy``'s ``dc=``. Contract validation (:func:`~.contracts._check_index_field_value`)
    recognizes a nested sub-model proxy by ``getattr(type(dc), "_is_outputs"/"_is_expose", False)``,
    the same attribute the real ``@outputs``/``@expose`` decorators set — not by dataclass identity
    — so this bare marker class satisfies that check and lets ``mv.outputs``/``mv.expose`` be
    nested in a parent's own ``@outputs``/``@expose`` field, the same as a plain :class:`Model`'s.
    """

    _is_outputs = True


class _MergedExposeMarker:
    """``dc=`` marker for a synthesized (not dataclass-backed) ``expose`` :class:`IOProxy`.

    See :class:`_MergedOutputsMarker` — same purpose, for ``expose``.
    """

    _is_expose = True


class ModelVariant:
    """Selects among :class:`Model` subclasses sharing the same I/O contract.

    Operates in two modes depending on the *selector* type:

    **Static mode** (``selector: str``) — resolves to exactly one variant at
    construction time and acts as a proxy for it.  Zero overhead: inactive
    variants do not appear in the graph at all.  ``expose`` is the one
    exception to "proxy the active variant" — see the ``expose`` note below.

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
        ``inputs`` and ``expose`` may differ across variants — ``mv.inputs``
        keeps them disjoint (keyed by variant), ``mv.expose`` exposes only
        the fields common to all of them (see Notes).
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
    ``outputs``, ``indexes``,
    ``abstract_indexes()``, ``is_instantiated()``, and any direct
    attribute lookup forwarded via ``__getattr__``.  ``inputs`` and
    ``expose`` are *not* full proxies even here — see below.

    **Runtime mode — merged graph**

    ``mv.outputs.field`` returns a real :class:`~.index.Index` backed by an
    :class:`~engine.frontend.graph.exclusive_multi_clause_where` node.
    Parent model formulas can wire these outputs directly.

    ``mv.abstract_indexes()`` returns the union of all variants' abstract
    indexes plus the :class:`~.index.CategoricalIndex` selector (if
    applicable).

    **``inputs`` — disjoint union, never merged, same rule in both modes**

    ``mv.inputs`` is a plain ``dict[str, IOProxy]`` keyed by variant —
    ``mv.inputs[key] is mv.variants[key].inputs`` — identical in static and
    runtime mode, in both cases covering every declared variant, active or
    not. Unlike ``outputs``/``expose``, input field names are never merged
    across variants by name: the models making up a group are not
    necessarily authored with that group in mind, so a shared field name is
    coincidence, not a shared meaning (e.g. a ``total`` input meaning "total
    passengers" on one variant and "total bikes" on another) — silently
    picking one variant's value, or requiring every variant to agree, would
    both be wrong for names that only accidentally collide. Use
    ``mv.inputs[key].x`` (equivalently ``mv.variants[key].inputs.x``) for a
    specific variant's value.

    **``expose`` — same rule in both modes**

    ``mv.expose`` returns only fields whose names appear in **all** declared
    variants' expose proxies (intersection by name), active or not, in
    *either* static or runtime mode. Unlike ``outputs`` (a hard, load-bearing
    contract validated eagerly at construction — see "Why are only outputs
    field names required to match across variants?" and "Why is expose
    narrowed to the intersection in both modes?" in dd-cdt-modularity.md),
    ``expose`` is diagnostic-only and never wired into further formulas, so
    variants are free to expose different diagnostics — the group as a whole
    only guarantees the common subset. This keeps ``mv.expose``'s accessible
    field set identical regardless of which selector value is chosen, or
    whether the group later moves from a static to a runtime selector — code
    reading ``mv.expose.x`` behaves the same either way. For a *specific*
    variant's full expose shape (including fields outside the intersection),
    use ``mv.variants[key].expose`` directly. Static mode's intersected
    fields still read their values from the *active* variant (the one
    variant that's actually meaningful); runtime mode reads them from an
    arbitrary representative variant, since no single variant is uniquely
    active across scenarios.

    ``mv._selector_index`` is a thin :class:`~.index.Index` wrapping
    ``_selector_node``; ``result[mv._selector_index]`` from an
    :class:`~simulation.evaluation.EvaluationResult` returns a ``(S, 1)``
    string array of the active variant key per scenario.

    **Architectural note**

    The static / runtime distinction is a deliberate simplification: the
    engine has no constant-folding pass, so there is no way to unify them
    today.  If it ever gained one, static mode could in principle become an
    optimised degenerate case of the runtime representation — a
    ``constant(key)`` selector feeding the same merged-graph machinery,
    folded down to the single active branch at build time — but this is
    unrealized future work with no concrete plan.

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
        variants: Mapping[str, Model | ModelVariant],
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

        merged_outputs = IOProxy(merged_entries, dc=_MergedOutputsMarker())  # type: ignore[arg-type]
        selector_index = Index(f"selector:{name}", selector_node)

        # Aggregate _node_functions from all branch models.  Node identity is
        # unique per branch (each model creates its own function_call nodes),
        # so a flat merge is correct — no key collisions can occur.
        merged_node_fns: dict[graph.Node, Functor] = {}
        for v in variants_dict.values():
            merged_node_fns.update(v._node_functions)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "variants", variants_dict)
        object.__setattr__(self, "_is_static", False)
        object.__setattr__(self, "_selector", selector)
        object.__setattr__(self, "_selector_node", selector_node)
        object.__setattr__(self, "_selector_index", selector_index)
        object.__setattr__(self, "_merged_outputs", merged_outputs)
        object.__setattr__(self, "_variant_selector", vs)
        object.__setattr__(self, "_node_functions", merged_node_fns)

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
    def inputs(self) -> dict[str, IOProxy[Any]]:
        """Inputs, keyed by variant — a disjoint union, never merged by field name.

        ``mv.inputs[key] == mv.variants[key].inputs``, for every declared
        variant, active or not, in both static and runtime mode. Unlike
        ``outputs``/``expose``, input field names are never merged across
        variants: the models making up a group are not necessarily authored
        with that group in mind, so the same field name on two variants may
        carry unrelated meanings (e.g. a ``total`` input meaning "total
        passengers" on one variant and "total bikes" on another) — silently
        picking one variant's value for a shared name, or requiring every
        variant to agree, would both be wrong. Access a specific variant's
        inputs explicitly through its key.
        """
        variants: dict[str, Model] = object.__getattribute__(self, "variants")
        return {key: model.inputs for key, model in variants.items()}

    @property
    def outputs(self) -> Any:
        """Outputs proxy.

        Static: proxies the active variant.
        Runtime: returns the merged outputs proxy (one
        :class:`~engine.frontend.graph.exclusive_multi_clause_where`-backed
        :class:`~.index.Index` per output field).

        Declared to return :data:`~typing.Any` so that ``mv.outputs`` can be assigned
        whole into a field annotated with a specific ``Outputs`` dataclass type — the
        "surfacing sub-model diagnostics in bulk" pattern — without a type checker
        rejecting the assignment. The actual runtime value is still an
        :class:`~.model.IOProxy`; :class:`Model` types its own ``outputs`` attribute
        the same way, for the same reason.
        """
        if object.__getattribute__(self, "_is_static"):
            return object.__getattribute__(self, "_active").outputs
        return object.__getattribute__(self, "_merged_outputs")

    @property
    def expose(self) -> Any:
        """Expose proxy.

        Declared to return :data:`~typing.Any`, same reasoning as ``outputs`` above:
        lets ``mv.expose`` be assigned whole into a field annotated with a specific
        ``Expose`` dataclass type without a type checker rejecting the assignment.

        Returns only fields whose names appear in **all** declared variants'
        expose proxies (intersection by name), in both static and runtime
        mode — like ``outputs``' field-set rule (also mode-independent,
        though ``outputs`` requires exact equality rather than an
        intersection), unlike ``inputs``' rule (genuinely mode-dependent:
        active variant only vs. union of all variants). Static mode reads
        the intersected fields' values from the active variant; runtime mode
        reads them from an arbitrary representative variant, since no single
        variant is uniquely active across scenarios. ``expose`` is not part
        of the I/O contract and must not be used for inter-model wiring. Use
        ``mv.variants[key].expose`` for a specific variant's full shape.
        """
        variants: dict[str, Model] = object.__getattribute__(self, "variants")
        variant_list = list(variants.values())
        common = set(_io_field_names(variant_list[0].expose))
        for v in variant_list[1:]:
            common &= set(_io_field_names(v.expose))
        source = (
            object.__getattribute__(self, "_active").expose
            if object.__getattribute__(self, "_is_static")
            else variant_list[0].expose
        )
        entries = [(field, getattr(source, field)) for field in _io_field_names(source) if field in common]
        return IOProxy(entries, dc=_MergedExposeMarker())

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


def _validate_io_contract(variant_group_name: str, variants: Mapping[str, Model | ModelVariant]) -> None:
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
