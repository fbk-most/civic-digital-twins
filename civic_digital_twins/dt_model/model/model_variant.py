"""ModelVariant — selects among Model subclasses sharing the same I/O contract."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .index import Distribution, DistributionIndex, GenericIndex, Index
from .model import IOProxy, Model

__all__ = ["ModelVariant"]


class ModelVariant:
    """Selects among :class:`Model` subclasses sharing the same constructor signature.

    A ``ModelVariant`` resolves to exactly one variant at construction time
    (static or concrete-``Index`` selector) and then acts as a fully
    transparent proxy for that variant.  Any attribute that is not defined
    directly on ``ModelVariant`` itself is forwarded to the active
    :class:`Model` instance, so a ``ModelVariant`` can be used anywhere a
    plain ``Model`` is expected.

    Parameters
    ----------
    name:
        Human-readable name for the variant group.
    variants:
        Mapping from string key to an already-constructed :class:`Model`
        instance.  All variants must share the same ``inputs`` field names
        and the same ``outputs`` field names.
    selector:
        One of:

        * A **string literal** — selects the variant identified by that key
          at construction time.
        * A **concrete** :class:`~.index.Index` — the index's current value
          is read and converted to ``str`` to select a key.
        * A :class:`~.index.Distribution` — ensemble execution is not yet
          supported; raises :exc:`NotImplementedError`.

    Raises
    ------
    ValueError
        If *selector* is a string that does not match any key in *variants*.
    ValueError
        If *selector* is a concrete :class:`~.index.Index` whose ``str``
        value does not match any key in *variants*.
    ValueError
        If the ``inputs`` field names differ across variants.
    ValueError
        If the ``outputs`` field names differ across variants.
    NotImplementedError
        If *selector* is a :class:`~.index.Distribution` (ensemble execution
        is out of scope; tracked as Issue 3 / Branch 7).

    Notes
    -----
    **Transparency**

    ``ModelVariant`` proxies the following attributes of the active variant:

    * ``inputs``, ``outputs``, ``expose``, ``indexes``
    * ``abstract_indexes()``, ``is_instantiated()``
    * Any direct attribute lookup (e.g. a ``GenericIndex`` assigned to
      ``self.*`` inside the variant's ``__init__``).

    **Accessing inactive variants**

    Internal indexes that belong to *inactive* variants are not reachable
    through ``ModelVariant`` attribute access or ``indexes``.  They are
    accessible only via explicit navigation::

        model_variant.variants["other_key"].some_index

    **Interface contract**

    ``inputs`` and ``outputs`` field names must be identical across all
    variants.  The identity of the index *objects* inside those fields may
    differ (each variant owns its own index instances), but the field names
    — which define the public I/O contract — must match exactly.  A
    descriptive :exc:`ValueError` is raised at construction time if they
    differ.

    Examples
    --------
    Static selector::

        selector_idx = Index("mode", "bike")

        mv = ModelVariant(
            "TransportModel",
            variants={
                "bike":  BikeModel(capacity=100),
                "train": TrainModel(capacity=500),
            },
            selector="bike",
        )
        mv.outputs.emissions   # delegates to BikeModel instance

    ``Index`` selector::

        mode = Index("mode", "train")
        mv = ModelVariant(
            "TransportModel",
            variants={
                "bike":  BikeModel(capacity=100),
                "train": TrainModel(capacity=500),
            },
            selector=mode,
        )
        mv.outputs.emissions   # delegates to TrainModel instance
    """

    def __init__(
        self,
        name: str,
        variants: Mapping[str, Model],
        selector: str | Index | Distribution,
    ) -> None:
        # ---------------------------------------------------------------
        # Guard: Distribution selector is out of scope.
        # ---------------------------------------------------------------
        if isinstance(selector, Distribution):
            raise NotImplementedError(
                "Distribution-based selectors (ensemble execution over variant keys) "
                "are out of scope for this release.  They are tracked as Issue 3 / "
                "Branch 7 (feat/model-variant-ensemble).  Use a string literal or a "
                "concrete Index selector instead."
            )

        if not variants:
            raise ValueError(f"ModelVariant {name!r}: 'variants' must not be empty.")

        # ---------------------------------------------------------------
        # Validate interface contract: inputs and outputs field names must
        # be identical across *all* variants.
        # ---------------------------------------------------------------
        _validate_io_contract(name, variants)

        # ---------------------------------------------------------------
        # Resolve the active key.
        # ---------------------------------------------------------------
        if isinstance(selector, str):
            active_key = selector
        elif isinstance(selector, Index):
            # DistributionIndex subclasses Index; an abstract (not-yet-sampled)
            # DistributionIndex has a Distribution as its .value — that would
            # be ensemble territory, so reject it here with a clear message.
            if isinstance(selector, DistributionIndex) or isinstance(selector.value, Distribution):
                raise NotImplementedError(
                    "Distribution-based selectors (ensemble execution over variant keys) "
                    "are out of scope for this release.  They are tracked as Issue 3 / "
                    "Branch 7 (feat/model-variant-ensemble).  Use a string literal or a "
                    "concrete Index selector instead."
                )
            if selector.value is None:
                raise ValueError(
                    f"ModelVariant {name!r}: selector Index {selector.name!r} has no "
                    "concrete value (value is None).  Provide a concrete Index."
                )
            active_key = str(selector.value)
        else:
            # Should be unreachable given the Distribution guard above, but
            # catch any unexpected type gracefully.
            raise TypeError(
                f"ModelVariant {name!r}: selector must be a str, a concrete Index, "
                f"or a Distribution (the last raises NotImplementedError).  "
                f"Got {type(selector)!r}."
            )

        if active_key not in variants:
            raise ValueError(
                f"ModelVariant {name!r}: selector resolved to key {active_key!r}, "
                f"which is not among the declared variant keys: {list(variants)}."
            )

        # ---------------------------------------------------------------
        # Store public attributes.
        # ---------------------------------------------------------------
        # Use object.__setattr__ so that our __setattr__ override does not
        # interfere during construction.
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "variants", dict(variants))
        object.__setattr__(self, "_active_key", active_key)
        object.__setattr__(self, "_active", variants[active_key])

    # -------------------------------------------------------------------
    # Transparent proxy — core Model attributes
    # -------------------------------------------------------------------

    @property
    def inputs(self) -> IOProxy[Any]:
        """Proxy for the active variant's ``inputs``."""
        return object.__getattribute__(self, "_active").inputs

    @property
    def outputs(self) -> IOProxy[Any]:
        """Proxy for the active variant's ``outputs``."""
        return object.__getattribute__(self, "_active").outputs

    @property
    def expose(self) -> IOProxy[Any]:
        """Proxy for the active variant's ``expose``."""
        return object.__getattribute__(self, "_active").expose

    @property
    def indexes(self) -> list[GenericIndex]:
        """Index list of the active variant only.

        Internal indexes belonging to *inactive* variants are not visible
        here.  Access them via ``model_variant.variants["key"].indexes``.
        """
        return object.__getattribute__(self, "_active").indexes

    def abstract_indexes(self) -> list[GenericIndex]:
        """Delegate to the active variant's :meth:`~Model.abstract_indexes`.

        Returns
        -------
        list[GenericIndex]
            Abstract indexes of the active variant.
        """
        return object.__getattribute__(self, "_active").abstract_indexes()

    def is_instantiated(self) -> bool:
        """Delegate to the active variant's :meth:`~Model.is_instantiated`.

        Returns
        -------
        bool
            ``True`` if the active variant has no abstract indexes.
        """
        return object.__getattribute__(self, "_active").is_instantiated()

    # -------------------------------------------------------------------
    # Fall-through attribute access — forwards everything else to the
    # active variant, enabling transparent use in place of a plain Model.
    # -------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute lookups to the active variant.

        Parameters
        ----------
        name:
            Attribute name to look up.

        Returns
        -------
        Any
            The attribute value from the active variant.

        Raises
        ------
        AttributeError
            If the active variant does not have the requested attribute.
        """
        # __getattr__ is only called when normal lookup has failed, so
        # 'name', 'variants', '_active_key', '_active' are all handled by
        # __getattribute__ before reaching here.
        active: Model = object.__getattribute__(self, "_active")
        return getattr(active, name)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        name: str = object.__getattribute__(self, "name")
        active_key: str = object.__getattribute__(self, "_active_key")
        variants: dict[str, Model] = object.__getattribute__(self, "variants")
        return f"ModelVariant({name!r}, active={active_key!r}, keys={list(variants)})"


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
    """Ensure all variants share identical ``inputs`` and ``outputs`` field names.

    Parameters
    ----------
    variant_group_name:
        Name of the :class:`ModelVariant` group — used in error messages.
    variants:
        Mapping of key → :class:`Model` instance.

    Raises
    ------
    ValueError
        If any variant's ``inputs`` or ``outputs`` field names differ from
        the first variant's.
    """
    items = list(variants.items())
    ref_key, ref_model = items[0]
    ref_input_names = _io_field_names(ref_model.inputs)
    ref_output_names = _io_field_names(ref_model.outputs)

    for key, model in items[1:]:
        input_names = _io_field_names(model.inputs)
        if input_names != ref_input_names:
            raise ValueError(
                f"ModelVariant {variant_group_name!r}: 'inputs' field names differ "
                f"between variants {ref_key!r} and {key!r}.  "
                f"Expected {ref_input_names}, got {input_names}."
            )

        output_names = _io_field_names(model.outputs)
        if output_names != ref_output_names:
            raise ValueError(
                f"ModelVariant {variant_group_name!r}: 'outputs' field names differ "
                f"between variants {ref_key!r} and {key!r}.  "
                f"Expected {ref_output_names}, got {output_names}."
            )
