"""
Classes representing index variables.

An index variable is a variable that is used to represent a conversion factor or a
parameter that is used to calculate the value of a symbol. The index variable can
be a constant, a distribution, or a symbolic expression.
"""
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from ..axes import DOMAIN, TIME_AXIS, Axis, filter_by_role
from ..engine.frontend import graph


class AxesInferenceWarning(UserWarning):
    """Warns that a formula's inferred axes are unlikely to be what was intended.

    Emitted when an undeclared formula-backed index combines operands in a way
    that broadens the result's axes (see :class:`Index`). The trigger is a
    heuristic, not a contract: silence it for a formula that is deliberately an
    outer product either by declaring ``axes=`` on the index — which states the
    intent and is verified — or by filtering this category.
    """


def _combining_operand_axes(node: graph.Node) -> list[tuple[Axis, ...]] | None:
    """Return each operand's own axes for a combining node, or ``None`` if it is not one.

    "Combining" means a node whose ``output_axes`` is the union of several
    independently axis-carrying operands — ``BinaryOp``, ``where``, and
    ``MultiClauseOp``. Those are the only shapes that can produce the emergent
    outer product :func:`_surprising_axes_reason` looks for.
    """
    if isinstance(node, graph.BinaryOp):
        return [node.left.output_axes, node.right.output_axes]
    if isinstance(node, graph.where):
        return [node.condition.output_axes, node.then.output_axes, node.otherwise.output_axes]
    if isinstance(node, graph.MultiClauseOp):
        operand_axes = [ax for cond, val in node.clauses for ax in (cond.output_axes, val.output_axes)]
        operand_axes.append(node.default_value.output_axes)
        return operand_axes
    return None


def _surprising_axes_reason(node: graph.Node) -> str | None:
    """Return why *node*'s inferred axes may be unintended, or ``None`` if they look fine.

    Two independent triggers:

    * **Emergent outer product** — a combining node whose result axes are a
      strict superset of *every* operand's own axes. No operand already
      carried the full result, so the extra axes emerged from combining
      disjoint contributions (``a:(time,) * b:(space,)`` -> ``(time, space)``).
      Broadcasting a scalar, or combining operands that already share the
      result's axes, is not surprising and does not trigger.
    * **Unsigned function_call** — a ``function_call`` with non-empty axes and
      no functor-declared ``output_axes``. Its inference is a conservative
      union, so it over-estimates whenever the function reduces an axis — even
      when only one axis is involved.
    """
    if isinstance(node, graph.function_call):
        if node.output_axes and not node.has_declared_output_axes:
            return "function_call axis inference is a conservative union and may over-estimate"
        return None
    operand_axes = _combining_operand_axes(node)
    if operand_axes is None:
        return None
    result = set(node.output_axes)
    if all(set(operand) != result for operand in operand_axes):
        return "combining operands with disjoint axes produced a result broader than any single operand"
    return None


def _verify_declared_axes(owner: str, declared: tuple[Axis, ...], inferred: tuple[Axis, ...]) -> None:
    """Raise ``ValueError`` unless *declared* and *inferred* hold the same axes.

    Compared as **sets**: the order of a formula's inferred ``output_axes`` is
    an artifact of how ``union_axes`` walked the operands (``a * b`` and
    ``b * a`` order the same result differently), so it carries no meaning to
    assert against. Contrast the injected-value modes, where *axes* is zipped
    positionally against the array's shape and order is load-bearing.
    """
    if set(declared) != set(inferred):
        raise ValueError(
            f"{owner}: declared axes {declared!r} do not match the formula's inferred "
            f"output_axes {inferred!r} (compared as sets, order is not significant). "
            f"Declaring axes verifies the formula, it cannot relabel it — fix whichever "
            f"of the two is wrong."
        )


def _warn_if_axes_surprising(owner: str, node: graph.Node) -> None:
    """Emit :class:`AxesInferenceWarning` if *node*'s inferred axes look unintended."""
    reason = _surprising_axes_reason(node)
    if reason is not None:
        warnings.warn(
            f"{owner}: inferred output_axes={node.output_axes!r} — {reason}. "
            f"Pass axes=... to declare the result explicitly (it is verified against "
            f"the formula), or restructure the formula to avoid combining the axes.",
            AxesInferenceWarning,
            stacklevel=3,
        )


@runtime_checkable
class Distribution(Protocol):
    """Protocol for scipy compatible distributions."""

    def cdf(
        self,
        x: float | np.ndarray,
        *args,
        **kwds,
    ) -> np.ndarray:
        """Cumulative distribution function."""
        ...  # pragma: no cover

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        **kwargs,
    ) -> float | np.ndarray:
        """Random variable sampling."""
        ...  # pragma: no cover

    def mean(self, *args, **kwds) -> float | np.ndarray:
        """Random variable mean."""
        ...  # pragma: no cover

    def std(self, *args, **kwds) -> float | np.ndarray:
        """Random variable standard deviation."""
        ...  # pragma: no cover


DomainValue = float | np.ndarray | Distribution | str | dict[str, float] | list[str]
"""A concrete value assignable to an Index at scenario-evaluation time.

Covers all kinds of domain-level assignment:

- ``float`` — a scalar constant (e.g. a parking cost).
- ``np.ndarray`` — a timeseries or multi-dimensional array.
- :class:`Distribution` — a probability distribution (sampled by the ensemble).
- ``str`` — a categorical outcome pin (for :class:`CategoricalIndex` and
  :class:`ConditionalCategoricalIndex`).
- ``dict[str, float]`` — a probability weight map (for :class:`CategoricalIndex`).
- ``list[str]`` — a restriction list (for :class:`CategoricalIndex` only);
  renormalises the original probabilities over the listed subset at
  :class:`~simulation.scenario.Scenario` construction time.
"""


class GenericIndex(ABC):
    """Abstract base class for all index types.

    Provides arithmetic and comparison operators that delegate to the
    underlying computation graph node, so index objects can participate
    directly in formulas without having to access ``.node`` explicitly.

    Comparison operators return graph nodes rather than booleans (the same
    lazy-evaluation contract as ``graph.Node``).  To keep index objects
    usable as dict keys, ``__hash__`` is intentionally kept identity-based
    — exactly the same approach ``graph.Node`` uses.
    """

    @property
    @abstractmethod
    def node(self) -> graph.Node:
        """The underlying computation graph node."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The human-readable name of this index."""

    @property
    def output_axes(self) -> tuple[Axis, ...]:
        """Semantic domain axes carried by this index's output.

        Delegates to the underlying graph node's :attr:`~graph.Node.output_axes`
        inference so that the axis provenance of any formula is automatically
        tracked without user annotation.
        """
        return self.node.output_axes

    def _node_of(self, other: object) -> graph.Node | graph.Scalar:
        """Unwrap *other* to a graph node when it is a ``GenericIndex``."""
        return other.node if isinstance(other, GenericIndex) else other  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Hashing — must be explicit because we override __eq__ below.
    # Identity-based, matching graph.Node behaviour.
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        """Return identity-based hash (required because __eq__ is overridden)."""
        return id(self)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __neg__(self) -> graph.Node:
        """Return a graph node for -self."""
        return graph.negate(self.node)

    def __add__(self, other: object) -> graph.Node:
        """Return a graph node for self + other."""
        return self.node + self._node_of(other)

    def __radd__(self, other: object) -> graph.Node:
        """Return a graph node for other + self."""
        return self._node_of(other) + self.node  # type: ignore[operator]

    def __sub__(self, other: object) -> graph.Node:
        """Return a graph node for self - other."""
        return self.node - self._node_of(other)

    def __rsub__(self, other: object) -> graph.Node:
        """Return a graph node for other - self."""
        return self._node_of(other) - self.node  # type: ignore[operator]

    def __mul__(self, other: object) -> graph.Node:
        """Return a graph node for self * other."""
        return self.node * self._node_of(other)

    def __rmul__(self, other: object) -> graph.Node:
        """Return a graph node for other * self."""
        return self._node_of(other) * self.node  # type: ignore[operator]

    def __truediv__(self, other: object) -> graph.Node:
        """Return a graph node for self / other."""
        return self.node / self._node_of(other)

    def __rtruediv__(self, other: object) -> graph.Node:
        """Return a graph node for other / self."""
        return self._node_of(other) / self.node  # type: ignore[operator]

    def __pow__(self, other: object) -> graph.Node:
        """Return a graph node for self ** other."""
        return self.node ** self._node_of(other)

    def __rpow__(self, other: object) -> graph.Node:
        """Return a graph node for other ** self."""
        return self._node_of(other) ** self.node  # type: ignore[operator]

    # ------------------------------------------------------------------
    # Comparison operators — return graph nodes (lazy evaluation).
    # See __hash__ comment above.
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> graph.Node:  # type: ignore[override]
        """Return a graph node for self == other (lazy evaluation)."""
        return self.node == self._node_of(other)

    def __ne__(self, other: object) -> graph.Node:  # type: ignore[override]
        """Return a graph node for self != other (lazy evaluation)."""
        return self.node != self._node_of(other)

    def __lt__(self, other: object) -> graph.Node:
        """Return a graph node for self < other (lazy evaluation)."""
        return self.node < self._node_of(other)

    def __le__(self, other: object) -> graph.Node:
        """Return a graph node for self <= other (lazy evaluation)."""
        return self.node <= self._node_of(other)

    def __gt__(self, other: object) -> graph.Node:
        """Return a graph node for self > other (lazy evaluation)."""
        return self.node > self._node_of(other)

    def __ge__(self, other: object) -> graph.Node:
        """Return a graph node for self >= other (lazy evaluation)."""
        return self.node >= self._node_of(other)

    # ------------------------------------------------------------------
    # Reduction operators
    # ------------------------------------------------------------------

    def _resolve_reduction_axis(self, axis: Axis | None) -> Axis:
        """Resolve the axis to reduce, defaulting to the index's unique DOMAIN axis.

        When *axis* is given it is used verbatim. When it is ``None``:

        * exactly one DOMAIN axis → that axis (a timeseries index has exactly
          one, time, so its reductions keep defaulting to time as before);
        * any other number → there is no unambiguous default, so an explicit
          ``axis=`` is required and a ``ValueError`` is raised.

        An index carrying *no* DOMAIN axis is included in that second case:
        there is no dimension to reduce, so nothing can be inferred. It used to
        fall back to the time axis, reproducing a "reduce the last dimension"
        convention from when every array was a timeseries. The executor now
        resolves a reduction to the position its axis *names*, so that fallback
        would ask it to reduce a time axis the evaluation does not carry.
        """
        if axis is not None:
            return axis
        domain_axes = filter_by_role(self.node.output_axes, DOMAIN)
        if len(domain_axes) == 1:
            return domain_axes[0]
        if not domain_axes:
            raise ValueError(
                f"{type(self).__name__} reduction needs an explicit axis=: this index carries no "
                f"DOMAIN axis, so there is no dimension to reduce over. Declare the index's shape "
                f"with axes=, or pass the axis to reduce explicitly."
            )
        raise ValueError(
            f"{type(self).__name__} reduction needs an explicit axis=: this index carries "
            f"{len(domain_axes)} DOMAIN axes {[ax.name for ax in domain_axes]}, "
            f"so there is no unique default axis to reduce over."
        )

    def sum(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that sums this index over the given axis.

        The default axis is the index's sole DOMAIN axis (for a timeseries
        index, ``Axis("time", DOMAIN)``); an index carrying several DOMAIN
        axes must pass ``axis=`` explicitly.  The reduced axis is always
        preserved as size 1,
        ensuring the result broadcasts correctly against both plain timeseries
        ``(T,)`` and ensemble-batched timeseries ``(size, T)``:

        * ``(T,)``      → ``(1,)``      (scalar-like, broadcasts with any T)
        * ``(size, T)`` → ``(size, 1)`` (per-sample scalar in correct shape)
        """
        return graph.project_using_sum(self.node, self._resolve_reduction_axis(axis))

    def mean(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that averages this index over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_mean(self.node, self._resolve_reduction_axis(axis))

    def min(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that computes the minimum of this index over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_min(self.node, self._resolve_reduction_axis(axis))

    def max(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that computes the maximum of this index over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_max(self.node, self._resolve_reduction_axis(axis))

    def std(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that computes the standard deviation of this index over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_std(self.node, self._resolve_reduction_axis(axis))

    def var(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that computes the variance of this index over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_var(self.node, self._resolve_reduction_axis(axis))

    def median(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that computes the median of this index over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_median(self.node, self._resolve_reduction_axis(axis))

    def prod(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that computes the product of this index over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_prod(self.node, self._resolve_reduction_axis(axis))

    def any(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that tests if any elements of this index are True over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_any(self.node, self._resolve_reduction_axis(axis))

    def all(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that tests if all elements of this index are True over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_all(self.node, self._resolve_reduction_axis(axis))

    def count_nonzero(self, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that counts non-zero elements of this index over the given axis.

        The default axis is the index's sole DOMAIN axis (time, for a
        timeseries index); an index carrying several DOMAIN axes must pass
        ``axis=`` explicitly. The reduced axis is always preserved as size 1.
        """
        return graph.project_using_count_nonzero(self.node, self._resolve_reduction_axis(axis))

    def quantile(self, q: float, axis: Axis | None = None) -> graph.Node:
        """Return a graph node that computes the quantile of this index over the given axis.

        Args:
            q: Quantile level in the range [0, 1]. For example, 0.5 for the median,
               0.95 for the 95th percentile.
            axis: Semantic axis along which to compute the quantile. Defaults
                to the index's sole DOMAIN axis (time, for a timeseries
                index); required when the index carries several.
        """
        return graph.project_using_quantile(self.node, self._resolve_reduction_axis(axis), q)


class Index(GenericIndex):
    """Index variable, scalar or domain-carrying.

    An ``Index`` is *any* index. Its shape is **declared** via ``axes=`` when
    the value comes from outside (concrete/placeholder — nothing else can know
    it) and **derived** from the computation graph when the value is computed
    by a formula (the graph already knows it exactly, via
    :attr:`~GenericIndex.output_axes`).

    Immutable after construction: ``name``, ``value``, and ``node`` are
    read-only.  To vary an index value across runs, use
    :class:`~simulation.scenario.Scenario`.

    Value-source modes — orthogonal to the domain signature, so each applies
    with or without ``axes=``:

    * **Concrete** — ``Index("cost", 8.0)`` (scalar; node is a
      ``graph.placeholder``) or ``Index("field", arr, axes=(x, y))`` (node is
      an ``array_placeholder``, per-axis sizes deduced by zipping *axes*
      against the array's shape in declaration order, so the array's rank must
      match the number of axes). The value is a default, injected by Scenario
      at evaluation time.
    * **Placeholder** — ``Index("cost")`` / ``Index("field", axes=(x, y))``:
      no default; value must be supplied via Scenario or ``parameters=``
      before evaluation.
    * **Formula** — ``Index("cost", formula_node)``: computed by the engine;
      no external injection needed. Another :class:`GenericIndex` is also
      accepted here and unwrapped to its underlying ``.node`` (reusing the
      formula), the same unwrapping the arithmetic operators perform.

    In formula mode ``axes=`` is an optional *assertion* about the formula's
    inferred ``output_axes`` — ``None`` declares nothing, ``()`` declares the
    result scalar, ``(x, y)`` declares those axes. It is a **verification, not
    an override**: a mismatch raises ``ValueError``, and there is no mechanism
    to relabel a formula's axes. The comparison is by **set**, since a
    formula's axis order is a traversal artifact rather than something the
    author chose (contrast the injected modes above, where order is zipped
    against the array's shape and does matter).

    Declaring ``axes=`` on a formula is also how you state intent: an
    undeclared formula whose inferred axes look accidental — an outer product
    emerging from operands with disjoint axes, or an unsigned ``function_call``
    whose union may over-estimate — raises :class:`AxesInferenceWarning`.

    For distribution-backed indexes use :class:`DistributionIndex`.
    """

    # Instance state, declared once here rather than in whichever __init__
    # branch happens to run first.  Subclasses that deliberately bypass
    # Index.__init__ (ConstIndex and its specializations, which build a
    # constant node instead of a placeholder) must set all five.
    _name: str
    _axes: tuple[Axis, ...] | None
    _value: "graph.Scalar | np.ndarray | graph.Node | None"
    _node: graph.Node
    _sizes: dict[str, int]

    def __init__(
        self,
        name: str,
        value: "graph.Scalar | np.ndarray | graph.Node | GenericIndex | None" = None,
        *,
        axes: tuple[Axis, ...] | None = None,
    ) -> None:
        self._name = name
        self._axes = axes

        # Unwrap a sibling index to its underlying node so a formula that
        # reuses it (``Index("y", inp.x)``) is treated as formula-backed,
        # matching how the arithmetic operators unwrap indexes via _node_of.
        if isinstance(value, GenericIndex):
            value = value.node

        if isinstance(value, Distribution):
            raise TypeError(
                f"Index {name!r} cannot be initialised with a Distribution. "
                f"Use DistributionIndex for distribution-backed indexes."
            )

        # Formula node: reuse it directly as this index's node. Any declared
        # *axes* is an assertion about its inferred output_axes, not an
        # override — sizes come from the graph, never from the declaration.
        if isinstance(value, graph.Node):
            value.maybe_set_name(name)
            self._value = value
            self._node = value
            self._sizes = {}
            if axes is not None:
                _verify_declared_axes(f"{type(self).__name__} {name!r}", axes, value.output_axes)
            else:
                _warn_if_axes_surprising(f"{type(self).__name__} {name!r}", value)

        # Concrete value: placeholder injected by Scenario at evaluation time.
        elif value is not None:
            if axes:
                arr = np.asarray(value)
                self._value = arr
                self._node = self._make_placeholder_node(name, axes)
                self._sizes = dict(zip((ax.name for ax in axes), arr.shape, strict=True))
            else:
                self._value = value
                self._node = graph.placeholder(name)
                self._sizes = {}

        # Bare placeholder.
        else:
            self._value = None
            self._node = self._make_placeholder_node(name, axes) if axes else graph.placeholder(name)
            self._sizes = {}

    @staticmethod
    def _make_placeholder_node(name: str, axes: tuple[Axis, ...]) -> graph.Node:
        """Build the placeholder node for the concrete/bare-placeholder modes.

        Subclasses fixing a named shape (:class:`TimeseriesIndex`) simply pass
        their *axes*; the node type is the same generic one either way.
        """
        return graph.array_placeholder(name, axes)

    @property
    def name(self) -> str:
        """The human-readable name of the index."""
        return self._name

    @property
    def node(self) -> graph.Node:
        """The underlying computation graph node."""
        return self._node

    @property
    def axes(self) -> tuple[Axis, ...] | None:
        """The domain axes *declared* for this index, in order, or ``None``.

        ``None`` means no declaration was made: for an injected value that
        implies a scalar, and for a formula it means the shape is left to
        graph inference. Read :attr:`~GenericIndex.output_axes` for the axes
        the value actually carries.
        """
        return self._axes

    @property
    def sizes(self) -> dict[str, int]:
        """Per-axis-name sizes deduced from a concrete array; empty otherwise."""
        return dict(self._sizes)

    @property
    def is_abstract(self) -> bool:
        """Whether this index requires an external value before evaluation."""
        return self._value is None

    @property
    def concrete_default(self) -> "graph.Scalar | np.ndarray | None":
        """The index's concrete default, or ``None`` if unset or formula-backed.

        Used by :class:`~simulation.scenario.Scenario` to seed the executor
        state with each index's default before overrides are applied. Most
        model-authoring code should not need this — read values through the
        computation graph (``.node``) instead.
        """
        return None if isinstance(self._value, graph.Node) else self._value

    def __repr__(self) -> str:
        """Return a string representation of the index."""
        if self._axes:
            axes_suffix = f", axes={self._axes!r}"
            if self._value is None:
                return f"idx({self._name!r}, placeholder{axes_suffix})"
            if isinstance(self._value, graph.Node):
                return f"idx({self._name!r}, <formula>{axes_suffix})"
            return f"idx({self._name!r}, {np.asarray(self._value).tolist()!r}{axes_suffix})"
        if self._value is None:
            return f"idx({self._name!r})"
        if isinstance(self._value, graph.Node):
            return f"idx({self._name!r}, <formula>)"
        return f"idx({self._name!r}, {self._value!r})"


class ConstIndex(Index):
    """Index baked into the computation graph as a constant node.

    Whereas a concrete :class:`Index` builds an overwritable placeholder (its
    value is a default a Scenario may replace), a ``ConstIndex`` bakes its
    value into a constant node: it is permanently fixed and cannot be
    overridden per scenario. Choosing between the two is a modeling decision
    about whether the value is a structural constant of the model (e.g. a unit
    conversion factor) or a scenario-varying input.

    Like :class:`Index`, it is scalar by default and domain-carrying when
    ``axes=`` is declared (``ConstIndex("field", arr, axes=(x, y))``, whose
    array rank must match the number of axes). Immutable after construction.
    """

    def __init__(
        self,
        name: str,
        value: "float | np.ndarray",
        *,
        axes: tuple[Axis, ...] | None = None,
    ) -> None:
        # Bypass Index.__init__: it would create a placeholder node first.
        self._name = name
        self._axes = axes
        if axes:
            arr = np.asarray(value)
            self._value = arr
            self._node = self._make_constant_node(arr, axes, name)
            self._sizes = dict(zip((ax.name for ax in axes), arr.shape, strict=True))
        else:
            self._value = cast(graph.Scalar, value)
            self._node = graph.constant(self._value, name)
            self._sizes = {}

    @staticmethod
    def _make_constant_node(value: np.ndarray, axes: tuple[Axis, ...], name: str) -> graph.Node:
        """Build the constant node for the domain-carrying mode.

        Subclasses fixing a named shape (:class:`ConstTimeseriesIndex`) simply
        pass their *axes*; the node type is the same generic one either way.
        """
        return graph.array_constant(value, axes, name)

    def __repr__(self) -> str:
        """Return a string representation of the constant index."""
        if self._axes:
            return f"const_idx({self._name!r}, {np.asarray(self._value).tolist()!r}, axes={self._axes!r})"
        return f"const_idx({self._value!r})"


class TimeseriesIndex(Index):
    """Time-indexed quantity.

    Specialization of :class:`Index` fixing ``axes=(TIME_AXIS,)``: a thin
    backward-compatible convenience over the generic domain-carrying index,
    not a new concept, and the model for any future named-shape convenience.
    Immutable after construction.

    Three modes mirror :class:`Index`:

    * **Fixed array** — ``TimeseriesIndex(name, np.array([...]))``
      Node is an ``array_placeholder`` over ``axes=(TIME_AXIS,)``; the array
      is the default, injected by :class:`~simulation.scenario.Scenario` at
      evaluation time.
    * **Placeholder** — ``TimeseriesIndex(name)``
      Node is an ``array_placeholder`` over ``axes=(TIME_AXIS,)``; value must
      be supplied via Scenario or ``parameters=`` before evaluation.
    * **Formula** — ``TimeseriesIndex(name, formula_node)``
      Node is the formula node directly; value is computed by the engine.
    """

    def __init__(
        self,
        name: str,
        value: np.ndarray | graph.Node | None = None,
    ) -> None:
        super().__init__(name, value, axes=(TIME_AXIS,))

    def __repr__(self) -> str:
        """Return a string representation of the timeseries index."""
        if self._value is None:
            return "timeseries_idx(placeholder)"
        if isinstance(self._value, np.ndarray):
            return f"timeseries_idx({self._value.tolist()!r})"
        return "timeseries_idx(<formula>)"


class ConstTimeseriesIndex(ConstIndex, TimeseriesIndex):
    """ConstIndex baked into the graph as an ``array_constant`` over ``axes=(TIME_AXIS,)``.

    Specialization of :class:`ConstIndex` fixing ``axes=(TIME_AXIS,)``.
    Immutable after construction; the node is permanently fixed.

    It is *also* a :class:`TimeseriesIndex`, deliberately: that is the type
    that means "time-shaped, whatever the value source", and it is what model
    ``Inputs``/``Outputs`` contracts annotate a time-shaped field with. Only
    :class:`ConstIndex` participates in behavioural dispatch (Scenario refuses
    to override a structural constant before any timeseries-specific
    validation is reached), so the second base is a shape declaration, not a
    behavioural one. Construction resolves to :meth:`ConstIndex.__init__`,
    which builds the node directly.

    Parameters
    ----------
    name:
        Human-readable name for this index.
    value:
        Fixed array of time-step values.  Stored via :func:`numpy.asarray`
        and used to create an ``array_constant`` graph node over
        ``axes=(TIME_AXIS,)``.

    Examples
    --------
    >>> import numpy as np
    >>> ts = ConstTimeseriesIndex("demand", np.array([10.0, 20.0, 30.0]))
    >>> ts
    const_timeseries_idx([10.0, 20.0, 30.0])
    """

    def __init__(self, name: str, value: np.ndarray) -> None:
        super().__init__(name, value, axes=(TIME_AXIS,))

    def __repr__(self) -> str:
        """Return a string representation of the constant timeseries index."""
        assert isinstance(self._value, np.ndarray)
        return f"const_timeseries_idx({self._value.tolist()!r})"


class DistributionIndex(Index):
    """Index backed by any scipy-compatible distribution.

    Immutable after construction.  The underlying node is a
    ``graph.placeholder`` (same as an abstract :class:`Index`); the ensemble
    samples the frozen distribution and injects the result at evaluation time.
    To replace the distribution across runs use :class:`~simulation.scenario.Scenario`
    with a :class:`Distribution` override.

    Parameters
    ----------
    name:
        Human-readable name for this index.
    distribution:
        A callable (e.g. ``scipy.stats.uniform``) that, when called with
        ``**params``, returns a frozen ``Distribution``-conformant object.
    params:
        Keyword arguments forwarded verbatim to *distribution*.  scipy
        validates the parameter values at construction time.

    Examples
    --------
    >>> from scipy import stats
    >>> idx = DistributionIndex("parking capacity", stats.uniform, {"loc": 350.0, "scale": 100.0})
    """

    def __init__(
        self,
        name: str,
        distribution: Callable[..., Any],
        params: dict[str, Any],
    ) -> None:
        self._distribution = distribution
        self._params = dict(params)
        self._frozen: Distribution = cast(Distribution, distribution(**params))
        super().__init__(name, None)  # placeholder; frozen dist stored separately

    @property
    def distribution(self) -> Callable[..., Any]:
        """The callable used to create the frozen distribution."""
        return self._distribution

    @property
    def params(self) -> dict[str, Any]:
        """Copy of the parameters used to create the frozen distribution."""
        return dict(self._params)

    @property
    def frozen_distribution(self) -> Distribution:
        """The frozen distribution instance sampled by the ensemble."""
        return self._frozen

    @property
    def is_abstract(self) -> bool:
        """Always ``True``: a distribution-backed index is always sampled by the ensemble."""
        return True

    def sample(self, rng: np.random.Generator | None = None, size: int = 1) -> np.ndarray:
        """Draw ``size`` samples from the frozen distribution.

        Parameters
        ----------
        rng:
            Optional :class:`numpy.random.Generator` for reproducibility.
            When ``None``, the global NumPy random state is used.
        size:
            Number of samples to draw. Defaults to 1.

        Returns
        -------
        np.ndarray
            Array of shape ``(size,)`` containing the samples.
        """
        return np.asarray(self._frozen.rvs(size=size, random_state=rng))

    def __repr__(self) -> str:
        """Return a string representation of the distribution index."""
        dist_name = getattr(self._distribution, "__name__", repr(self._distribution))
        return f"dist_idx({self._name!r}, {dist_name}, {self._params!r})"


class ConditionalDistributionIndex(Index):
    """Distribution-backed index whose distribution depends on resolved parent values.

    Always abstract (placeholder mode): the underlying node is a
    ``graph.placeholder``, like an unconditional abstract :class:`Index`.
    The ensemble resolves parent values first, then calls *factory* to obtain
    the frozen :class:`Distribution` for each joint parent configuration.

    The *factory* is called with keyword arguments keyed by parent names::

        temp_given_weather = ConditionalDistributionIndex(
            "temperature",
            parents=[cv_weather],
            factory=lambda weather: (
                stats.norm(loc=25.0, scale=3.0) if weather == "good"
                else stats.norm(loc=15.0, scale=5.0)
            ),
        )

    Parameters
    ----------
    name:
        Human-readable name.
    parents:
        Ordered list of parent indexes whose resolved values are forwarded to
        *factory* as keyword arguments.  Valid parent types are
        :class:`CategoricalIndex`, :class:`ConditionalCategoricalIndex`,
        :class:`DistributionIndex`, and :class:`ConditionalDistributionIndex`.
    factory:
        Callable ``(**parent_values) -> Distribution``.  Should return a frozen
        scipy-compatible distribution (satisfies :class:`Distribution` protocol).

    Raises
    ------
    TypeError
        If any parent is not of a supported type.
    """

    def __init__(
        self,
        name: str,
        parents: "Sequence[CategoricalIndex | ConditionalCategoricalIndex | DistributionIndex | ConditionalDistributionIndex]",  # noqa: E501
        factory: "Callable[..., Any]",
    ) -> None:
        for p in parents:
            if not isinstance(
                p,
                CategoricalIndex | ConditionalCategoricalIndex | DistributionIndex | ConditionalDistributionIndex,
            ):
                raise TypeError(
                    f"ConditionalDistributionIndex {name!r}: parent {p!r} must be a "
                    "CategoricalIndex, ConditionalCategoricalIndex, DistributionIndex, "
                    "or ConditionalDistributionIndex."
                )
        self._parents: list[
            CategoricalIndex | ConditionalCategoricalIndex | DistributionIndex | ConditionalDistributionIndex
        ] = list(parents)
        self._factory = factory
        super().__init__(name, None)  # placeholder mode — no unconditional distribution

    @property
    def parents(
        self,
    ) -> "list[CategoricalIndex | ConditionalCategoricalIndex | DistributionIndex | ConditionalDistributionIndex]":  # noqa: E501
        """Parent indexes whose values are passed to the factory."""
        return list(self._parents)

    def distribution_for(self, **parent_values: object) -> Distribution:
        """Return the frozen distribution for a given parent configuration.

        Parameters
        ----------
        **parent_values:
            Keyword arguments keyed by parent :attr:`~Index.name`.

        Returns
        -------
        Distribution
            A frozen scipy-compatible distribution.
        """
        return cast(Distribution, self._factory(**parent_values))

    def sample_for(
        self,
        rng: np.random.Generator | None = None,
        size: int = 1,
        **parent_values: str,
    ) -> np.ndarray:
        """Draw ``size`` samples for a given parent configuration.

        Parameters
        ----------
        rng:
            Optional :class:`numpy.random.Generator` for reproducibility.
            When ``None``, the global NumPy random state is used.
        size:
            Number of samples to draw. Defaults to 1.
        **parent_values:
            Keyword arguments keyed by parent :attr:`~Index.name`.

        Returns
        -------
        np.ndarray
            Array of shape ``(size,)`` containing the samples.
        """
        dist = self.distribution_for(**parent_values)
        return np.asarray(dist.rvs(size=size, random_state=rng))

    def __repr__(self) -> str:
        """Return a string representation of the conditional distribution index."""
        return f"ConditionalDistributionIndex({self.name!r}, parents={[p.name for p in self._parents]!r})"


class CategoricalIndex(Index):
    """Index backed by a finite string-keyed probability distribution.

    Always abstract (placeholder mode): the underlying node is a
    ``graph.placeholder``, equivalent to ``Index(name, value=None)``.
    The declared probabilities describe how the index should be sampled;
    they do not affect graph evaluation directly.

    Because :class:`GenericIndex.__eq__` returns a ``graph.equal`` node,
    a ``CategoricalIndex`` can be used as a guard in ``graph.piecewise``::

        mode = CategoricalIndex("mode", {"electric": 0.6, "diesel": 0.4})
        factor = Index("factor", graph.piecewise(
            (0.0,   mode == "electric"),
            (1.2,   True),
        ))

    Parameters
    ----------
    name:
        Human-readable name.
    outcomes:
        Maps each outcome key to its probability.  All values must be
        strictly positive and sum to 1.0 (validated within a small
        tolerance at construction time).

    Raises
    ------
    ValueError
        If *outcomes* is empty, any probability is non-positive, or the
        probabilities do not sum to 1.0.
    """

    def __init__(self, name: str, outcomes: dict[str, float]) -> None:
        if not outcomes:
            raise ValueError(f"CategoricalIndex {name!r}: 'outcomes' must not be empty.")
        non_positive = [k for k, p in outcomes.items() if p <= 0]
        if non_positive:
            raise ValueError(
                f"CategoricalIndex {name!r}: all probabilities must be strictly positive; "
                f"non-positive keys: {non_positive}."
            )
        total = sum(outcomes.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"CategoricalIndex {name!r}: probabilities must sum to 1.0; got {total}.")
        self._outcomes = dict(outcomes)
        super().__init__(name, None)  # placeholder mode

    @property
    def outcomes(self) -> dict[str, float]:
        """Outcome probabilities, in declaration order."""
        return dict(self._outcomes)

    @property
    def support(self) -> list[str]:
        """Ordered list of outcome keys."""
        return list(self._outcomes)

    def sample(self, rng: np.random.Generator | None = None, size: int = 1) -> np.ndarray:
        """Draw ``size`` keys proportionally to outcome probabilities.

        Parameters
        ----------
        rng:
            Optional :class:`numpy.random.Generator` for reproducibility.
            When ``None``, the global NumPy random state is used.
        size:
            Number of samples to draw. Defaults to 1.

        Returns
        -------
        np.ndarray
            Object-dtype array of shape ``(size,)`` containing the sampled keys.
        """
        keys = self.support
        probs = [self._outcomes[k] for k in keys]
        if rng is not None:
            return np.asarray(rng.choice(keys, size=size, p=probs), dtype=object)  # type: ignore[return-value]
        return np.asarray(np.random.choice(keys, size=size, p=probs), dtype=object)  # type: ignore[return-value]

    def __repr__(self) -> str:
        """Return a string representation of the categorical index."""
        return f"CategoricalIndex({self.name!r}, {self._outcomes!r})"


class ConditionalCategoricalIndex(Index):
    """Categorical index whose outcome probabilities depend on resolved parent values.

    Like :class:`CategoricalIndex`, it is always abstract (placeholder mode):
    the underlying node is a ``graph.placeholder``.  The *support* (set of
    possible outcome keys) is fixed at construction time; only the probability
    distribution over that support varies with the parent configuration.

    The *factory* is called by the ensemble once per joint parent configuration
    with keyword arguments keyed by parent names::

        P_weather_given_season = ConditionalCategoricalIndex(
            "weather",
            parents=[cv_season],
            support=["good", "unsettled", "bad"],
            factory=lambda season: {
                "good":      0.6 if season == "summer" else 0.2,
                "unsettled": 0.3 if season == "summer" else 0.3,
                "bad":       0.1 if season == "summer" else 0.5,
            },
        )

    Parameters
    ----------
    name:
        Human-readable name.
    parents:
        Ordered list of :class:`CategoricalIndex` or
        :class:`ConditionalCategoricalIndex` instances whose resolved values
        are forwarded to *factory* as keyword arguments.
    support:
        Fixed, non-empty list of string outcome keys.  The *factory* must
        return a probability dict with exactly these keys.
    factory:
        Callable ``(**parent_values: str) -> dict[str, float]``.  The returned
        dict must have the same keys as *support* with strictly positive values
        summing to 1.0 (validated at each call).

    Raises
    ------
    ValueError
        If *support* is empty, *parents* contains an invalid type, or the
        *factory* return value fails validation.
    """

    def __init__(
        self,
        name: str,
        parents: "Sequence[CategoricalIndex | ConditionalCategoricalIndex]",
        support: list[str],
        factory: "Callable[..., dict[str, float]]",
    ) -> None:
        if not support:
            raise ValueError(f"ConditionalCategoricalIndex {name!r}: 'support' must not be empty.")
        for p in parents:
            if not isinstance(p, CategoricalIndex | ConditionalCategoricalIndex):
                raise TypeError(
                    f"ConditionalCategoricalIndex {name!r}: parent {p!r} must be a "
                    "CategoricalIndex or ConditionalCategoricalIndex."
                )
        self._parents: list[CategoricalIndex | ConditionalCategoricalIndex] = list(parents)
        self._support: list[str] = list(support)
        self._factory = factory
        super().__init__(name, None)  # placeholder mode

    @property
    def parents(self) -> "list[CategoricalIndex | ConditionalCategoricalIndex]":
        """Parent indexes whose values are passed to the factory."""
        return list(self._parents)

    @property
    def support(self) -> list[str]:
        """Fixed ordered list of outcome keys."""
        return list(self._support)

    def outcomes_for(self, **parent_values: str) -> dict[str, float]:
        """Return the outcome probabilities for a given parent configuration.

        Parameters
        ----------
        **parent_values:
            Keyword arguments keyed by parent :attr:`~Index.name`, one per
            parent in :attr:`parents`.

        Returns
        -------
        dict[str, float]
            Outcome probabilities summing to 1.0, with keys matching
            :attr:`support`.

        Raises
        ------
        ValueError
            If the factory returns an invalid distribution.
        """
        outcomes = self._factory(**parent_values)
        keys_ok = set(outcomes) == set(self._support)
        if not keys_ok:
            raise ValueError(
                f"ConditionalCategoricalIndex {self.name!r}: factory returned keys "
                f"{sorted(outcomes)} but support is {self._support}."
            )
        non_positive = [k for k, p in outcomes.items() if p <= 0]
        if non_positive:
            raise ValueError(
                f"ConditionalCategoricalIndex {self.name!r}: factory returned non-positive "
                f"probabilities for keys: {non_positive}."
            )
        total = sum(outcomes.values())
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"ConditionalCategoricalIndex {self.name!r}: factory probabilities sum to {total}, expected 1.0."
            )
        return outcomes

    def sample_for(
        self,
        rng: np.random.Generator | None = None,
        size: int = 1,
        **parent_values: str,
    ) -> np.ndarray:
        """Draw ``size`` outcome keys for a given parent configuration.

        Parameters
        ----------
        rng:
            Optional :class:`numpy.random.Generator` for reproducibility.
            When ``None``, the global NumPy random state is used.
        size:
            Number of samples to draw. Defaults to 1.
        **parent_values:
            Keyword arguments keyed by parent :attr:`~Index.name`.

        Returns
        -------
        np.ndarray
            Object-dtype array of shape ``(size,)`` containing the sampled keys.
        """
        outcomes = self.outcomes_for(**parent_values)
        keys = list(outcomes.keys())
        probs = [outcomes[k] for k in keys]
        if rng is not None:
            return np.asarray(rng.choice(keys, size=size, p=probs), dtype=object)  # type: ignore[return-value]
        return np.asarray(np.random.choice(keys, size=size, p=probs), dtype=object)  # type: ignore[return-value]

    def __repr__(self) -> str:
        """Return a string representation of the conditional categorical index."""
        return (
            f"ConditionalCategoricalIndex({self.name!r}, "
            f"parents={[p.name for p in self._parents]!r}, "
            f"support={self._support!r})"
        )
