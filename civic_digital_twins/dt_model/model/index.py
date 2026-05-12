"""
Classes representing index variables.

An index variable is a variable that is used to represent a conversion factor or a
parameter that is used to calculate the value of a symbol. The index variable can
be a constant, a distribution, or a symbolic expression.
"""
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from ..engine.frontend import graph


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


DomainValue = float | np.ndarray | Distribution | str | dict[str, float]
"""A concrete value assignable to an Index at scenario-evaluation time.

Covers all kinds of domain-level assignment:

- ``float`` — a scalar constant (e.g. a parking cost).
- ``np.ndarray`` — a timeseries or multi-dimensional array.
- :class:`Distribution` — a probability distribution (sampled by the ensemble).
- ``str`` — a categorical outcome pin (for :class:`CategoricalIndex` and
  :class:`ConditionalCategoricalIndex`).
- ``dict[str, float]`` — a probability weight map (for :class:`CategoricalIndex`).
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

    def sum(self, axis: int = -1) -> graph.Node:
        """Return a graph node that sums this index over the given axis.

        The default axis ``-1`` sums across the last (time) dimension.
        The reduced axis is always preserved as size 1, ensuring the result
        broadcasts correctly against both plain timeseries ``(T,)`` and
        ensemble-batched timeseries ``(size, T)``:

        * ``(T,)``      → ``(1,)``      (scalar-like, broadcasts with any T)
        * ``(size, T)`` → ``(size, 1)`` (per-sample scalar in correct shape)
        """
        return graph.project_using_sum(self.node, axis)

    def mean(self, axis: int = -1) -> graph.Node:
        """Return a graph node that averages this index over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_mean(self.node, axis)

    def min(self, axis: int = -1) -> graph.Node:
        """Return a graph node that computes the minimum of this index over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_min(self.node, axis)

    def max(self, axis: int = -1) -> graph.Node:
        """Return a graph node that computes the maximum of this index over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_max(self.node, axis)

    def std(self, axis: int = -1) -> graph.Node:
        """Return a graph node that computes the standard deviation of this index over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_std(self.node, axis)

    def var(self, axis: int = -1) -> graph.Node:
        """Return a graph node that computes the variance of this index over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_var(self.node, axis)

    def median(self, axis: int = -1) -> graph.Node:
        """Return a graph node that computes the median of this index over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_median(self.node, axis)

    def prod(self, axis: int = -1) -> graph.Node:
        """Return a graph node that computes the product of this index over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_prod(self.node, axis)

    def any(self, axis: int = -1) -> graph.Node:
        """Return a graph node that tests if any elements of this index are True over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_any(self.node, axis)

    def all(self, axis: int = -1) -> graph.Node:
        """Return a graph node that tests if all elements of this index are True over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_all(self.node, axis)

    def count_nonzero(self, axis: int = -1) -> graph.Node:
        """Return a graph node that counts non-zero elements of this index over the given axis.

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_count_nonzero(self.node, axis)

    def quantile(self, q: float, axis: int = -1) -> graph.Node:
        """Return a graph node that computes the quantile of this index over the given axis.

        Args:
            q: Quantile level in the range [0, 1]. For example, 0.5 for the median,
               0.95 for the 95th percentile.
            axis: Axis along which to compute the quantile (default: -1).

        Same axis convention as :meth:`sum`; the reduced axis is always
        preserved as size 1.
        """
        return graph.project_using_quantile(self.node, axis, q)


class Index(GenericIndex):
    """Scalar-valued index variable.

    Immutable after construction: ``name``, ``value``, and ``node`` are
    read-only.  To vary an index value across runs, use
    :class:`~simulation.scenario.Scenario`.

    Three modes:

    * **Concrete** — ``Index("cost", 8.0)``: scalar default; node is a
      ``graph.placeholder`` filled by Scenario at evaluation time.
    * **Placeholder** — ``Index("cost", None)``: no default; value must be
      supplied via Scenario or ``parameters=`` before evaluation.
    * **Formula** — ``Index("cost", formula_node)``: computed by the engine;
      no external injection needed.

    For distribution-backed indexes use :class:`DistributionIndex`.
    """

    def __init__(
        self,
        name: str,
        value: graph.Scalar | graph.Node | None,
    ) -> None:
        self._name = name

        if isinstance(value, Distribution):
            raise TypeError(
                f"Index {name!r} cannot be initialised with a Distribution. "
                f"Use DistributionIndex for distribution-backed indexes."
            )

        # Formula node: reuse it directly as this index's node.
        if isinstance(value, graph.Node):
            value.maybe_set_name(name)
            self._value: graph.Scalar | graph.Node | None = value
            self._node: graph.Node = value

        # Concrete scalar: placeholder injected by Scenario at evaluation time.
        elif value is not None:
            self._value = value
            self._node = graph.placeholder(name)

        # Bare placeholder.
        else:
            self._value = None
            self._node = graph.placeholder(name)

    @property
    def name(self) -> str:
        """The human-readable name of the index."""
        return self._name

    @property
    def value(self) -> graph.Scalar | graph.Node | None:
        """The default scalar / formula node, or ``None`` for a bare placeholder."""
        return self._value

    @property
    def node(self) -> graph.Node:
        """The underlying computation graph node."""
        return self._node

    def __repr__(self) -> str:
        """Return a string representation of the index."""
        if self._value is None:
            return f"idx({self._name!r})"
        if isinstance(self._value, graph.Node):
            return f"idx({self._name!r}, <formula>)"
        return f"idx({self._name!r}, {self._value!r})"


class ConstIndex(Index):
    """Index baked into the computation graph as a ``graph.constant``.

    Immutable after construction; the node is permanently fixed.  Use this
    when the value is a structural constant (e.g. a unit conversion factor),
    not a scenario parameter.
    """

    def __init__(
        self,
        name: str,
        value: float,
    ) -> None:
        # Bypass Index.__init__: it would create a placeholder node first.
        self._name = name
        self._value: graph.Scalar | graph.Node | None = value
        self._node: graph.Node = graph.constant(value, name)

    def __repr__(self) -> str:
        """Return a string representation of the constant index."""
        return f"const_idx({self._value!r})"


class TimeseriesIndex(GenericIndex):
    """Time-indexed quantity.

    Sibling of :class:`Index` (both extend :class:`GenericIndex` directly);
    not a subclass — ``Index`` is scalar-valued, ``TimeseriesIndex`` carries
    a DOMAIN (time) axis.  Immutable after construction.

    Three modes mirror :class:`Index`:

    * **Fixed array** — ``TimeseriesIndex(name, np.array([...]))``
      Node is a ``timeseries_placeholder``; the array is the default,
      injected by :class:`~simulation.scenario.Scenario` at evaluation time.
    * **Placeholder** — ``TimeseriesIndex(name)``
      Node is a ``timeseries_placeholder``; value must be supplied via
      Scenario or ``parameters=`` before evaluation.
    * **Formula** — ``TimeseriesIndex(name, formula_node)``
      Node is the formula node directly; value is computed by the engine.
    """

    def __init__(
        self,
        name: str,
        value: np.ndarray | graph.Node | None = None,
    ) -> None:
        self._name = name

        # Formula node: reuse it directly as this index's node.
        if isinstance(value, graph.Node):
            value.maybe_set_name(name)
            self._value: np.ndarray | graph.Node | None = value
            self._node: graph.Node = value

        # Concrete array: placeholder injected by Scenario at evaluation time.
        elif value is not None:
            arr = np.asarray(value)
            self._value = arr
            self._node = graph.timeseries_placeholder(name)

        # Bare placeholder.
        else:
            self._value = None
            self._node = graph.timeseries_placeholder(name)

    @property
    def name(self) -> str:
        """The human-readable name of the index."""
        return self._name

    @property
    def value(self) -> np.ndarray | graph.Node | None:
        """The default array / formula node, or ``None`` for a bare placeholder."""
        return self._value

    @property
    def node(self) -> graph.Node:
        """The underlying computation graph node."""
        return self._node

    def __repr__(self) -> str:
        """Return a string representation of the timeseries index."""
        if self._value is None:
            return "timeseries_idx(placeholder)"
        if isinstance(self._value, np.ndarray):
            return f"timeseries_idx({self._value.tolist()!r})"
        return "timeseries_idx(<formula>)"


class ConstTimeseriesIndex(TimeseriesIndex):
    """TimeseriesIndex baked into the graph as a ``timeseries_constant``.

    Timeseries analogue of :class:`ConstIndex`.  Immutable after
    construction; the node is permanently fixed.

    Parameters
    ----------
    name:
        Human-readable name for this index.
    value:
        Fixed array of time-step values.  Stored via :func:`numpy.asarray`
        and used to create a ``timeseries_constant`` graph node.

    Examples
    --------
    >>> import numpy as np
    >>> ts = ConstTimeseriesIndex("demand", np.array([10.0, 20.0, 30.0]))
    >>> ts.value
    array([10., 20., 30.])
    """

    def __init__(self, name: str, value: np.ndarray) -> None:
        # Bypass TimeseriesIndex.__init__: it would create a timeseries_placeholder first.
        self._name = name
        arr = np.asarray(value)
        self._value: np.ndarray | graph.Node | None = arr
        self._node: graph.Node = graph.timeseries_constant(arr, name)

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
    def value(self) -> Distribution:  # type: ignore[override]
        """The frozen distribution instance."""
        return self._frozen

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
