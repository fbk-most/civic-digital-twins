"""
Classes representing index variables.

An index variable is a variable that is used to represent a conversion factor or a
parameter that is used to calculate the value of a symbol. The index variable can
be a constant, a distribution, or a symbolic expression.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, cast, runtime_checkable

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
    """Class to represent an index variable."""

    def __init__(
        self,
        name: str,
        value: graph.Scalar | Distribution | graph.Node | None,
    ) -> None:
        self.name = name

        # We model a distribution index as a distribution to invoke when
        # scheduling the model and a placeholder to fill with the result
        # of sampling from the index's distribution.
        if isinstance(value, Distribution):
            self.value = value
            self.node = graph.placeholder(name)

        # We model a constant-value index as a constant value and a
        # corresponding constant node. An alternative modeling could
        # be to use a placeholder and fill it when scheduling.
        elif isinstance(value, graph.Scalar):
            self.value = value
            self.node = graph.constant(value, name)

        # Otherwise, it's just a reference to an existing node (which
        # typically is the result of defining a formula).
        #
        # For debuggability, let's assign the name to the node if it
        # has not been already set by previous code.
        elif value is not None:
            value.maybe_set_name(name)
            self.value = value
            self.node = value

        # The last remaining case is when the value is None, in which
        # case we just create a value-less placeholder.
        else:
            self.value = None
            self.node = graph.placeholder(name)

    @property
    def node(self) -> graph.Node:
        """The underlying computation graph node."""
        return self._node

    @node.setter
    def node(self, value: graph.Node) -> None:
        self._node = value


class DistributionIndex(Index):
    """Index backed by any scipy-compatible distribution.

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
    >>> idx.params |= {"loc": 400.0}   # partial update — scipy re-validates
    """

    def __init__(
        self,
        name: str,
        distribution: Callable[..., Any],
        params: dict[str, Any],
    ) -> None:
        self._distribution = distribution
        self._params = dict(params)
        super().__init__(name, cast(Distribution, distribution(**params)))

    @property
    def distribution(self) -> Callable[..., Any]:
        """The callable used to create the frozen distribution."""
        return self._distribution

    @property
    def params(self) -> dict[str, Any]:
        """Copy of the parameters used to create the frozen distribution."""
        return dict(self._params)

    @params.setter
    def params(self, new_params: dict[str, Any]) -> None:
        """Re-freeze the distribution with new params.

        scipy validates the parameter values; invalid params raise at
        assignment time.  Supports full replacement or partial update via
        the dict-merge operator::

            idx.params = {"loc": 200.0, "scale": 100.0}  # full replacement
            idx.params |= {"loc": 200.0}                 # partial update (Python 3.9+)
        """
        self._params = dict(new_params)
        self.value = cast(Distribution, self._distribution(**self._params))


class ConstIndex(Index):
    """Class to represent an index as a constant."""

    def __init__(
        self,
        name: str,
        v: float,
    ) -> None:
        super().__init__(name, v)
        self._v = v

    @property
    def v(self):
        """Value of the constant index."""
        return self._v

    @v.setter
    def v(self, new_v):
        """Set the value of the constant index."""
        if self._v != new_v:
            self._v = new_v
            self.value = new_v
            self.node = graph.constant(new_v, self.name)

    def __str__(self):
        """Return a string representation of the constant index."""
        return f"const_idx({self.v})"


class TimeseriesIndex(Index):
    """Class to represent a time-indexed quantity.

    A TimeseriesIndex holds a deterministic sequence of values indexed by
    time step.  There are three modes, mirroring what ``Index`` does for
    scalar quantities:

    * **Fixed array** — ``TimeseriesIndex(name, np.array([...]))``
      The node is a ``timeseries_constant`` that evaluates to the stored
      array.
    * **Placeholder** — ``TimeseriesIndex(name)``
      The node is a ``timeseries_placeholder`` whose value must be
      supplied via the executor state before evaluation.
    * **Formula** — ``TimeseriesIndex(name, formula_node)``
      The node is an arbitrary computation graph node whose result is
      time-indexed (analogous to passing a ``graph.Node`` to ``Index``).
      The ``values`` attribute is ``None`` in this mode.
    """

    def __init__(
        self,
        name: str,
        values: np.ndarray | graph.Node | None = None,
    ) -> None:
        # We bypass Index.__init__ and set attributes directly because
        # numpy arrays are not a recognised Index value type.
        self.name = name
        if isinstance(values, graph.Node):
            # Formula mode — same dispatch as Index for graph nodes.
            values.maybe_set_name(name)
            self._values = None
            self.value = values
            self.node = values
        elif values is not None:
            self._values = np.asarray(values)
            self.value = self._values
            self.node = graph.timeseries_constant(self._values, name)
        else:
            self._values = None
            self.value = None
            self.node = graph.timeseries_placeholder(name)

    @property
    def values(self) -> np.ndarray | None:
        """The timeseries values, or None when this index is a placeholder."""
        return self._values

    @values.setter
    def values(self, new_values: np.ndarray | None) -> None:
        """Set the timeseries values and refresh the graph node.

        Setting to None converts the index to a timeseries placeholder.
        Setting to an array converts it to a timeseries constant.
        """
        if new_values is None:
            if self._values is not None:
                self._values = None
                self.value = None
                self.node = graph.timeseries_placeholder(self.name)
        else:
            new_values = np.asarray(new_values)
            if self._values is None or not np.array_equal(self._values, new_values):
                self._values = new_values
                self.value = self._values
                self.node = graph.timeseries_constant(self._values, self.name)

    def __str__(self) -> str:
        """Return a string representation of the timeseries index."""
        if isinstance(self.node, graph.timeseries_placeholder):
            return "timeseries_idx(placeholder)"
        if self._values is not None:
            return f"timeseries_idx({self._values.tolist()!r})"
        return f"timeseries_idx({self.value})"


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
            raise ValueError(
                f"CategoricalIndex {name!r}: probabilities must sum to 1.0; got {total}."
            )
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

    def sample(self, rng: np.random.Generator | None = None) -> str:
        """Draw one key proportional to outcome probabilities.

        Parameters
        ----------
        rng:
            Optional :class:`numpy.random.Generator` for reproducibility.
            When ``None``, the global NumPy random state is used.

        Returns
        -------
        str
            One of the keys from :attr:`support`.
        """
        keys = self.support
        probs = [self._outcomes[k] for k in keys]
        if rng is not None:
            return rng.choice(keys, p=probs)  # type: ignore[return-value]
        return np.random.choice(keys, p=probs)  # type: ignore[return-value]

    def __repr__(self) -> str:
        """Return a string representation of the categorical index."""
        return f"CategoricalIndex({self.name!r}, {self._outcomes!r})"
