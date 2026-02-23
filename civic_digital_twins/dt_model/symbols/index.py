"""
Classes representing index variables.

An index variable is a variable that is used to represent a conversion factor or a
parameter that is used to calculate the value of a symbol. The index variable can
be a constant, a distribution, or a symbolic expression.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, cast, runtime_checkable

import numpy as np
from scipy import stats

from ..engine.frontend import graph
from .context_variable import ContextVariable


@runtime_checkable
class Distribution(Protocol):
    """Protocol for scipy compatible distributions."""

    def cdf(
        self,
        x: float | np.ndarray,
        *args,
        **kwds,
    ) -> float | np.ndarray:
        """Cumulative distribution function."""
        ...

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        **kwargs,
    ) -> float | np.ndarray:
        """Random variable sampling."""
        ...

    def mean(self, *args, **kwds) -> float | np.ndarray:
        """Random variable mean."""
        ...

    def std(self, *args, **kwds) -> float | np.ndarray:
        """Random variable standard deviation."""
        ...


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
        ``keepdims=True`` is used so that the reduced axis is preserved as
        size 1.  This ensures the result broadcasts correctly against both
        plain timeseries ``(T,)`` and ensemble-batched timeseries
        ``(size, T)``:

        * ``(T,)``      → ``(1,)``      (scalar-like, broadcasts with any T)
        * ``(size, T)`` → ``(size, 1)`` (per-sample scalar in correct shape)
        """
        return graph.reduce_sum(self.node, axis, keepdims=True)

    def mean(self, axis: int = -1) -> graph.Node:
        """Return a graph node that averages this index over the given axis.

        Same axis and keepdims convention as :meth:`sum`.
        """
        return graph.reduce_mean(self.node, axis, keepdims=True)


class Index(GenericIndex):
    """Class to represent an index variable."""

    def __init__(
        self,
        name: str,
        value: graph.Scalar | Distribution | graph.Node | None,
        cvs: list[ContextVariable] | None = None,
    ) -> None:
        self.name = name
        self.cvs = cvs

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


class UniformDistIndex(Index):
    """Class to represent an index as a uniform distribution."""

    def __init__(
        self,
        name: str,
        loc: float,
        scale: float,
    ) -> None:
        super().__init__(
            name,
            cast(
                Distribution,
                stats.uniform(loc=loc, scale=scale),
            ),
        )
        self._loc = loc
        self._scale = scale

    @property
    def loc(self):
        """Location parameter."""
        return self._loc

    @loc.setter
    def loc(self, new_loc):
        """Location parameter setter."""
        if self._loc != new_loc:
            self._loc = new_loc
            self.value = stats.uniform(loc=self._loc, scale=self._scale)

    @property
    def scale(self):
        """Scale parameter."""
        return self._scale

    @scale.setter
    def scale(self, new_scale):
        """Scale parameter setter."""
        if self._scale != new_scale:
            self._scale = new_scale
            self.value = stats.uniform(loc=self._loc, scale=self._scale)

    def __str__(self):
        """Represent the index using a string."""
        return f"uniform_dist_idx({self.loc}, {self.scale})"


class LognormDistIndex(Index):
    """Class to represent an index as a lognorm distribution."""

    def __init__(
        self,
        name: str,
        loc: float,
        scale: float,
        s: float,
    ) -> None:
        super().__init__(
            name,
            cast(
                Distribution,
                stats.lognorm(loc=loc, scale=scale, s=s),
            ),
        )
        self._loc = loc
        self._scale = scale
        self._s = s

    @property
    def loc(self):
        """Location parameter of the lognorm distribution."""
        return self._loc

    @loc.setter
    def loc(self, new_loc):
        """Set the location parameter of the lognorm distribution."""
        if self._loc != new_loc:
            self._loc = new_loc
            self.value = stats.lognorm(loc=self._loc, scale=self._scale, s=self.s)

    @property
    def scale(self):
        """Scale parameter of the lognorm distribution."""
        return self._scale

    @scale.setter
    def scale(self, new_scale):
        """Set the scale parameter of the lognorm distribution."""
        if self._scale != new_scale:
            self._scale = new_scale
            self.value = stats.lognorm(loc=self._loc, scale=self._scale, s=self._s)

    @property
    def s(self):
        """Shape parameter of the lognorm distribution."""
        return self._s

    @s.setter
    def s(self, new_s):
        """Set the shape parameter of the lognorm distribution."""
        if self._s != new_s:
            self._s = new_s
            self.value = stats.lognorm(loc=self._loc, scale=self._scale, s=self._s)

    def __str__(self):
        """Represent the index using a string."""
        return f"longnorm_dist_idx({self.loc}, {self.scale}, {self.s})"


class TriangDistIndex(Index):
    """Class to represent an index as a triangular distribution."""

    def __init__(
        self,
        name: str,
        loc: float,
        scale: float,
        c: float,
    ) -> None:
        super().__init__(
            name,
            cast(
                Distribution,
                stats.triang(loc=loc, scale=scale, c=c),
            ),
        )
        self._loc = loc
        self._scale = scale
        self._c = c

    @property
    def loc(self):
        """Location parameter of the triangular distribution."""
        return self._loc

    @loc.setter
    def loc(self, new_loc):
        """Set the location parameter of the triangular distribution."""
        if self._loc != new_loc:
            self._loc = new_loc
            self.value = stats.triang(loc=self._loc, scale=self._scale, c=self._c)

    @property
    def scale(self):
        """Scale parameter of the triangular distribution."""
        return self._scale

    @scale.setter
    def scale(self, new_scale):
        """Set the scale parameter of the triangular distribution."""
        if self._scale != new_scale:
            self._scale = new_scale
            self.value = stats.triang(loc=self._loc, scale=self._scale, c=self._c)

    @property
    def c(self):
        """Shape parameter of the triangular distribution."""
        return self._c

    @c.setter
    def c(self, new_c):
        """Set the shape parameter of the triangular distribution."""
        if self._c != new_c:
            self._c = new_c
            self.value = stats.triang(loc=self._loc, scale=self._scale, c=self._c)

    def __str__(self):
        """Return a string representation of the triangular distribution index."""
        return f"triang_dist_idx({self.loc}, {self.scale}, {self.c})"


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


class SymIndex(Index):
    """Class to represent an index as a symbolic value."""

    def __init__(
        self,
        name: str,
        value: graph.Node,
        cvs: list[ContextVariable] | None = None,
    ) -> None:
        super().__init__(name, value, cvs)
        self.sym_value = value

    def __str__(self):
        """Return a string representation of the symbolic index."""
        return f"sympy_idx({self.value})"


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
      The ``values`` and ``times`` attributes are ``None`` in this mode.

    The optional ``times`` parameter attaches a companion array of time
    points to fixed-array and placeholder modes.
    """

    def __init__(
        self,
        name: str,
        values: np.ndarray | graph.Node | None = None,
        times: np.ndarray | None = None,
        cvs: list[ContextVariable] | None = None,
    ) -> None:
        # We bypass Index.__init__ and set attributes directly because
        # numpy arrays are not a recognised Index value type.
        self.name = name
        self.cvs = cvs
        self._times = np.asarray(times) if times is not None else None
        if isinstance(values, graph.Node):
            # Formula mode — same dispatch as Index for graph nodes.
            values.maybe_set_name(name)
            self._values = None
            self.value = values
            self.node = values
        elif values is not None:
            self._values = np.asarray(values)
            self.value = self._values
            self.node = graph.timeseries_constant(self._values, self._times, name)
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
                self.node = graph.timeseries_constant(self._values, self._times, self.name)

    @property
    def times(self) -> np.ndarray | None:
        """The time axis (optional)."""
        return self._times

    @times.setter
    def times(self, new_times: np.ndarray | None) -> None:
        """Set the time axis and refresh the graph node.

        When no values are set (placeholder mode), only the stored times are
        updated; the node remains a timeseries_placeholder.
        """
        self._times = np.asarray(new_times) if new_times is not None else None
        if self._values is not None:
            self.node = graph.timeseries_constant(self._values, self._times, self.name)

    def __str__(self) -> str:
        """Return a string representation of the timeseries index."""
        if isinstance(self.node, graph.timeseries_placeholder):
            return "timeseries_idx(placeholder)"
        if self._values is not None:
            return f"timeseries_idx({self._values.tolist()!r})"
        return f"timeseries_idx({self.value})"
