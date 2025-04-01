"""
This module contains the classes to represent index variables. An index variable is
a variable that is used to represent a conversion factor or a parameter that is used
to calculate the value of a symbol. The index variable can be a constant, a distribution,
or a symbolic expression.
"""

from __future__ import annotations

from typing import Protocol, cast, runtime_checkable

import numpy as np
from scipy import stats
from sympy import Symbol, lambdify

from ..engine.frontend import graph
from ._base import SymbolExtender
from .context_variable import ContextVariable


@runtime_checkable
class Distribution(Protocol):
    """Protocol for scipy compatible distributions."""

    def cdf(
        self,
        x: float | np.ndarray,
        *args,
        **kwds,
    ) -> float | np.ndarray: ...

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        **kwargs,
    ) -> float | np.ndarray: ...


class Index(SymbolExtender):
    """
    Class to represent an index variable.
    """

    def __init__(
        self,
        name: str,
        value: Symbol | Distribution | graph.Scalar,
        cvs: list[ContextVariable] | None = None,
        group: str | None = None,
        ref_name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.group = group
        self.ref_name = ref_name if ref_name is not None else name
        self.cvs = cvs
        if cvs is not None:
            self.value = lambdify(cvs, value, "numpy")
        else:
            self.value = value


class UniformDistIndex(Index):
    """
    Class to represent an index as a uniform distribution
    """

    def __init__(
        self,
        name: str,
        loc: float,
        scale: float,
        group: str | None = None,
        ref_name: str | None = None,
    ) -> None:
        super().__init__(
            name,
            cast(
                Distribution,
                stats.uniform(loc=loc, scale=scale),
            ),
            group=group,
            ref_name=ref_name,
        )
        self._loc = loc
        self._scale = scale

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, new_loc):
        if self._loc != new_loc:
            self._loc = new_loc
            self.value = stats.uniform(loc=self._loc, scale=self._scale)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, new_scale):
        if self._scale != new_scale:
            self._scale = new_scale
            self.value = stats.uniform(loc=self._loc, scale=self._scale)

    def __str__(self):
        return f"uniform_dist_idx({self.loc}, {self.scale})"


class LognormDistIndex(Index):
    """
    Class to represent an index as a lognorm distribution
    """

    def __init__(
        self,
        name: str,
        loc: float,
        scale: float,
        s: float,
        group: str | None = None,
        ref_name: str | None = None,
    ) -> None:
        super().__init__(
            name,
            cast(
                Distribution,
                stats.lognorm(loc=loc, scale=scale, s=s),
            ),
            group=group,
            ref_name=ref_name,
        )
        self._loc = loc
        self._scale = scale
        self._s = s

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, new_loc):
        if self._loc != new_loc:
            self._loc = new_loc
            self.value = stats.lognorm(loc=self._loc, scale=self._scale, s=self.s)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, new_scale):
        if self._scale != new_scale:
            self._scale = new_scale
            self.value = stats.lognorm(loc=self._loc, scale=self._scale, s=self._s)

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, new_s):
        if self._s != new_s:
            self._s = new_s
            self.value = stats.lognorm(loc=self._loc, scale=self._scale, s=self._s)

    def __str__(self):
        return f"longnorm_dist_idx({self.loc}, {self.scale}, {self.s})"


class TriangDistIndex(Index):
    """
    Class to represent an index as a triangular distribution
    """

    def __init__(
        self,
        name: str,
        loc: float,
        scale: float,
        c: float,
        group: str | None = None,
        ref_name: str | None = None,
    ) -> None:
        super().__init__(
            name,
            cast(
                Distribution,
                stats.triang(loc=loc, scale=scale, c=c),
            ),
            group=group,
            ref_name=ref_name,
        )
        self._loc = loc
        self._scale = scale
        self._c = c

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, new_loc):
        if self._loc != new_loc:
            self._loc = new_loc
            self.value = stats.triang(loc=self._loc, scale=self._scale, c=self._c)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, new_scale):
        if self._scale != new_scale:
            self._scale = new_scale
            self.value = stats.triang(loc=self._loc, scale=self._scale, c=self._c)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, new_c):
        if self._c != new_c:
            self._c = new_c
            self.value = stats.triang(loc=self._loc, scale=self._scale, c=self._c)

    def __str__(self):
        return f"triang_dist_idx({self.loc}, {self.scale}, {self.c})"


class ConstIndex(Index):
    """
    Class to represent an index as a constant
    """

    def __init__(
        self,
        name: str,
        v: float,
        group: str | None = None,
        ref_name: str | None = None,
    ) -> None:
        super().__init__(name, v, group=group, ref_name=ref_name)
        self._v = v

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, new_v):
        if self._v != new_v:
            self._v = new_v
            self.value = new_v

    def __str__(self):
        return f"const_idx({self.v})"


class SymIndex(Index):
    """
    Class to represent an index as a symbolic value
    """

    def __init__(
        self,
        name: str,
        value: Symbol,
        cvs: list[ContextVariable] | None = None,
        group: str | None = None,
        ref_name: str | None = None,
    ) -> None:
        super().__init__(name, value, cvs, group=group, ref_name=ref_name)
        self.sym_value = value

    def __str__(self):
        return f"sympy_idx({self.value})"
