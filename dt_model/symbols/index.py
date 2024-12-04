from __future__ import annotations

from typing import Any

from sympy import lambdify
from scipy import stats

from dt_model.symbols._base import SymbolExtender
from dt_model.symbols.context_variable import ContextVariable


class Index(SymbolExtender):
    """
    Class to represent an index variable.
    """

    def __init__(self, name: str, value: Any, cvs: list[ContextVariable] | None = None, group: str | None = None) -> None:
        super().__init__(name)
        self.group = group
        self.cvs = cvs
        if cvs is not None:
            self.value = lambdify(cvs, value, "numpy")
        else:
            self.value = value

class UniformDistIndex(Index):
    """
    Class to represent an index as a uniform distribution
    """

    def __init__(self, name: str, loc: float, scale: float, group: str | None = None) -> None:
        super().__init__(name, stats.uniform(loc=loc, scale=scale), group=group)
        self.loc = loc
        self.scale = scale

    def __str__(self):
        return f"uniform_dist_idx({self.loc}, {self.scale})"

class LognormDistIndex(Index):
    """
    Class to represent an index as a longnorm distribution
    """

    def __init__(self, name: str, loc: float, scale: float, s: float, group: str | None = None) -> None:
        super().__init__(name, stats.lognorm(loc=loc, scale=scale, s=s), group=group)
        self.loc = loc
        self.scale = scale
        self.s = s

    def __str__(self):
        return f"longnorm_dist_idx({self.loc}, {self.scale}, {self.s})"
        

class TriangDistIndex(Index):
    """
    Class to represent an index as a longnorm distribution
    """

    def __init__(self, name: str, loc: float, scale: float, c: float, group: str | None = None) -> None:
        super().__init__(name, stats.triang(loc=loc, scale=scale, c=c), group=group)
        self.loc = loc
        self.scale = scale
        self.c = c

    def __str__(self):
        return f"triang_dist_idx({self.loc}, {self.scale}, {self.c})"

class ConstIndex(Index):
    """
    Class to represent an index as a longnorm distribution
    """

    def __init__(self, name: str, v: float, group: str | None = None) -> None:
        super().__init__(name, v, group=group)
        self.v = v

    def __str__(self):
        return f"const_idx({self.v})"

class SymIndex(Index):
    """
    Class to represent an index as a symbolic value
    """
    
    def __init__(self, 
                 name: str, 
                 value: Any, 
                 cvs: list[ContextVariable] | None = None
                 , group: str | None = None
                ) -> None:
        super().__init__(name, value, cvs, group=group)
        self.cvs = cvs
        if cvs is not None:
            self.value = lambdify(cvs, value, "numpy")
        else:
            self.value = value

        self.sym_value = value

    def __str__(self):
        return f"sympy_idx({self.value})"

