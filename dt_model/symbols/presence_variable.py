"""
This module defines presence variables. A presence variable is a model variable that
represents the presence of a certain entity in the modeled system.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats

from ..engine.frontend import graph
from ..internal.sympyke.symbol import SymbolValue
from .context_variable import ContextVariable


class PresenceVariable:
    """
    Class to represent a presence variable.
    """

    def __init__(
        self,
        name: str,
        cvs: list[ContextVariable],
        distribution: Callable | None = None,
    ) -> None:
        self.name = name
        self.node = graph.placeholder(name)
        self.cvs = cvs
        self.distribution = distribution

    def sample(self, cvs: dict | None = None, nr: int = 1) -> np.ndarray:
        """
        Returns a list of values sampled from the presence variable or provided
        subset.
        If a distribution is provided in the constructor, the values will be
        sampled according to that distribution.

        Parameters
        ----------
        cvs: dict
            Dict of context variables to sample.
        nr: int
            Number of values to sample.

        Returns
        -------
        np.array
            List of sampled values.
        """
        assert nr > 0

        all_cvs = []
        # TODO: check this functionality
        if cvs is not None:
            all_cvs = [cvs[cv] for cv in self.cvs if cv in cvs.keys()]
            # TODO: solve this issue of symbols vs names
            all_cvs = list(map(lambda v: v.name if isinstance(v, SymbolValue) else v, all_cvs))
        assert self.distribution is not None
        distr: dict = self.distribution(*all_cvs)
        return np.asarray(
            stats.truncnorm.rvs(-distr["mean"] / distr["std"], 10, loc=distr["mean"], scale=distr["std"], size=nr),
        )
