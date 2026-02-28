"""
Presence variables for the overtourism vertical.

A presence variable is a placeholder index that represents the number of
entities of a given type present in the modelled system.  It extends
:class:`~dt_model.symbols.index.Index` so that it participates in the
standard abstract-index detection logic of
:class:`~dt_model.model.model.Model`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model.model.index import Index
from .context_variable import ContextVariable


class PresenceVariable(Index):
    """Placeholder index with a presence-distribution sampler.

    Parameters
    ----------
    name:
        Name of the presence variable.
    cvs:
        Context variables that influence the presence distribution.
    distribution:
        Callable that accepts the CV values and returns a dict with
        ``"mean"`` and ``"std"`` keys used to parameterise a truncated
        normal distribution.
    """

    def __init__(
        self,
        name: str,
        cvs: list[ContextVariable],
        distribution: Callable | None = None,
    ) -> None:
        super().__init__(name, None)
        self.cvs = cvs
        self.distribution = distribution

    def sample(self, cvs: dict | None = None, nr: int = 1) -> np.ndarray:
        """Return values sampled from the presence distribution.

        Parameters
        ----------
        cvs:
            Mapping from context variable to its current value.
        nr:
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array of sampled presence values.
        """
        assert nr > 0

        all_cvs = []
        if cvs is not None:
            all_cvs = [cvs[cv] for cv in self.cvs if cv in cvs.keys()]
        assert self.distribution is not None
        distr: dict = self.distribution(*all_cvs)
        return np.asarray(
            stats.truncnorm.rvs(
                -distr["mean"] / distr["std"],
                10,
                loc=distr["mean"],
                scale=distr["std"],
                size=nr,
            ),
        )
