from __future__ import annotations

import random

from dt_model.symbols._base import SymbolExtender


class ContextVariable(SymbolExtender):
    """
    Class to represent a context variable.
    """

    def __init__(self, name: str, values: list, distribution: dict | None = None) -> None:
        super().__init__(name)
        self.values = values
        self.distribution = distribution

    def sample(self, nr: int = 1, subset: list | None = None) -> list:
        """
        Returns a list of values sampled from the context variable or provided
        subset.
        If a distribution is provided in the constructor, the values will be
        sampled according to that distribution.

        Parameters
        ----------
        nr: int
            Number of values to sample.
        subset: list
            List of values to sample.

        Returns
        -------
        list
            List of sampled values.
        """
        assert nr > 0

        values = self.values if subset is None else subset

        if self.distribution:
            distribution = [self.distribution[v] for v in values]
            return random.choices(values, weights=distribution, k=nr)

        return random.choices(values, k=nr)
