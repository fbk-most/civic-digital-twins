"""Implementation of the ensemble."""

from __future__ import annotations
import json
import hashlib

from functools import reduce

from ..internal.sympyke.symbol import SymbolValue
from ..model.instantiated_model import InstantiatedModel
from ..symbols.context_variable import ContextVariable


class Ensemble:
    """Iterator generating ensemble conditions."""

    def __init__(
        self,
        model: InstantiatedModel,
        scenario: dict[ContextVariable, list[SymbolValue]],
        cv_ensemble_size: int = 20,
    ):
        """Initialize the ensemble."""
        # TODO: what if cvs is empty?
        self.model = model
        self.ensemble = {}
        self.size = 1
        self.scenario = scenario
        for cv in model.abs.cvs:
            if cv in scenario.keys():
                if len(scenario[cv]) == 1:
                    self.ensemble[cv] = [(1, scenario[cv][0])]
                else:
                    self.ensemble[cv] = cv.sample(cv_ensemble_size, subset=scenario[cv])
                    self.size *= cv_ensemble_size
            else:
                self.ensemble[cv] = cv.sample(cv_ensemble_size)
                self.size *= cv_ensemble_size

    def compute_hash(self, additional_params: list[int] | None = None) -> str:
        """Compute a stable, unique hash for this ensemble configuration and model.

        Args:
            additional_params (list[int] | None): Optional list of integers
                to include in the hash computation for reproducibility.
            custom_name (str | None): Optional custom name defined by the user
        """

        def serialize(obj):
            if hasattr(obj, "name"):
                return obj.name
            if hasattr(obj, "__dict__"):
                return str(vars(obj))
            return str(obj)

        model_id = getattr(self.model.abs, "name", str(self.model.abs))
        serialized = {
            "model": model_id,
            "size": self.size,
            "scenario": {
                serialize(k): [serialize(vv) for vv in vs]
                for k, vs in sorted(self.scenario.items(), key=lambda x: str(x[0]))
            },
        }

        if additional_params is not None:
            serialized["additional_params"] = [int(v) for v in additional_params]

        json_str = json.dumps(serialized, sort_keys=True)

        return hashlib.sha1(json_str.encode()).hexdigest()[:10]

    def __iter__(self):
        """Return an iterator over the ensemble."""
        self.pos = {k: 0 for k in self.ensemble.keys()}
        self.pos[list(self.ensemble.keys())[0]] = -1
        return self

    def __next__(self):
        """Return the next ensemble condition."""
        for k in self.ensemble.keys():
            self.pos[k] += 1
            if self.pos[k] < len(self.ensemble[k]):
                cv_values = {
                    k: self.ensemble[k][self.pos[k]][1] for k in self.ensemble.keys()
                }
                cv_probability = reduce(
                    lambda x, y: x * y,
                    [self.ensemble[k][self.pos[k]][0] for k in self.ensemble.keys()],
                )
                return cv_probability, cv_values
            self.pos[k] = 0
        raise StopIteration
