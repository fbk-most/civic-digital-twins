"""Ensemble generator for the overtourism vertical."""

from __future__ import annotations

import itertools

from civic_digital_twins.dt_model.simulation.evaluation import WeightedScenario
from civic_digital_twins.dt_model.model.index import Distribution, Index
from .context_variable import ContextVariable
from .model import OvertourismModel


class OvertourismEnsemble:
    """Iterable that yields weighted scenarios for an :class:`OvertourismModel`.

    Each iteration produces a :data:`~dt_model.simulation.evaluation.WeightedScenario`
    assigning a concrete value to every context variable and every
    distribution-backed (non-PV, non-CV) abstract index in the model.
    Presence variables are *not* included — they are provided as grid axes
    to :meth:`~dt_model.simulation.evaluation.Evaluation.evaluate`.

    Parameters
    ----------
    model:
        The overtourism model whose CVs are to be sampled.
    scenario:
        Optional override: maps a CV to a list of specific values to use
        instead of sampling from its full support.
    cv_ensemble_size:
        Number of samples per CV when sampling from the full support.
    """

    def __init__(
        self,
        model: OvertourismModel,
        scenario: dict[ContextVariable, list],
        cv_ensemble_size: int = 20,
    ) -> None:
        self.model = model

        # Build per-CV list of (prob, value) pairs.
        self._cv_samples: dict[ContextVariable, list] = {}
        for cv in model.cvs:
            if cv in scenario:
                subset = scenario[cv]
                if len(subset) == 1:
                    self._cv_samples[cv] = [(1.0, subset[0])]
                else:
                    self._cv_samples[cv] = cv.sample(cv_ensemble_size, subset=subset)
            else:
                self._cv_samples[cv] = cv.sample(cv_ensemble_size)

        # Total number of scenarios = product of all per-CV sample sizes.
        self.size = 1
        for samples in self._cv_samples.values():
            self.size *= len(samples)

        # Distribution-backed non-PV non-CV abstract indexes (e.g. capacities
        # and distribution-backed conversion factors).
        # NOTE: use identity comparison (id()) to exclude PVs because
        # GenericIndex.__eq__ returns a graph.Node (always truthy), making
        # the list `in` operator unreliable for these objects.
        pv_ids = {id(pv) for pv in model.pvs}
        self._dist_indexes: list[Index] = [
            idx  # type: ignore[misc]
            for idx in model.abstract_indexes()
            if not isinstance(idx, ContextVariable)
            and id(idx) not in pv_ids
            and isinstance(idx, Index)
            and isinstance(idx.value, Distribution)
        ]

        # Pre-sample S values for each distribution-backed index.
        S = max(self.size, 1)
        self._dist_samples: dict[Index, object] = {
            idx: idx.value.rvs(size=S)  # type: ignore[union-attr]
            for idx in self._dist_indexes
        }

    def __iter__(self):
        """Iterate over all CV combinations, yielding one WeightedScenario each."""
        cvs = list(self._cv_samples.keys())
        cv_sample_lists = [self._cv_samples[cv] for cv in cvs]

        for i, combo in enumerate(itertools.product(*cv_sample_lists)):
            weight = 1.0
            assignments: dict = {}

            for cv, (prob, val) in zip(cvs, combo):
                weight *= prob
                assignments[cv] = val

            # Include pre-sampled values for distribution-backed indexes.
            for idx in self._dist_indexes:
                assignments[idx] = self._dist_samples[idx][i]  # type: ignore[index]

            yield weight, assignments

    def __len__(self) -> int:
        """Return the total number of scenarios."""
        return self.size
