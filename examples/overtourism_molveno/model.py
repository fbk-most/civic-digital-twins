"""OvertourismModel — a Model with labeled overtourism-specific subsets."""

from __future__ import annotations

from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.model.index import GenericIndex
from .constraint import Constraint
from .context_variable import ContextVariable
from .presence_variable import PresenceVariable


class OvertourismModel(Model):
    """A :class:`~dt_model.model.model.Model` with overtourism domain structure.

    Extends the core ``Model`` by labelling subsets of indexes as context
    variables, presence variables, and capacity indexes, and by attaching
    named constraints.  All items in *cvs*, *pvs*, *indexes*, *capacities*,
    and the usage/capacity indexes of *constraints* are merged into the flat
    ``Model.indexes`` list automatically.

    Parameters
    ----------
    name:
        Human-readable name for the model.
    cvs:
        Context variables (sampled externally by the ensemble).
    pvs:
        Presence variables (grid axes in evaluation).
    indexes:
        Plain model indexes (conversion factors, usage factors, etc.).
    capacities:
        Capacity indexes for each constraint.
    constraints:
        Named usage/capacity constraint pairs.
    """

    def __init__(
        self,
        name: str,
        cvs: list[ContextVariable],
        pvs: list[PresenceVariable],
        indexes: list[GenericIndex],
        capacities: list[GenericIndex],
        constraints: list[Constraint],
    ) -> None:
        # Collect all indexes into the flat Model list; CVs and PVs are
        # abstract (placeholder) indexes and are included so that
        # abstract_indexes() finds them automatically.
        all_indexes: list[GenericIndex] = list(cvs) + list(pvs) + list(indexes) + list(capacities)
        super().__init__(name, all_indexes)

        self.cvs = cvs
        self.pvs = pvs
        self.domain_indexes = indexes
        self.capacities = capacities
        self.constraints = constraints
