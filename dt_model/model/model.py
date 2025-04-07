from __future__ import annotations

from dt_model.model.abstract_model import AbstractModel
from dt_model.model.instantiated_model import InstantiatedModel
from dt_model.simulation.evaluation import Evaluation
from dt_model.symbols.constraint import Constraint
from dt_model.symbols.context_variable import ContextVariable
from dt_model.symbols.index import Index
from dt_model.symbols.presence_variable import PresenceVariable


class Model:
    def __init__(
        self,
        name: str,
        cvs: list[ContextVariable],
        pvs: list[PresenceVariable],
        indexes: list[Index],
        capacities: list[Index],
        constraints: list[Constraint],
    ) -> None:
        self.abs = AbstractModel(name, cvs, pvs, indexes, capacities, constraints)
        self.evaluation = None

    @property
    def name(self):
        return self.abs.name

    # TODO: Remove, should be immutable
    @name.setter
    def name(self, value):
        self.abs.name = value

    @property
    def cvs(self):
        return self.abs.cvs

    @property
    def pvs(self):
        return self.abs.pvs

    @property
    def indexes(self):
        return self.abs.indexes

    @property
    def capacities(self):
        return self.abs.capacities

    @property
    def constraints(self):
        return self.abs.constraints

    @property
    def index_vals(self):
        assert self.evaluation is not None
        return self.evaluation.index_vals

    @property
    def field_elements(self):
        assert self.evaluation is not None
        return self.evaluation.field_elements

    def reset(self):
        self.evaluation = None

    def evaluate(self, grid, ensemble):
        assert self.evaluation is None
        evaluation = Evaluation(InstantiatedModel(self.abs))
        result = evaluation.evaluate(grid, ensemble)
        self.evaluation = evaluation
        return result

    def get_index_value(self, i: Index) -> float:
        assert self.evaluation is not None
        return self.evaluation.get_index_value(i)

    def get_index_mean_value(self, i: Index) -> float:
        assert self.evaluation is not None
        return self.evaluation.get_index_mean_value(i)

    def compute_sustainable_area(self) -> float:
        assert self.evaluation is not None
        return self.evaluation.compute_sustainable_area()

    # TODO: change API - order of presence variables
    def compute_sustainability_index(self, presences: list) -> float:
        assert self.evaluation is not None
        return self.evaluation.compute_sustainability_index(presences)

    def compute_sustainability_index_per_constraint(self, presences: list) -> dict:
        assert self.evaluation is not None
        return self.evaluation.compute_sustainability_index_per_constraint(presences)

    def compute_modal_line_per_constraint(self) -> dict:
        assert self.evaluation is not None
        return self.evaluation.compute_modal_line_per_constraint()

    def variation(self, new_name, *, change_indexes=None, change_capacities=None):
        # TODO(bassosimone): the .subs functionality is not available anymore
        # however it was meant as a workaround for sympy having a single global
        # namespace, therefore, it should be possible to remove it.

        # TODO: check if changes are valid (ie they change elements present in the model)
        if change_indexes is None:
            new_indexes = self.indexes
            change_indexes = {}
        else:
            new_indexes = []
            for index in self.indexes:
                if index in change_indexes:
                    new_indexes.append(change_indexes[index])
                else:
                    new_indexes.append(index)

        if change_capacities is None:
            new_capacities = self.capacities
            change_capacities = {}
        else:
            new_capacities = []
            for capacity in self.capacities:
                if capacity in change_capacities:
                    new_capacities.append(change_capacities[capacity])
                else:
                    new_capacities.append(capacity)

        new_constraints = []

        for constraint in self.constraints:
            new_constraints.append(
                Constraint(
                    constraint.usage.subs(change_indexes),
                    constraint.capacity.subs(change_capacities),
                    group=constraint.group,
                    name=constraint.name,
                )
            )

        return Model(new_name, self.cvs, self.pvs, new_indexes, new_capacities, new_constraints)
