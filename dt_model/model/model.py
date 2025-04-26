"""High-level model API."""

from __future__ import annotations

from dt_model.model.abstract_model import AbstractModel
from dt_model.model.instantiated_model import InstantiatedModel
from dt_model.simulation.evaluation import Evaluation
from dt_model.symbols.constraint import Constraint
from dt_model.symbols.context_variable import ContextVariable
from dt_model.symbols.index import Index
from dt_model.symbols.presence_variable import PresenceVariable


class Model:
    """High-level model API."""

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
        """Name of the model."""
        return self.abs.name

    # TODO: Remove, should be immutable
    @name.setter
    def name(self, value):
        """Set the name of the model."""
        self.abs.name = value

    @property
    def cvs(self):
        """List of context variables in the model."""
        return self.abs.cvs

    @property
    def pvs(self):
        """List of presence variables in the model."""
        return self.abs.pvs

    @property
    def indexes(self):
        """List of indexes in the model."""
        return self.abs.indexes

    @property
    def capacities(self):
        """List of capacities in the model."""
        return self.abs.capacities

    @property
    def constraints(self):
        """List of constraints in the model."""
        return self.abs.constraints

    @property
    def index_vals(self):
        """List of index values in the model."""
        assert self.evaluation is not None
        return self.evaluation.index_vals

    @property
    def field_elements(self):
        """List of field elements in the model."""
        assert self.evaluation is not None
        return self.evaluation.field_elements

    def reset(self):
        """Reset the model state."""
        self.evaluation = None

    def evaluate(self, grid, ensemble):
        """Evaluate the model on the given grid and ensemble."""
        assert self.evaluation is None
        evaluation = Evaluation(InstantiatedModel(self.abs), grid, ensemble)
        result = evaluation.evaluate()
        self.evaluation = evaluation
        return result

    def get_index_value(self, i: Index) -> float:
        """Get the value of the model at the given index."""
        assert self.evaluation is not None
        return self.evaluation.get_index_value(i)

    def get_index_mean_value(self, i: Index) -> float:
        """Get the mean value of the model at the given index."""
        assert self.evaluation is not None
        return self.evaluation.get_index_mean_value(i)

    def compute_sustainable_area(self) -> float:
        """Compute the sustainable area of the model."""
        assert self.evaluation is not None
        return self.evaluation.compute_sustainable_area()

    # TODO: change API - order of presence variables
    def compute_sustainability_index(self, presences: list) -> float:
        """Compute the sustainability index of the model given the presence variables."""
        assert self.evaluation is not None
        return self.evaluation.compute_sustainability_index(presences)

    def compute_sustainability_index_per_constraint(self, presences: list) -> dict:
        """Compute the sustainability index per constraint of the model given the presence variables."""
        assert self.evaluation is not None
        return self.evaluation.compute_sustainability_index_per_constraint(presences)

    def compute_modal_line_per_constraint(self) -> dict:
        """Compute the modal line per constraint of the model."""
        assert self.evaluation is not None
        return self.evaluation.compute_modal_line_per_constraint()
