"""Code to evaluate a model in specific conditions."""

from ..model.instantiated_model import InstantiatedModel
from ..symbols.index import Index


class Evaluation:
    """Evaluate a model in specific conditions."""

    def __init__(self, inst: InstantiatedModel, ensemble):
        self.inst = inst
        self.ensemble = ensemble

    def evaluate_grid(self, grid):
        """Evaluate the model in the given grid and ensemble conditions."""
        self.inst.legacy.reset()
        self.inst.legacy.evaluate_grid(grid, self.ensemble, self.inst.values)
        return self.inst.legacy.field

    def evaluate_usage(self, presences):
        """Evaluate the model in the given grid and ensemble conditions."""
        return self.inst.legacy.evaluate_usage(presences, self.ensemble, self.inst.values)

    @property
    def index_vals(self):
        """Return the values of the indices."""
        return self.inst.legacy.index_vals

    @property
    def field_elements(self):
        """Return the elements of the field."""
        return self.inst.legacy.field_elements

    def get_index_value(self, i: Index) -> float:
        """Return the value of the index."""
        assert self.inst.legacy is not None
        return self.inst.legacy.get_index_value(i)

    def get_index_mean_value(self, i: Index) -> float:
        """Return the mean value of the index."""
        assert self.inst.legacy is not None
        return self.inst.legacy.get_index_mean_value(i)

    def compute_sustainable_area(self) -> float:
        """Return the sustainable area."""
        assert self.inst.legacy is not None
        return self.inst.legacy.compute_sustainable_area()

    # TODO: change API - order of presence variables
    def compute_sustainability_index(self, presences: list) -> float:
        """Return the sustainability index."""
        assert self.inst.legacy is not None
        return self.inst.legacy.compute_sustainability_index(presences)

    def compute_sustainability_index_per_constraint(self, presences: list) -> dict:
        """Return the sustainability index per constraint."""
        assert self.inst.legacy is not None
        return self.inst.legacy.compute_sustainability_index_per_constraint(presences)

    def compute_modal_line_per_constraint(self) -> dict:
        """Return the modal line per constraint."""
        assert self.inst.legacy is not None
        return self.inst.legacy.compute_modal_line_per_constraint()
