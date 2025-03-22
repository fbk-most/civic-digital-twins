from dt_model.model.instantiated_model import InstantiatedModel
from dt_model.symbols.index import Index


class Evaluation:
    def __init__(self, inst: InstantiatedModel):
        self.inst = inst

    def evaluate(self, grid, ensemble):
        return self.inst.legacy.evaluate(grid, ensemble)

    @property
    def index_vals(self):
        return self.inst.legacy.index_vals

    @property
    def field_elements(self):
        return self.inst.legacy.field_elements

    def get_index_value(self, i: Index) -> float:
        assert(self.inst.legacy is not None)
        return self.inst.legacy.get_index_value(i)

    def get_index_mean_value(self, i: Index) -> float:
        assert(self.inst.legacy is not None)
        return self.inst.legacy.get_index_mean_value(i)

    def compute_sustainable_area(self) -> float:
        assert(self.inst.legacy is not None)
        return self.inst.legacy.compute_sustainable_area()

    # TODO: change API - order of presence variables
    def compute_sustainability_index(self, presences: list) -> float:
        assert(self.inst.legacy is not None)
        return self.inst.legacy.compute_sustainability_index(presences)

    def compute_sustainability_index_per_constraint(self, presences: list) -> dict:
        assert(self.inst.legacy is not None)
        return self.inst.legacy.compute_sustainability_index_per_constraint(presences)

    def compute_modal_line_per_constraint(self) -> dict:
        assert (self.inst.legacy is not None)
        return self.inst.legacy.compute_modal_line_per_constraint()
