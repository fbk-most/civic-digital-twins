from __future__ import annotations

import numbers

from functools import reduce

import numpy as np
import pandas as pd
from sympy import lambdify
from scipy import interpolate, ndimage, stats

from dt_model.symbols.constraint import Constraint
from dt_model.symbols.context_variable import ContextVariable
from dt_model.symbols.index import Index
from dt_model.symbols.presence_variable import PresenceVariable


class Model:
    def __init__(
        self,
        name,
        cvs: list[ContextVariable],
        pvs: list[PresenceVariable],
        indexes: list[Index],
        capacities: list[Index],
        constraints: list[Constraint],
    ) -> None:
        self.name = name
        self.cvs = cvs
        self.pvs = pvs
        self.indexes = indexes
        self.capacities = capacities
        self.constraints = constraints
        self.grid = None
        self.field = None
        self.field_elements = None

    def reset(self):
        self.grid = None
        self.field = None
        self.field_elements = None

    def evaluate(self, grid, ensemble):
        assert self.grid is None
        c_weight = np.array([c[0] for c in ensemble])
        c_values = pd.DataFrame([c[1] for c in ensemble])
        c_size = c_values.shape[0]
        c_subs = {}
        for index in self.indexes:
            if index.cvs is None:
                if isinstance(index.value, numbers.Number):
                    c_subs[index] = [index.value] * c_size
                else:
                    c_subs[index] = index.value.rvs(size=c_size)
            else:
                args = [c_values[cv].values for cv in index.cvs]
                c_subs[index] = index.value(*args)
        self.grid = grid
        self.field = 1.0
        self.field_elements = {}
        assert len(self.pvs) == 2  # TODO: generalize
        grid_shape = (grid[self.pvs[0]].size, grid[self.pvs[1]].size)
        p_values = [np.expand_dims(grid[pv], axis=(i, 2)) for i,pv in enumerate(self.pvs)]
        c_values = [np.expand_dims(c_subs[index], axis=(0, 1)) for index in self.indexes]
        for constraint in self.constraints:
            usage = lambdify(self.pvs + self.indexes, constraint.usage, "numpy")(*p_values, *c_values)
            capacity = constraint.capacity
            # TODO: model type in declaration
            if isinstance(capacity.value, numbers.Number):
                unscaled_result = usage <= capacity.value
            else:
                unscaled_result = 1.0 - capacity.value.cdf(usage)
            result = np.broadcast_to(np.dot(unscaled_result, c_weight), grid_shape)
            self.field_elements[constraint] = result
            self.field *= result
        return self.field

    def compute_sustainable_area(self) -> float:
        assert self.grid is not None
        return self.field.sum() * reduce(lambda x, y: x*y,
                                         [axis.max() / (axis.size - 1) + 1 for axis in list(self.grid.values())])

    # TODO: change API - order of presence variables
    def compute_sustainability_index(self, presences: list) -> float:
        assert self.grid is not None
        # TODO: fill value
        index = interpolate.interpn(self.grid.values(), self.field, np.array(presences),
                                    bounds_error=False, fill_value=0.0)
        return np.mean(index)

    def compute_sustainability_index_per_constraint(self, presences: list) -> dict:
        assert self.grid is not None
        # TODO: fill value
        indexes = {}
        for c in self.constraints:
            index = interpolate.interpn(self.grid.values(), self.field_elements[c], np.array(presences),
                                        bounds_error=False, fill_value=0.0)
            indexes[c] = np.mean(index)
        return indexes

    def compute_modal_line_per_constraint(self) -> dict:
        assert self.grid is not None
        modal_lines = {}
        for c in self.constraints:
            fe = self.field_elements[c]
            matrix = (fe <= 0.5) & ((ndimage.shift(fe, (0,1)) > 0.5) | (ndimage.shift(fe, (0,-1)) > 0.5) |
                                    (ndimage.shift(fe, (1,0)) > 0.5) | (ndimage.shift(fe, (-1,0)) > 0.5))
            (yi, xi) = np.nonzero(matrix)

            # TODO: decide whether two regressions are really necessary
            horizontal_regr = None
            vertical_regr = None
            try:
                horizontal_regr = stats.linregress(self.grid[self.pvs[0]][xi], self.grid[self.pvs[1]][yi])
            except ValueError:
                pass
            try:
                vertical_regr = stats.linregress(self.grid[self.pvs[1]][yi], self.grid[self.pvs[0]][xi])
            except ValueError:
                pass
            # TODO: find a better way to represent the lines (at the moment, we need to encode the endopoints
            # TODO: even before we implement the previous TODO, avoid hardcoding of line length (10000)
            if (horizontal_regr is None) and (vertical_regr is None):
                pass  # No regression is possible (eg median not intersecting the grid)
            elif (horizontal_regr is None) or (horizontal_regr.rvalue < vertical_regr.rvalue):
                if vertical_regr.slope != 0.00:
                    modal_lines[c] = ((vertical_regr.intercept, 0.0),
                                      (0.0, -vertical_regr.intercept / vertical_regr.slope))
                else:
                    modal_lines[c] = ((vertical_regr.intercept, vertical_regr.intercept),
                                      (0.0, 10000.0))
            else:
                if horizontal_regr.slope != 0.0:
                    modal_lines[c] = ((0.0, -horizontal_regr.intercept / horizontal_regr.slope),
                                      (horizontal_regr.intercept, 0.0))
                else:
                    modal_lines[c] = ((0.0, 10000.0),
                                      (horizontal_regr.intercept, horizontal_regr.intercept))
        return modal_lines

    def variation(self, new_name, *, change_indexes=None, change_capacities=None):
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
            new_constraints.append(Constraint(constraint.usage.subs(change_indexes),
                                              constraint.capacity.subs(change_capacities)))
        return Model(new_name, self.cvs, self.pvs, new_indexes, new_capacities, new_constraints)
