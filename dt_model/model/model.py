from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from sympy import lambdify

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

    def evaluate(self, p_case, c_case):
        c_weight = np.array([c[0] for c in c_case])
        c_values = pd.DataFrame([c[1] for c in c_case])
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
        probability = 1.0
        p_values = [np.expand_dims(p_case[pv], axis=2) for pv in self.pvs]
        c_values = [np.expand_dims(c_subs[index], axis=(0, 1)) for index in self.indexes]
        for constraint in self.constraints:
            usage = lambdify(self.pvs + self.indexes, constraint.usage, "numpy")(*p_values, *c_values)
            capacity = constraint.capacity
            # TODO: model type in declaration
            if isinstance(capacity.value, numbers.Number):
                unscaled_result = usage <= capacity.value
            else:
                unscaled_result = 1.0 - capacity.value.cdf(usage)
            result = np.dot(unscaled_result, c_weight)
            probability *= result
        return probability

    # TODO: to be removed in the future
    def evaluate_single_case(self, p_case, c_case):
        c_subs = {}
        for index in self.indexes:
            if index.cvs is None:
                c_subs[index] = index.value
            else:
                args = [c_case[cv] for cv in index.cvs]
                c_subs[index] = index.value(*args)[()]
        probability = 1
        for constraint in self.constraints:
            usage = lambdify(self.pvs, constraint.usage.subs(c_subs), "numpy")(*[p_case[pv] for pv in self.pvs])
            capacity = constraint.capacity
            # TODO: model type in declaration
            if isinstance(capacity.value, numbers.Number):
                result = usage <= capacity.value
            else:
                result = 1 - capacity.value.cdf(usage)
            probability *= result
        return probability

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
