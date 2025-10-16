"""Code to evaluate a model in specific conditions."""

from functools import reduce

import numpy as np
from scipy import interpolate, ndimage, stats

from ..engine.frontend import graph, linearize
from ..engine.numpybackend import executor
from ..internal.sympyke import symbol
from ..model.instantiated_model import InstantiatedModel
from ..symbols.context_variable import ContextVariable
from ..symbols.index import Distribution, Index


class Evaluation:
    """Evaluate a model in specific conditions."""

    def __init__(self, inst: InstantiatedModel, ensemble):
        self.inst = inst
        self.ensemble = ensemble
        self.index_vals = None
        self.grid = None
        self.field = None
        self.field_elements = None

    def evaluate_grid(self, grid):
        """Evaluate the model according to the grid."""
        if self.inst.values is None:
            assignments = {}
        else:
            assignments = self.inst.values

        # [pre] extract the weights and the size of the ensemble
        c_weight = np.array([c[0] for c in self.ensemble])
        c_size = c_weight.shape[0]

        # [pre] create empty placeholders
        c_subs: dict[graph.Node, np.ndarray] = {}

        # [pre] add global unique symbols
        for entry in symbol.symbol_table.values():
            c_subs[entry.node] = np.array(entry.name)

        # [pre] add context variables
        collector: dict[ContextVariable, list[float]] = {}
        for _, entry in self.ensemble:
            for cv, value in entry.items():
                collector.setdefault(cv, []).append(value)
        for key, values in collector.items():
            c_subs[key.node] = np.asarray(values)

        # [pre] evaluate the indexes depending on distributions
        #
        # TODO(bassosimone): the size used here is too small
        # TODO(pistore): if index is in self.capacities AND type is Distribution,
        #  there is no need to compute the sample, as the cdf of the distribution is directly
        #  used in the constraint calculation below (unless index_vals is used)
        for index in self.inst.abs.indexes + self.inst.abs.capacities:
            if index.name in assignments:
                value = assignments[index.name]
                if isinstance(value, Distribution):
                    c_subs[index.node] = np.asarray(value.rvs(size=c_size))
                else:
                    c_subs[index.node] = np.full(c_size, value)
            else:
                if isinstance(index.value, Distribution):
                    c_subs[index.node] = np.asarray(index.value.rvs(size=c_size))
                # else: not needed, covered by default placeholder behavior

        # [eval] expand dimensions for all values computed thus far
        for key in c_subs:
            c_subs[key] = np.expand_dims(c_subs[key], axis=(0, 1))

        # [eval] add presence variables and expand dimensions
        assert len(self.inst.abs.pvs) == 2  # TODO: generalize
        for i, pv in enumerate(self.inst.abs.pvs):
            c_subs[pv.node] = np.expand_dims(grid[pv], axis=(i, 2))

        # [eval] collect all the nodes to evaluate
        all_nodes: list[graph.Node] = []
        for constraint in self.inst.abs.constraints:
            all_nodes.append(constraint.usage.node)
            if not isinstance(constraint.capacity.value, Distribution):
                all_nodes.append(constraint.capacity.node)
        for index in self.inst.abs.indexes + self.inst.abs.capacities:
            all_nodes.append(index.node)

        # [eval] actually evaluate all the nodes
        state = executor.State(c_subs)
        for node in linearize.forest(*all_nodes):
            executor.evaluate(state, node)

        # [fix] Ensure that we have the correct shape for operands
        def _fix_shapes(value: np.ndarray) -> np.ndarray:
            if value.ndim == 3 and value.shape[2] == 1:
                return np.broadcast_to(value, value.shape[:2] + (c_size,))
            return value

        # [post] compute the sustainability field
        grid_shape = (grid[self.inst.abs.pvs[0]].size, grid[self.inst.abs.pvs[1]].size)
        field = np.ones(grid_shape)
        field_elements = {}
        for constraint in self.inst.abs.constraints:
            # Get usage
            usage = _fix_shapes(np.asarray(c_subs[constraint.usage.node]))

            # Get capacity
            capacity = constraint.capacity
            if capacity.name in assignments:
                capacity_value = assignments[capacity.name]
            else:
                capacity_value = capacity.value

            if not isinstance(capacity_value, Distribution):
                unscaled_result = usage <= _fix_shapes(np.asarray(c_subs[capacity.node]))
            else:
                unscaled_result = 1.0 - capacity_value.cdf(usage)

            # Apply weights and store the result
            result = np.broadcast_to(np.dot(unscaled_result, c_weight), grid_shape)
            field_elements[constraint] = result
            field *= result

        # [post] store the results
        self.index_vals = c_subs
        self.grid = grid
        self.field = field
        self.field_elements = field_elements
        return self.field

    def evaluate_usage(self, presences):
        """Evaluate the model according to the presence argument."""
        if self.inst.values is None:
            assignments = {}
        else:
            assignments = self.inst.values

        # [pre] extract the weights and the size of the ensemble
        c_weight = np.array([c[0] for c in self.ensemble])
        c_size = c_weight.shape[0]

        # [pre] create empty placeholders
        c_subs: dict[graph.Node, np.ndarray] = {}

        # [pre] add global unique symbols
        for entry in symbol.symbol_table.values():
            c_subs[entry.node] = np.array(entry.name)

        # [pre] add context variables
        collector: dict[ContextVariable, list[float]] = {}
        for _, entry in self.ensemble:
            for cv, value in entry.items():
                collector.setdefault(cv, []).append(value)
        for key, values in collector.items():
            c_subs[key.node] = np.asarray(values)

        # [pre] evaluate the indexes depending on distributions
        #
        # TODO(bassosimone): the size used here is too small
        # TODO(pistore): if index is in self.capacities AND type is Distribution,
        #  there is no need to compute the sample, as the cdf of the distribution is directly
        #  used in the constraint calculation below (unless index_vals is used)
        for index in self.inst.abs.indexes + self.inst.abs.capacities:
            if index.name in assignments:
                value = assignments[index.name]
                if isinstance(value, Distribution):
                    c_subs[index.node] = np.asarray(value.rvs(size=c_size))
                else:
                    c_subs[index.node] = np.full(c_size, value)
            else:
                if isinstance(index.value, Distribution):
                    c_subs[index.node] = np.asarray(index.value.rvs(size=c_size))
                # else: not needed, covered by default placeholder behavior

        # [eval] expand dimensions for all values computed thus far
        for key in c_subs:
            c_subs[key] = np.expand_dims(c_subs[key], axis=0)  # CHANGED

        # [eval] add presence variables and expand dimensions
        assert len(self.inst.abs.pvs) == 2  # TODO: generalize
        for i, pv in enumerate(self.inst.abs.pvs):
            c_subs[pv.node] = np.expand_dims(presences[i], axis=1)  # CHANGED

        # [eval] collect all the nodes to evaluate
        all_nodes: list[graph.Node] = []
        for constraint in self.inst.abs.constraints:
            all_nodes.append(constraint.usage.node)
            if not isinstance(constraint.capacity.value, Distribution):
                all_nodes.append(constraint.capacity.node)
        for index in self.inst.abs.indexes + self.inst.abs.capacities:
            all_nodes.append(index.node)

        # [eval] actually evaluate all the nodes
        state = executor.State(c_subs)
        for node in linearize.forest(*all_nodes):
            executor.evaluate(state, node)

        # CHANGED FROM HERE
        # [post] compute the usage map
        usage_elements = {}
        for constraint in self.inst.abs.constraints:
            # Compute and store constraint usage
            usage = np.asarray(c_subs[constraint.usage.node]).mean(axis=1)
            usage_elements[constraint] = usage

        # [post] return the results
        return usage_elements

    def get_index_value(self, i: Index) -> float:
        """Get the value of the given index."""
        assert self.index_vals is not None
        return self.index_vals[i.node]

    def get_index_mean_value(self, i: Index) -> float:
        """Get the mean value of the given index."""
        assert self.index_vals is not None
        return np.average(self.index_vals[i.node])

    def compute_sustainable_area(self) -> float:
        """Compute the sustainable area."""
        assert self.grid is not None
        assert self.field is not None

        grid = self.grid
        field = self.field

        return field.sum() * reduce(
            lambda x, y: x * y, [axis.max() / (axis.size - 1) + 1 for axis in list(grid.values())]
        )

    # TODO: use evaluate_usage instead of evaluate_grid?
    # TODO: change API - order of presence variables
    def compute_sustainability_index(self, presences: list) -> float:
        """Compute the sustainability index."""
        assert self.grid is not None
        grid = self.grid
        field = self.field
        # TODO: fill value
        index = interpolate.interpn(grid.values(), field, np.array(presences), bounds_error=False, fill_value=0.0)
        return np.mean(index)  # type: ignore

    def compute_sustainability_index_with_ci(self, presences: list, confidence: float = 0.9) -> (float, float):
        """Compute the sustainability index with confidence value."""
        assert self.grid is not None
        grid = self.grid
        field = self.field
        # TODO: fill value
        index = interpolate.interpn(grid.values(), field, np.array(presences), bounds_error=False, fill_value=0.0)
        m, se = np.mean(index), stats.sem(index)
        h = se * stats.t.ppf((1 + confidence) / 2.0, index.size - 1)
        return m, h  # type: ignore

    def compute_sustainability_index_per_constraint(self, presences: list) -> dict:
        """Compute the sustainability index per constraint."""
        assert self.grid is not None
        assert self.field_elements is not None

        grid = self.grid
        field_elements = self.field_elements
        # TODO: fill value
        indexes = {}
        for c in self.inst.abs.constraints:
            index = interpolate.interpn(
                grid.values(), field_elements[c], np.array(presences), bounds_error=False, fill_value=0.0
            )
            indexes[c] = np.mean(index)
        return indexes

    def compute_sustainability_index_with_ci_per_constraint(self, presences: list, confidence: float = 0.9) -> dict:
        """Compute the sustainability index with confidence value for each constraint."""
        assert self.grid is not None
        assert self.field_elements is not None

        grid = self.grid
        field_elements = self.field_elements
        # TODO: fill value
        indexes = {}
        for c in self.inst.abs.constraints:
            index = interpolate.interpn(
                grid.values(), field_elements[c], np.array(presences), bounds_error=False, fill_value=0.0
            )
            m, se = np.mean(index), stats.sem(index)
            h = se * stats.t.ppf((1 + confidence) / 2.0, index.size - 1)
            indexes[c] = (m, h)
        return indexes

    def compute_modal_line_per_constraint(self) -> dict:
        """Compute the modal line per constraint."""
        assert self.grid is not None
        assert self.field_elements is not None

        grid = self.grid
        field_elements = self.field_elements
        modal_lines = {}
        for c in self.inst.abs.constraints:
            fe = field_elements[c]
            matrix = (fe <= 0.5) & (
                (ndimage.shift(fe, (0, 1)) > 0.5)
                | (ndimage.shift(fe, (0, -1)) > 0.5)
                | (ndimage.shift(fe, (1, 0)) > 0.5)
                | (ndimage.shift(fe, (-1, 0)) > 0.5)
            )
            (yi, xi) = np.nonzero(matrix)

            # TODO: decide whether two regressions are really necessary
            horizontal_regr = None
            vertical_regr = None
            try:
                horizontal_regr = stats.linregress(grid[self.inst.abs.pvs[0]][xi], grid[self.inst.abs.pvs[1]][yi])
            except ValueError:
                pass
            try:
                vertical_regr = stats.linregress(grid[self.inst.abs.pvs[1]][yi], grid[self.inst.abs.pvs[0]][xi])
            except ValueError:
                pass

            # TODO(pistore,bassosimone): find a better way to represent the lines (at the
            # moment, we need to encode the endpoints
            # TODO(pistore,bassosimone): even before we implement the previous TODO,
            # avoid hardcoding of the length (10000)
            # TODO(pistore): slopes whould be negative, otherwise th approach may not work

            def _vertical(regr) -> tuple[tuple[float, float], tuple[float, float]]:
                """Logic for computing the points with vertical regression."""
                if regr.slope < 0.00:
                    return ((regr.intercept, 0.0), (0.0, -regr.intercept / regr.slope))
                else:
                    return ((regr.intercept, regr.intercept), (0.0, 10000.0))

            def _horizontal(regr) -> tuple[tuple[float, float], tuple[float, float]]:
                """Logic for computing the points with horizontal regression."""
                if regr.slope < 0.0:
                    return ((0.0, -regr.intercept / regr.slope), (regr.intercept, 0.0))
                else:
                    return ((0.0, 10000.0), (regr.intercept, regr.intercept))

            if horizontal_regr and vertical_regr:
                # Use regression with better fit (higher rvalue)
                if vertical_regr.rvalue >= horizontal_regr.rvalue:
                    modal_lines[c] = _vertical(vertical_regr)
                else:
                    modal_lines[c] = _horizontal(horizontal_regr)

            elif horizontal_regr:
                modal_lines[c] = _horizontal(horizontal_regr)

            elif vertical_regr:
                modal_lines[c] = _vertical(vertical_regr)

            else:
                pass  # No regression is possible (eg median not intersecting the grid)

        return modal_lines
