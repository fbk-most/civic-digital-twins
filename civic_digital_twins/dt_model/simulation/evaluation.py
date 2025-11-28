"""Code to evaluate a model in specific conditions."""

import os
from functools import reduce
import numpy as np
from scipy import interpolate, ndimage, stats
from concurrent.futures import ThreadPoolExecutor

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

    def _prepare_and_evaluate_nodes(self, asset, grid_mode=True):
        """Common preprocessing and node evaluation logic for the ensemble."""
        if self.inst.values is None:
            assignments = {}
        else:
            assignments = self.inst.values

        # [pre] extract the weights and the size of the ensemble
        c_weight = np.array([c[0] for c in self.ensemble], dtype=float)
        # normalize weights so they always sum to 1 (protects against slicing)
        w_sum = c_weight.sum()
        if w_sum <= 0:
            # fallback to uniform weights to avoid divide-by-zero
            c_weight = np.full_like(c_weight, 1.0 / c_weight.size)
        else:
            c_weight = c_weight / w_sum
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
            if grid_mode:
                c_subs[key] = np.expand_dims(c_subs[key], axis=(0, 1))
            else:
                c_subs[key] = np.expand_dims(c_subs[key], axis=0)

        # [eval] add presence variables and expand dimensions
        assert len(self.inst.abs.pvs) == 2  # TODO: generalize
        if grid_mode:
            for i, pv in enumerate(self.inst.abs.pvs):
                c_subs[pv.node] = np.expand_dims(asset[pv], axis=(i, 2))
        else:
            for i, pv in enumerate(self.inst.abs.pvs):
                c_subs[pv.node] = np.expand_dims(asset[i], axis=1)  # CHANGED

        # [eval] collect all nodes to evaluate
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

        if grid_mode:
            return c_subs, c_weight, c_size, assignments
        else:
            return c_subs

    def evaluate_grid(self, grid):
        """Evaluate the model according to the grid."""

        c_subs, c_weight, c_size, assignments = self._prepare_and_evaluate_nodes(
            grid, True
        )

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
                unscaled_result = usage <= _fix_shapes(
                    np.asarray(c_subs[capacity.node])
                )
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

        # [post] compute confidence field
        fields_stack = np.stack(list(field_elements.values()), axis=0)
        variance_map = np.var(fields_stack, axis=0)
        mean_map = np.mean(fields_stack, axis=0)
        eps = 1e-9
        confidence_field = 1.0 - (variance_map / (np.abs(mean_map) + eps))
        confidence_field = np.clip(confidence_field, 0.0, 1.0)
        self.confidence_field = confidence_field
        return self.field

    def evaluate_grid_for_subensemble(self, grid, sub_ensemble):
        """
        Stateless batch evaluation suitable for running in a DAG node.
        Does NOT modify self.ensemble or any class-level state.
        Returns (field_batch, conf_batch).
        """

        # Temporarily override the ensemble (local only)
        original_ensemble = self.ensemble
        try:
            self.ensemble = sub_ensemble
            field = self.evaluate_grid(grid)
            conf = self.confidence_field
            return field, conf
        finally:
            # Always restore state
            self.ensemble = original_ensemble

    def aggregate_batch_results(
        self, batch_results, previous_state, early_stopping_params
    ):
        """
        Aggregates a list of batch results:
            batch_results = [(field_batch, conf_batch), ...]
        previous_state contains:
            cumulative_field, cumulative_weights,
            cumulative_conf, conf_weights,
            prev_field, batches_completed

        Returns:
            new_state, stop_decision (bool)
        """

        # Unpack state
        (
            cumulative_field,
            cumulative_weights,
            cumulative_conf,
            conf_weights,
            prev_field,
            batches_completed,
        ) = previous_state

        stop = False

        # Process each batch result
        for field_batch, conf_batch in batch_results:
            cumulative_field += field_batch
            cumulative_weights += 1.0

            # Confidence accumulation
            if cumulative_conf is None:
                cumulative_conf = conf_batch.copy()
                conf_weights = np.ones_like(conf_batch)
            else:
                cumulative_conf += conf_batch
                conf_weights += 1.0

            batches_completed += 1

        # Compute aggregate confidence
        mean_conf_grid = cumulative_conf / np.maximum(conf_weights, 1e-9)
        total_conf = float(np.mean(mean_conf_grid))

        # Evaluate early stopping
        if batches_completed >= early_stopping_params["min_batch_iterations"]:

            # Criterion 1: global confidence threshold
            if total_conf >= early_stopping_params["confidence_threshold"]:
                stop = True

            # Criterion 2: field stability
            # Only check last batch of the group
            last_field = batch_results[-1][0]
            if prev_field is not None:
                delta = np.mean(np.abs(last_field - prev_field))
                if delta < early_stopping_params["stability_tolerance"]:
                    stop = True

            prev_field = last_field.copy()

        new_state = (
            cumulative_field,
            cumulative_weights,
            cumulative_conf,
            conf_weights,
            prev_field,
            batches_completed,
        )

        return new_state, stop

    from concurrent.futures import ThreadPoolExecutor

    def evaluate_grid_incremental(
        self, grid, early_stopping_params, threads_per_worker=1
    ):
        """Evaluate ensemble incrementally using batches with threads."""
        ensemble = list(self.ensemble)
        total_batches = int(
            np.ceil(len(ensemble) / early_stopping_params["batch_size"])
        )
        N = early_stopping_params["evaluate_every_n_batches"]
        tmp = 4
        max_threads = min(N, threads_per_worker, os.cpu_count())

        # Prepare empty cumulative state
        grid_shape = (grid[self.inst.abs.pvs[0]].size, grid[self.inst.abs.pvs[1]].size)
        state = (
            np.zeros(grid_shape),  # cumulative_field
            np.zeros(grid_shape),  # cumulative_weights
            None,  # cumulative_conf
            None,  # conf_weights
            None,  # prev_field
            0,  # batches_completed
        )

        for batch_group_start in range(0, total_batches, N):
            batch_indices = range(
                batch_group_start, min(batch_group_start + N, total_batches)
            )

            # Evaluate N batches in parallel threads
            def process_batch(batch_idx):
                start = batch_idx * early_stopping_params["batch_size"]
                end = min(start + early_stopping_params["batch_size"], len(ensemble))
                sub_ensemble = ensemble[start:end]
                return self.evaluate_grid_for_subensemble(grid, sub_ensemble)

            with ThreadPoolExecutor(max_workers=max_threads) as executor_pool:
                batch_results = list(executor_pool.map(process_batch, batch_indices))

            # Aggregate results
            state, stop = self.aggregate_batch_results(
                batch_results, state, early_stopping_params
            )
            if stop:
                print("[Early stop triggered]")
                break

        # Final normalization
        cumulative_field, cumulative_weights, *_ = state
        self.field = cumulative_field / np.maximum(cumulative_weights, 1e-9)
        self.grid = grid
        return self.field

    def evaluate_usage(self, presences):
        """Evaluate the model according to the presence argument."""

        c_subs = self._prepare_and_evaluate_nodes(presences, False)

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
            lambda x, y: x * y,
            [axis.max() / (axis.size - 1) + 1 for axis in list(grid.values())],
        )

    # TODO: use evaluate_usage instead of evaluate_grid?
    # TODO: change API - order of presence variables
    def compute_sustainability_index(self, presences: list) -> float:
        """Compute the sustainability index."""
        assert self.grid is not None
        grid = self.grid
        field = self.field
        # TODO: fill value
        index = interpolate.interpn(
            grid.values(),
            field,
            np.array(presences),
            bounds_error=False,
            fill_value=0.0,
        )
        return np.mean(index)  # type: ignore

    def compute_sustainability_index_with_ci(
        self, presences: list, confidence: float = 0.9
    ) -> (float, float):
        """Compute the sustainability index with confidence value."""
        assert self.grid is not None
        grid = self.grid
        field = self.field
        # TODO: fill value
        index = interpolate.interpn(
            grid.values(),
            field,
            np.array(presences),
            bounds_error=False,
            fill_value=0.0,
        )
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
                grid.values(),
                field_elements[c],
                np.array(presences),
                bounds_error=False,
                fill_value=0.0,
            )
            indexes[c] = np.mean(index)
        return indexes

    def compute_sustainability_index_with_ci_per_constraint(
        self, presences: list, confidence: float = 0.9
    ) -> dict:
        """Compute the sustainability index with confidence value for each constraint."""
        assert self.grid is not None
        assert self.field_elements is not None

        grid = self.grid
        field_elements = self.field_elements
        # TODO: fill value
        indexes = {}
        for c in self.inst.abs.constraints:
            index = interpolate.interpn(
                grid.values(),
                field_elements[c],
                np.array(presences),
                bounds_error=False,
                fill_value=0.0,
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
                horizontal_regr = stats.linregress(
                    grid[self.inst.abs.pvs[0]][xi], grid[self.inst.abs.pvs[1]][yi]
                )
            except ValueError:
                pass
            try:
                vertical_regr = stats.linregress(
                    grid[self.inst.abs.pvs[1]][yi], grid[self.inst.abs.pvs[0]][xi]
                )
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
