"""Simple example to benchmark the experimental Numba JIT compiler.

By default this script does NOT use JIT compilation. To enable it:

    export DTMODEL_ENGINE_FLAGS=jit

To change the number of repetitions:

    export _JIT_EXPERIMENT_RUNS=100

Usage::

    uv run python examples/jit.py
"""

# SPDX-License-Identifier: Apache-2.0

import os
import time

import numpy as np

from civic_digital_twins.dt_model.engine.frontend import forest, graph, ir
from civic_digital_twins.dt_model.engine.numpybackend import executor


def main() -> None:
    """Create a small computation graph and evaluate it, optionally using JIT."""
    # Build the graph
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    c = graph.exp(a) + 55 / a  # first root
    d = c * b + 1024  # second root
    e = graph.power(a, c) * 144  # third root

    # Partition into trees and compile to DAG IR
    trees = forest.partition(c, e, d)
    dag = ir.compile_trees(*trees)

    # Configuration
    reps = max(1, int(os.getenv("_JIT_EXPERIMENT_RUNS") or "1"))
    print(f"DTMODEL_ENGINE_FLAGS={os.getenv('DTMODEL_ENGINE_FLAGS')}")
    print(f"_JIT_EXPERIMENT_RUNS={reps}")

    # Run the experiment; reuse the jitted cache across iterations
    begin = time.time()
    for _ in range(reps):
        state = executor.State(
            values={
                a: np.asarray([4.0, 3.0]),
                b: np.asarray([2.0, 0.1]),
            },
        )
        executor.evaluate_dag(state, dag)

        if reps == 1:
            print("=== c ===")
            print(state.get_node_value(c))
            print("=== d ===")
            print(state.get_node_value(d))
            print("=== e ===")
            print(state.get_node_value(e))

    elapsed = time.time() - begin
    print(f"elapsed: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
