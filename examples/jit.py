"""Simple interim example to test the JIT features.

Not meant to be merged to the main branch.

By default this code DOES NOT use jit. To enable JIT use:

    export DTMODEL_ENGINE_FLAGS=jit

This would instruct the compiler to jit as many functions as possible.

To change the number of repetitions do:

    export _JIT_EXPERIMENT_RUNS=100

This would instruct this code to repeat execution 100 times.
"""

# SPDX-License-Identifier: Apache-2.0

import os
import time
import types

import numpy as np

from civic_digital_twins.dt_model.engine.frontend import forest, graph, ir
from civic_digital_twins.dt_model.engine.numpybackend import executor


def main():
    """Create and evaluate a simple model."""
    # create the graph placeholders
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    # create the first root
    c = graph.exp(a) + 55 / a

    # create the second root
    d = c * b + 1024

    # create the third root
    e = graph.power(a, c) * 144

    # partition the graph into trees
    trees = forest.partition(c, e, d)

    # obtain the DAG IR
    dag = ir.compile_trees(*trees)

    # obtain the number of repetitions
    reps = max(1, int(os.getenv("_JIT_EXPERIMENT_RUNS") or "1"))

    # print the configuration
    print(f"DTMODEL_ENGINE_FLAGS={os.getenv('DTMODEL_ENGINE_FLAGS')}")
    print(f"_JIT_EXPERIMENT_RUNS={reps}")

    # repeat the experiment 100 times with fresh state
    begin = time.time()
    jitted: dict[forest.Tree, types.FunctionType] = {}
    for _ in range(reps):
        # be prepared for executing the code
        state = executor.State(
            values={
                a: np.asarray([4, 3]),
                b: np.asarray([2, 0.1]),
            },
            jitted=jitted,
        )

        # evaluate all the overall DAG as whole
        executor.evaluate_dag(state, dag)

        # conditionally print the results
        if reps > 1:
            continue
        print("=== c ===")
        print(state.get_node_value(c))

        print("=== d ===")
        print(state.get_node_value(d))

        print("=== e ===")
        print(state.get_node_value(e))

    end = time.time()
    print(f"elapsed: {end - begin}")


if __name__ == "__main__":
    main()
