"""Simple interim example to test the JIT features.

Not meant to be merged to the main branch.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np

from civic_digital_twins.dt_model.engine.frontend import forest, graph
from civic_digital_twins.dt_model.engine.numpybackend import executor


def main():
    """Create and evaluate a simple model."""
    # create the graph placeholders
    a = graph.placeholder("a")
    b = graph.placeholder("b")

    # create the first leaf
    c = graph.exp(a) + 55 / a

    # create the second leaf
    d = c * b + 1024

    # create the third leaf
    e = graph.power(a, c) * 144

    # partition the graph into trees
    trees = forest.partition(c, e, d)

    # be prepared for executing the code
    state = executor.State(
        values={
            a: np.asarray([4, 3]),
            b: np.asarray([2, 0.1]),
        },
    )

    # evaluate all the functions in the graph
    for tree in trees:
        executor.evaluate_single_tree(state, tree)

    # print the results
    print("=== c ===")
    print(state.get_node_value(c))

    print("=== d ===")
    print(state.get_node_value(d))

    print("=== e ===")
    print(state.get_node_value(e))


if __name__ == "__main__":
    main()
