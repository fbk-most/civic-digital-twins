"""Simple interim example to verify the codegen features.

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

    # print the trees that we have created
    print("=== trees ===")
    for idx, tree in enumerate(trees):
        print(f"--- tree #{idx} ---")
        print(tree.format())
        print("")
    print("")

    # be prepared for executing the code
    state = executor.State(
        values={
            a: np.asarray(4),
            b: np.asarray(2),
        }
    )

    # evaluate all the trees in the graph
    for tree in trees:
        executor.evaluate_tree(state, tree)

    # print the results
    print("=== c ===")
    print(state.get_node_value(c))

    print("=== d ===")
    print(state.get_node_value(d))

    print("=== e ===")
    print(state.get_node_value(e))


if __name__ == "__main__":
    main()
