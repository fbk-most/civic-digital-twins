"""Provide a NumPy Backend for evaluating computation graphs.

This package transforms the symbolic operations defined in the graph
into concrete numerical computations using NumPy, thus evaluating the model
represented by the computation graph.

Key Components
--------------

- executor: Contains the State class and evaluate function that execute
  a previously-topologically-sorted computation graph efficiently
  without recursion.

- dispatch: Maps symbolic operations from the frontend graph to their
  corresponding NumPy implementations through dispatch tables.

- debug: Provides utilities for tracing and visualizing graph execution,
  helping with troubleshooting and performance analysis.

Usage Example:
-------------
```python
from dt_model.engine.frontend import graph, linearize
from dt_model.engine.numpybackend import executor

import numpy as np

# Create a graph
x = graph.placeholder("x")
y = graph.placeholder("y")
z = graph.add(x, y)

# Linearize the graph
sorted_nodes = linearize.forest(z)

# Initialize state with placeholder values
state = executor.State(values={x: np.array(2), y: np.array(3)})

# Execute the graph
for node in sorted_nodes:
    executor.evaluate(state, node)

# Access result
result = state.values[z]  # array(5)
```

Implementation Details
----------------------

The executor processes nodes that have already been sorted in topological order, evaluating
each node exactly once and caching the results into the State. The dispatch system uses lookup
tables to map graph operations to NumPy functions, making it easy to extend.
"""

# SPDX-License-Identifier: Apache-2.0
