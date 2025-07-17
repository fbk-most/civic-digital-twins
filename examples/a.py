from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor

import numpy as np

a = graph.placeholder("a")
b = graph.placeholder("b")
c = a + b
d = graph.function("_userfunc", a=a, c=c)
e = d - 1

plan = linearize.forest(e)

for node in plan:
    print(node)

state = executor.State(
    values={
        a: np.asarray(14),
        b: np.asarray(17),
    },
    functions={
        d: executor.LambdaAdapter(lambda *, a, c: np.add(a, c))
    },
)
executor.evaluate_nodes(state, *plan)
print(state.get_node_value(e))
