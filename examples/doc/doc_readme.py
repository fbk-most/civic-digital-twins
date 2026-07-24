"""Runnable snippets from README.md."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, Model, define, inputs, outputs
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor

# ---------------------------------------------------------------------------
# README — Engine layer snippet
# ---------------------------------------------------------------------------

a = graph.placeholder("a")
b = graph.placeholder("b")
c = a * 2 + b

state = executor.State(values={a: np.asarray(3.0), b: np.asarray(1.0)})
executor.evaluate_nodes(state, *linearize.forest(c))
assert float(state.get_node_value(c)) == 7.0  # 7.0

# ---------------------------------------------------------------------------
# README — Model layer snippet
# ---------------------------------------------------------------------------


@define("example")
class ExampleModel(Model):
    @inputs
    class Inputs:
        x: DistributionIndex
        y: DistributionIndex

    @outputs
    class Outputs:
        result: Index

    def compute(self, inputs: Inputs) -> Outputs:
        result = Index("result", inputs.x + inputs.y)
        return ExampleModel.Outputs(result=result)


model = ExampleModel(inputs=ExampleModel.Inputs(
    x=DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 1.0}),
    y=DistributionIndex("y", stats.uniform, {"loc": 0.0, "scale": 1.0}),
))

assert model is not None
assert len(model.abstract_indexes()) == 2  # x and y

if __name__ == "__main__":
    print("doc_readme.py: all snippets OK")
