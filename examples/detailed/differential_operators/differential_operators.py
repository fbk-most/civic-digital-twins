# SPDX-License-Identifier: Apache-2.0

"""Differential operators on a SpaceType axis: gradient, laplacian, boundaries.

Narrative and explanations: see Differential_Operators_Tutorial.md in this
directory. Each "PART N" banner below matches a section of the same name there.

  PART 1  What .gradient() computes: a directional derivative along ONE axis.
  PART 2  The 2-D case: .gradient is per-axis, .laplacian is multi-axis.
  PART 3  Boundary conditions: every policy, against both operators.
  PART 4  The ORDER of axes= on an injected array is load-bearing and unchecked.

Deliberately abstract naming throughout: f is the field, x/y/depth are axes.
"""

import numpy as np

from civic_digital_twins.dt_model import (
    DistributionEnsemble,
    Evaluation,
    Index,
    Model,
    Scenario,
    define,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.axes import (
    Constant,
    DomainAxis,
    Linear,
    Nearest,
    Neumann,
    SpaceType,
    Wrap,
)

# =============================================================================
# PART 1 -- what .gradient(axis=...) actually computes
# =============================================================================
# Deliberately non-linear data: a straight line would hide the mechanism.
DEPTH_VALS = np.array([0.0, 10.0, 20.0, 50.0])   # note the jump at the end

print("\n=== PART 1 -- what .gradient(axis=...) computes ===")
print(f"  temperature down a borehole: {DEPTH_VALS}")

for spacing in (1.0, 2.0):
    grid = DomainAxis("depth", type=SpaceType(spacing=spacing))

    @define(f"grad_{spacing}")
    class Grad(Model):
        @inputs
        class Inputs:
            T: Index

        @outputs
        class Outputs:
            slope: Index   # d(temperature) / d(depth)
            step: Index    # raw difference, for contrast

        def compute(self, inputs: Inputs) -> Outputs:
            return Grad.Outputs(
                slope=Index("slope", inputs.T.gradient(axis=grid), axes=(grid,)),
                step=Index("step", inputs.T.diff(axis=grid), axes=(grid,)),
            )

    gmodel = Grad(inputs=Grad.Inputs(T=Index("T", DEPTH_VALS, axes=(grid,))))
    gscenario = Scenario(gmodel)
    gres = Evaluation(gscenario).evaluate(
        ensemble=DistributionEnsemble(gscenario, size=1),
    )
    slope = gres.labeled(gmodel.outputs.slope).values   # dims ('depth',)
    step = gres.labeled(gmodel.outputs.step).values
    print(f"\n  spacing={spacing:g} m")
    print(f"    .gradient -> {slope}   (degrees per METRE)")
    print(f"    .diff     -> {step}   (degrees per SAMPLE, spacing ignored)")


# =============================================================================
# PART 2 -- the 2-D case: gradient is per-axis, laplacian is multi-axis
# =============================================================================
# f(x, y) = x^2:  df/dx = 2x,  df/dy = 0,  laplacian = 2 + 0 = 2
y_axis = DomainAxis("y", type=SpaceType(spacing=1.0))
x_axis = DomainAxis("x", type=SpaceType(spacing=1.0))

XS = np.arange(4.0)                                   # x = 0,1,2,3
FIELD = (XS**2)[None, :] * np.ones((3, 1))            # f = x^2, 3 rows of y
#        ^ shape (3, 4): rows indexed by y, columns by x -> axes=(y_axis, x_axis)


@define("field2d")
class Field2D(Model):
    @inputs
    class Inputs:
        F: Index

    @outputs
    class Outputs:
        df_dx: Index   # one component of the gradient
        df_dy: Index   # the other component
        lap: Index     # both second derivatives, summed

    def compute(self, inputs: Inputs) -> Outputs:
        return Field2D.Outputs(
            df_dx=Index("df_dx", inputs.F.gradient(axis=x_axis), axes=(y_axis, x_axis)),
            df_dy=Index("df_dy", inputs.F.gradient(axis=y_axis), axes=(y_axis, x_axis)),
            # axes=(y,x) -- BOTH axes, summed into a single field.
            lap=Index("lap", inputs.F.laplacian(axes=(y_axis, x_axis)),
                      axes=(y_axis, x_axis)),
        )


fmodel = Field2D(inputs=Field2D.Inputs(F=Index("F", FIELD, axes=(y_axis, x_axis))))
fscenario = Scenario(fmodel)
fresult = Evaluation(fscenario).evaluate(
    ensemble=DistributionEnsemble(fscenario, size=1),
)

# Results come back with dims ('x', 'y'): check .dims, never assume the order.
df_dx = fresult.labeled(fmodel.outputs.df_dx)
df_dy = fresult.labeled(fmodel.outputs.df_dy)
lap = fresult.labeled(fmodel.outputs.lap)


print("\n=== PART 2 -- a 2-D field: f(x, y) = x^2, constant along y ===")
print(f"  f (rows are y, columns are x = {XS}):")
print(FIELD)

print("\n  .gradient(axis=x)  ->  df/dx, exact answer 2x = [0, 2, 4, 6]")
print(df_dx.values.T)        # .T: rows are y, as in FIELD

print("\n  .gradient(axis=y)  ->  df/dy, exact answer 0 everywhere")
print(df_dy.values.T)

print("\n  .laplacian(axes=(y, x))  ->  d2f/dx2 + d2f/dy2, exact answer 2")
print(lap.values.T)


# =============================================================================
# PART 3 -- boundary conditions: both operators, every policy
# =============================================================================
#     gradient   (f[i+1] - f[i-1]) / (2h)
#     laplacian  (f[i-1] - 2 f[i] + f[i+1]) / h^2
# run everywhere; the boundary rule fills one ghost cell past each edge.
print("\n=== PART 3 -- boundary conditions: both operators, every policy ===")
EDGE_ROW = np.array([0.0, 10.0, 20.0, 50.0])
print(f"  f = {EDGE_ROW}   (spacing h = 1)")
print(f"\n  {'boundary':20s} {'gradient':26s} {'laplacian':26s}")

for bc, label in (
    (Neumann(0.0), "Neumann(0)  default"),
    (Neumann(5.0), "Neumann(5)"),
    (Constant(0.0), "Constant(0)"),
    (Constant(100.0), "Constant(100)"),
    (Nearest(), "Nearest()"),
    (Wrap(), "Wrap()"),
    (Linear(), "Linear()"),
):
    edge_axis = DomainAxis("edge", type=SpaceType(spacing=1.0, boundary=bc))

    @define(f"edge_{label}")
    class EdgeModel(Model):
        @inputs
        class Inputs:
            T: Index

        @outputs
        class Outputs:
            grad: Index
            lap: Index

        def compute(self, inputs: Inputs) -> Outputs:
            return EdgeModel.Outputs(
                grad=Index("grad", inputs.T.gradient(axis=edge_axis),
                           axes=(edge_axis,)),
                lap=Index("lap", inputs.T.laplacian(axes=(edge_axis,)),
                          axes=(edge_axis,)),
            )

    edge_model = EdgeModel(inputs=EdgeModel.Inputs(
        T=Index("T", EDGE_ROW, axes=(edge_axis,)),
    ))
    edge_scenario = Scenario(edge_model)
    edge_res = Evaluation(edge_scenario).evaluate(
        ensemble=DistributionEnsemble(edge_scenario, size=1),
    )
    g = edge_res.labeled(edge_model.outputs.grad).values
    lp = edge_res.labeled(edge_model.outputs.lap).values
    print(f"  {label:20s} {str(g):26s} {str(lp):26s}")


# =============================================================================
# PART 4 -- axis ORDER on input is load-bearing (and unchecked)
# =============================================================================
print("\n=== PART 4 -- axis ORDER on input is load-bearing (and unchecked) ===")
for label, order in (("axes=(y,x)  correct", (y_axis, x_axis)),
                     ("axes=(x,y)  swapped", (x_axis, y_axis))):
    swapped = Index("F", FIELD, axes=order)
    print(f"  {label}: recorded sizes {swapped.sizes}")
