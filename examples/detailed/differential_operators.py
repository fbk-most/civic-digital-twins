"""Differential operators on a SpaceType axis: gradient, laplacian, boundaries.

An axis declared with SpaceType carries two pieces of physical metadata that
nothing else in the library provides:

    spacing    how far apart two neighbouring samples are
    boundary   what the field does just OUTSIDE the domain

Those two unlock .gradient() and .laplacian(), and they decide the numbers
those operators return. This file covers:

  PART 1  What .gradient() actually computes -- a DIRECTIONAL derivative along
          ONE named axis, not the vector-calculus gradient over all of them.
          Also why spacing matters, contrasted with .diff().

  PART 2  The 2-D case, where the split between the two operators is clearest:
          .gradient takes ONE axis and returns one partial derivative (call it
          twice for a 2-D gradient), while .laplacian takes SEVERAL axes and
          sums them into a single scalar field.

  PART 3  Boundary conditions. Both operators use the same central-difference
          stencil everywhere, including at the edges, where the missing
          neighbour comes from a ghost cell filled in by the axis's boundary
          rule. Constant(v) and Neumann(v) state an actual physical edge;
          Nearest/Wrap/Linear extrapolate. We run all of them against both
          operators so the effect is visible rather than described.

  PART 4  A trap that bites here specifically: the ORDER of axes= on an
          injected array is load-bearing and unchecked, and getting it wrong
          silently differentiates along the wrong dimension.

Deliberately abstract naming throughout: f is the field, x/y/depth are axes.
"""

import numpy as np

from civic_digital_twins.dt_model import (
    define, inputs, outputs, Index, Model,
    Scenario, DistributionEnsemble, Evaluation,
)
from civic_digital_twins.dt_model.axes import (
    Axis, DOMAIN, DomainAxis, SpaceType,
    Constant, Neumann, Nearest, Wrap, Linear,
)

# =============================================================================
# What gradient() actually computes
# =============================================================================
# "Gradient" in vector calculus is the vector of partial derivatives over ALL
# dimensions. That is not what this operator is. `.gradient(axis=ax)` takes a
# DIRECTIONAL derivative along ONE named axis -- a single partial derivative,
# d(value)/d(ax) -- and is perfectly well defined in 1-D. It is exactly
# np.gradient(values, spacing, axis=<that axis>); the library's backend calls
# precisely that. For a full multi-dimensional gradient you would call it once
# per axis and keep the results separately.
#
# How each point is computed:
#   EVERY point -- CENTRAL difference: (next - prev) / (2 * spacing)
# including the two edges, which get their missing neighbour from a ghost cell
# filled in by the axis's BOUNDARY CONDITION. The output therefore has the SAME
# length as the input; nothing is dropped, and nothing uses a different formula.
#
# We use deliberately non-linear data below. A straight line would make every
# output value identical and hide the mechanism completely.
DEPTH_VALS = np.array([0.0, 10.0, 20.0, 50.0])   # note the jump at the end

print("\n=== what .gradient(axis=...) computes ===")
print(f"  temperature down a borehole: {DEPTH_VALS}")
print("  it is a DIRECTIONAL derivative along ONE axis (d temp / d depth),")
print("  not the vector-calculus gradient over all axes.")

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
    slope = np.squeeze(gres.labeled(gmodel.outputs.slope).values)
    step = np.squeeze(gres.labeled(gmodel.outputs.step).values)
    print(f"\n  spacing={spacing:g} m")
    print(f"    .gradient -> {slope}   (degrees per METRE)")
    print(f"    .diff     -> {step}   (degrees per SAMPLE, spacing ignored)")

print("""
  Checking the spacing=1 row by hand. Every position uses the SAME central
  difference (f[i+1] - f[i-1]) / (2h); the two edges get their missing
  neighbour from the axis's boundary condition, which defaults to Neumann(0)
  -- "the slope just outside is zero":
    i=0  edge     slope pinned to 0 by Neumann(0)   =  0
    i=1  interior (20 -  0) / (2 * 1)               = 10
    i=2  interior (50 - 10) / (2 * 1)               = 20
    i=3  edge     slope pinned to 0 by Neumann(0)   =  0
  So the default deliberately reports a flat edge rather than guessing a
  one-sided slope. If a real slope at the boundary matters to you, say so:
  Linear() extends the local trend, Nearest() repeats the edge cell, and
  Constant(v) / Neumann(v) state an actual physical condition. The section
  below shows all of them side by side.

  Doubling the spacing halves every value: the same temperature change spread
  over twice the distance is half the rate. That is the whole point of the
  metric -- .diff is unchanged by spacing because it never divides by it, so
  .diff answers "per sample" while .gradient answers "per metre".""")

# =============================================================================
# The 2-D case: gradient is per-axis, laplacian is multi-axis
# =============================================================================
# This is where the difference between the two operators becomes obvious:
#
#   .gradient(axis=ax)    ONE axis, one partial derivative. To get the full
#                         vector-calculus gradient in 2-D you call it TWICE,
#                         once per axis, and keep both components.
#   .laplacian(axes=(..)) MULTIPLE axes at once. It sums the second derivative
#                         over every axis given, collapsing them into ONE
#                         scalar field. Note the plural keyword.
#
# We use f(x, y) = x^2, which is constant in y, because its derivatives are
# known exactly and so every printed number can be checked:
#   df/dx = 2x      df/dy = 0      laplacian = d2f/dx2 + d2f/dy2 = 2 + 0 = 2
# Why y BEFORE x below: when you attach axes= to an injected array, the axes
# are zipped POSITIONALLY against that array's shape. FIELD has shape (3, 4) --
# 3 rows, 4 columns -- so axes=(y_axis, x_axis) declares "first dimension is y
# with 3 points, second is x with 4". The order follows the array, not taste.
#
# Careful: the wrong order is NOT rejected. axes=(x_axis, y_axis) builds
# happily and simply records x:3, y:4 -- it relabels your dimensions. Every
# later gradient(axis=x_axis) would then differentiate down the rows instead
# of across the columns and return zeros here, with no error anywhere. Axis
# order is load-bearing on input; only afterwards is it safe to ignore it and
# select by name (as labeled() does).
y_axis = DomainAxis("y", type=SpaceType(spacing=1.0))
x_axis = DomainAxis("x", type=SpaceType(spacing=1.0))

XS = np.arange(4.0)                                   # x = 0,1,2,3
FIELD = (XS**2)[None, :] * np.ones((3, 1))            # f = x^2, 3 rows of y
#        ^ shape (3, 4): rows indexed by y, columns by x


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


def field(name):
    """An output as a (y, x) array, transposed by NAME rather than by luck."""
    lab = fresult.labeled(getattr(fmodel.outputs, name))
    vals = np.squeeze(lab.values)
    dims = tuple(d for d in lab.dims if d != "_ensemble")
    return np.transpose(vals, [dims.index(n) for n in ("y", "x")])


print("\n=== a 2-D field: f(x, y) = x^2, constant along y ===")
print(f"  f (rows are y, columns are x = {XS}):")
print(FIELD)

print("\n  .gradient(axis=x)  ->  df/dx, exact answer 2x = [0, 2, 4, 6]")
print(field("df_dx"))
print("    interior points are exact (2 and 4). The EDGES are set by the")
print("    default Neumann(0) boundary, which pins the slope there to 0 --")
print("    right by luck at x=0, wrong at x=3 where the true value is 6.")

print("\n  .gradient(axis=y)  ->  df/dy, exact answer 0 everywhere")
print(field("df_dy"))
print("    f does not vary along y, so this component is identically zero.")
print("    Two calls, two components: THAT is the full 2-D gradient.")

print("\n  .laplacian(axes=(y, x))  ->  d2f/dx2 + d2f/dy2, exact answer 2")
print(field("lap"))
print("    one scalar field, not two: the axes were summed over, not kept.")
print("    Interior columns are exactly 2 (= 2 from x, 0 from y).")
# =============================================================================
# HOW THE EDGES ARE OBTAINED -- boundary conditions
# =============================================================================
# Both operators keep the output the same length as the input, so both need a
# value for the cell just PAST each edge. They get it the same way: the axis's
# SpaceType carries a BOUNDARY CONDITION, the kernel pads one ghost cell on
# each side according to it, and then the SAME central-difference stencil runs
# everywhere -- edges included.
#
#     gradient   (f[i+1] - f[i-1]) / (2h)
#     laplacian  (f[i-1] - 2 f[i] + f[i+1]) / h^2
#
# So a boundary condition is really just "the rule for filling in one value
# past each edge". The available rules split into two families:
#
#   VALUE-fixing (Dirichlet):  Constant(v)  -- the field just outside is v.
#                              Dirichlet(v) is an alias for the same thing.
#   DERIVATIVE-fixing (Neumann): Neumann(v) -- the slope just outside is v.
#                              Reflect(v) is an alias. Neumann(0.0) is the
#                              DEFAULT, i.e. a "no flux through the wall" edge.
#   structural:  Nearest()  repeat the outermost cell
#                Wrap()     periodic -- opposite borders connect
#                Linear()   extend the local linear trend
#
# Both operators respond to the choice, and the two value-carrying ones let you
# state an actual physical edge condition (a wall held at 20 degrees, a known
# flux) rather than only picking an extrapolation shape.
print("\n=== boundary conditions: both operators, every policy ===")
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
    g = np.squeeze(edge_res.labeled(edge_model.outputs.grad).values)
    lp = np.squeeze(edge_res.labeled(edge_model.outputs.lap).values)
    print(f"  {label:20s} {str(g):26s} {str(lp):26s}")

print("""
  Reading the table:
    * The INTERIOR values never move (gradient 10 and 20, laplacian 0 and 20).
      A boundary condition only ever changes the two end positions.
    * Neumann(0) forces the slope to 0 at both ends, so the gradient's first
      and last entries are exactly 0. Neumann(5) forces them to 5. That is a
      DERIVATIVE being pinned -- read straight off the output.
    * Constant(100) puts a large value just outside the left edge, so the
      gradient there goes sharply negative: the field appears to fall steeply
      as you enter the domain.
    * Linear() extends the local trend, which makes the laplacian 0 at both
      ends -- a straight line has no curvature.
  Pick the rule that matches the physics: Wrap for a periodic domain,
  Constant(v) for a boundary held at a known value, Neumann(v) for a known
  flux, Nearest/Linear for a neutral continuation.""")

# --- why the ORDER of axes= matters, demonstrated ---------------------------
# Declaring the axes in the wrong order is accepted silently, so this failure
# is worth seeing once: the same array, the same request for df/dx, two orders.

# =============================================================================
# PART 4 -- axis ORDER on input is load-bearing (and unchecked)
# =============================================================================
# Declaring the axes in the wrong order is accepted silently, so this failure
# is worth seeing once: the same array, the same request for df/dx, two orders.
print("\n=== axis ORDER on input is load-bearing (and unchecked) ===")
for label, order in (("axes=(y,x)  correct", (y_axis, x_axis)),
                     ("axes=(x,y)  swapped", (x_axis, y_axis))):
    swapped = Index("F", FIELD, axes=order)
    print(f"  {label}: recorded sizes {swapped.sizes}")

print("""
  Both are accepted -- the second just relabels the dimensions. Asking for
  df/dx then differentiates down the ROWS instead of across the columns, and
  since f is constant along y that returns all zeros rather than 2x. A wrong
  answer with no warning, which is why axes= must follow the array's shape.""")
