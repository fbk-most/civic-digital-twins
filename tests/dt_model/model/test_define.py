"""Tests for the @define decorator."""

# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from civic_digital_twins.dt_model import NumpyBackend, define, expose, functions, inputs, outputs
from civic_digital_twins.dt_model.model.index import Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation

# ---------------------------------------------------------------------------
# Decoration-time error checks
# ---------------------------------------------------------------------------


def test_define_raises_if_compute_absent():
    """@define raises TypeError at decoration time when compute() is missing."""
    with pytest.raises(TypeError, match="requires.*compute"):

        @define("M")
        class M(Model):
            @inputs
            class Inputs:
                x: Index

            @outputs
            class Outputs:
                y: Index


def test_define_raises_if_init_defined():
    """@define raises TypeError when the class also defines __init__."""
    with pytest.raises(TypeError, match="must not define __init__"):

        @define("M")
        class M(Model, legacy=True):  # legacy=True suppresses __init_subclass__ warning
            @inputs
            class Inputs:
                x: Index

            @outputs
            class Outputs:
                y: Index

            def __init__(self) -> None:  # noqa: D107
                pass

            def compute(self, inputs: Inputs) -> Outputs:
                """Return a dummy output."""
                return M.Outputs(y=Index("y", 1.0))


def test_define_raises_if_expose_declared_but_missing_from_return():
    """@define raises TypeError when @expose Expose is declared but compute returns only Outputs."""
    with pytest.raises(TypeError, match="declares an @expose Expose inner class"):

        @define("M")
        class M(Model):
            @inputs
            class Inputs:
                x: Index

            @outputs
            class Outputs:
                y: Index

            @expose
            class Expose:
                z: Index

            def compute(self, inputs: Inputs) -> Outputs:  # missing Expose in return
                """Return a dummy output."""
                return M.Outputs(y=Index("y", 1.0))


def test_define_get_type_hints_fallback():
    """@define falls back to raw __annotations__ when get_type_hints() fails."""
    # Use a string annotation that cannot be resolved to trigger the except branch.
    # The decorator should still succeed (falling back to raw annotations, which
    # yields no 'return' hint), so no Expose-consistency error fires either.
    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            y: Index

        def compute(self, inputs: Inputs) -> "NonExistentType":  # type: ignore[name-defined]  # noqa: F821
            """Return a dummy output — annotation is deliberately unresolvable."""
            return M.Outputs(y=Index("y", 1.0))

    # The class was decorated successfully; instantiate to confirm.
    x = Index("x", 1.0)
    m = M(inputs=M.Inputs(x=x))  # type: ignore[call-arg]
    assert m.outputs.y is not None


# ---------------------------------------------------------------------------
# Generated __init__ paths
# ---------------------------------------------------------------------------


def test_define_with_functions_and_expose():
    """Generated _init_with_fns dispatches correctly when returns_expose=True."""
    p_x = __import__("civic_digital_twins.dt_model.engine.frontend.graph", fromlist=["placeholder"]).placeholder(
        "x", default_value=3.0
    )
    fc = __import__("civic_digital_twins.dt_model.engine.frontend.graph", fromlist=["function_call"]).function_call(
        "double", p_x
    )

    x_idx = Index("x", p_x)
    y_idx = Index("y", fc)
    z_idx = Index("z", 1.0)

    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            x: Index

        @functions
        class Functions:
            double: Any

        @outputs
        class Outputs:
            y: Index

        @expose
        class Expose:
            z: Index

        def compute(self, inputs: Inputs, *, fns: Functions) -> tuple[Outputs, Expose]:
            """Compute y via function_call, expose z."""
            return M.Outputs(y=y_idx), M.Expose(z=z_idx)

    functor = NumpyBackend.adapt(lambda x: x * 2)
    m = M(inputs=M.Inputs(x=x_idx), fns=M.Functions(double=functor))  # type: ignore[call-arg]

    result = Evaluation(m).evaluate(backend=NumpyBackend)
    assert float(result[y_idx]) == pytest.approx(6.0)
    assert m.expose.z is z_idx
