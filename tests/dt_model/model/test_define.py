"""Tests for the @define decorator."""

# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from civic_digital_twins.dt_model import NumpyBackend, config, define, expose, functions, inputs, outputs
from civic_digital_twins.dt_model.model.index import Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.simulation.scenario import Scenario

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
    m = M(inputs=M.Inputs(x=x))
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
    m = M(inputs=M.Inputs(x=x_idx), fns=M.Functions(double=functor))

    result = Evaluation(Scenario(m)).evaluate(backend=NumpyBackend)
    assert float(result[y_idx]) == pytest.approx(6.0)
    assert m.expose.z is z_idx


def test_define_empty_inputs_no_functions():
    """@define auto-constructs Inputs() when Inputs has no fields and no Functions."""

    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            y: Index

        def compute(self, inp: Inputs) -> Outputs:
            """Return a constant output."""
            return M.Outputs(y=Index("y", 1.0))

    m = M()  # no inputs argument — Inputs() is auto-constructed
    assert m.outputs.y is not None


def test_define_with_config_only():
    """Generated _init_with_config threads config into compute() and never into super()."""

    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            x: Index

        @config
        class Config:
            multiplier: float = 1.0

        @outputs
        class Outputs:
            y: Index

        def compute(self, inputs: Inputs, *, config: Config) -> Outputs:
            """Scale x by config.multiplier."""
            return M.Outputs(y=Index("y", inputs.x * config.multiplier))

    x = Index("x", 2.0)
    m = M(inputs=M.Inputs(x=x), config=M.Config(multiplier=3.0))

    result = Evaluation(Scenario(m)).evaluate(backend=NumpyBackend)
    assert float(result[m.outputs.y]) == pytest.approx(6.0)


def test_define_with_functions_and_config():
    """Generated _init_with_fns_config threads both fns and config into compute()."""

    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            x: Index

        @functions
        class Functions:
            f: Any

        @config
        class Config:
            multiplier: float = 1.0

        @outputs
        class Outputs:
            y: Index

        def compute(self, inputs: Inputs, *, fns: Functions, config: Config) -> Outputs:
            """Apply the function then scale by config.multiplier."""
            return M.Outputs(y=Index("y", inputs.x * config.multiplier))

    x = Index("x", 2.0)
    m = M(
        inputs=M.Inputs(x=x),
        fns=M.Functions(f=NumpyBackend.adapt(lambda v: v)),
        config=M.Config(multiplier=5.0),
    )

    result = Evaluation(Scenario(m)).evaluate(backend=NumpyBackend)
    assert float(result[m.outputs.y]) == pytest.approx(10.0)


def test_define_config_never_reaches_super_init():
    """Config is graph-inert: @define's generated __init__ never forwards it to super().

    This exercises the real ``@define``-generated ``_run_compute`` wiring (not
    a hand-written ``legacy=True`` init that could pass or drop config by
    construction): ``Model.__init__`` has no real ``config=`` parameter, so if
    the generated ``__init__`` ever leaked ``config`` into
    ``super().__init__()``, this call would raise ``TypeError: unexpected
    keyword argument 'config'`` before reaching the assertions below.
    """

    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            x: Index

        @config
        class Config:
            policy: str = "default"

        @outputs
        class Outputs:
            y: Index

        def compute(self, inputs: Inputs, *, config: Config) -> Outputs:
            """Return y regardless of config; config only picks a branch."""
            return M.Outputs(y=Index("y", inputs.x))

    x = Index("x", 1.0)
    m = M(inputs=M.Inputs(x=x), config=M.Config(policy="peak"))
    assert len(m.indexes) == 2  # just x and y — no trace of Config
    assert m.abstract_indexes() == []


def test_define_no_config_declared_is_unchanged():
    """Without a @config Config inner class, passing config= raises TypeError (regression guard)."""

    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            y: Index

        def compute(self, inputs: Inputs) -> Outputs:
            """Return a dummy output."""
            return M.Outputs(y=Index("y", inputs.x))

    x = Index("x", 1.0)
    with pytest.raises(TypeError, match="unexpected keyword argument 'config'"):
        M(inputs=M.Inputs(x=x), config="not declared")  # type: ignore[call-arg]


def test_define_empty_inputs_with_functions():
    """@define auto-constructs Inputs() when Inputs has no fields and Functions is declared."""

    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            pass

        @functions
        class Functions:
            f: Any

        @outputs
        class Outputs:
            y: Index

        def compute(self, inp: Inputs, *, fns: Functions) -> Outputs:
            """Return a constant output."""
            return M.Outputs(y=Index("y", 1.0))

    m = M(fns=M.Functions(f=NumpyBackend.adapt(lambda x: x)))
    assert m.outputs.y is not None
