# SPDX-License-Identifier: Apache-2.0
"""Runnable snippets from docs/design/dd-cdt-simulation.md."""

import dataclasses

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    AsyncEvaluationHandle,
    DistributionEnsemble,
    DistributionIndex,
    Evaluation,
    EvaluationConfig,
    EvaluationHandle,
    EvaluationPlan,
    EvaluationResult,
    Index,
    IncompatibleResultError,
    Model,
    ModelEvaluator,
    ModelOutput,
    ModelRunHandle,
    Scenario,
    define,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.simulation.ensemble import BatchDrawable, FrozenEnsemble


def _is_in(idx, seq) -> bool:
    """Identity-based membership test (GenericIndex.__eq__ returns a graph node)."""
    return any(idx is item for item in seq)


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md TL;DR — shared model definition
# ---------------------------------------------------------------------------


@define("Concentration")
class ConcentrationModel(Model):

    @inputs
    class Inputs:
        variability:  DistributionIndex   # uncertain multiplier (sampled per scenario)
        base_level:   Index               # concrete baseline, overridable via Scenario
        traffic_load: Index               # abstract (None) → supplied as PARAMETER axis

    @outputs
    class Outputs:
        concentration: Index

    def compute(self, inputs: Inputs) -> Outputs:
        load          = Index("load", inputs.traffic_load * inputs.variability)
        concentration = Index("concentration", inputs.base_level + load)
        return ConcentrationModel.Outputs(concentration=concentration)


variability  = DistributionIndex("variability", stats.norm, {"loc": 1.0, "scale": 0.2})
base_level   = Index("base_level", 15.0)
traffic_load = Index("traffic_load", None)    # value=None → abstract index

model = ConcentrationModel(inputs=ConcentrationModel.Inputs(
    variability=variability,
    base_level=base_level,
    traffic_load=traffic_load,
))

traffic_grid = np.array([50.0, 100.0, 200.0])


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — Scenario: overrides and parameter_axes
# ---------------------------------------------------------------------------


def _demo_scenario() -> None:
    """Scenario construction: overrides and parameter_axes."""
    # Override base_level for a high-emission what-if
    scenario_hi = Scenario(
        model,
        overrides={model.inputs.base_level: 25.0},
    )
    assert scenario_hi.overrides[model.inputs.base_level] == 25.0

    # Mark traffic_load as a PARAMETER axis (excluded from ensemble sampling)
    scenario_param = Scenario(
        model,
        parameter_axes=[model.inputs.traffic_load],
    )
    abstract = list(scenario_param.abstract_indexes())
    assert not _is_in(model.inputs.traffic_load, abstract)
    assert _is_in(model.inputs.variability, abstract)

    # Combine override and parameter_axes
    scenario_combined = Scenario(
        model,
        overrides={model.inputs.base_level: 20.0},
        parameter_axes=[model.inputs.traffic_load],
    )
    assert scenario_combined.overrides[model.inputs.base_level] == 20.0


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — Basic evaluation
# ---------------------------------------------------------------------------


def _demo_basic_eval() -> None:
    """Direct Evaluation.evaluate() call with a parameter sweep."""
    scenario = Scenario(model, parameter_axes=[model.inputs.traffic_load])
    ens = DistributionEnsemble(scenario, 100)

    result = Evaluation(scenario).evaluate(
        ensemble=ens,
        parameters={model.inputs.traffic_load: traffic_grid},
    )

    mean_conc = result.expected_value(model.outputs.concentration)
    assert mean_conc.shape == (3,)          # one value per traffic level
    assert np.all(mean_conc > 0)


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — EvaluationHandle: incremental ensemble extension
# ---------------------------------------------------------------------------


def _demo_handle() -> None:
    """EvaluationHandle: initial evaluation + ensemble extension + parameter extension."""
    scenario = Scenario(model, parameter_axes=[model.inputs.traffic_load])
    ev = Evaluation(scenario)

    # Build plan and run initial ensemble of 100 samples
    handle = EvaluationHandle.evaluate(
        ev, 100,
        parameters={model.inputs.traffic_load: traffic_grid},
    )
    assert handle.result.expected_value(model.outputs.concentration).shape == (3,)

    # Extend: draw 50 more Monte Carlo samples (same parameter grid)
    handle.extend(50)
    assert handle.result.expected_value(model.outputs.concentration).shape == (3,)

    # Extend parameter grid: two more traffic levels, same ensemble
    handle.extend(
        extra_parameters={model.inputs.traffic_load: np.array([300.0, 400.0])},
    )
    assert handle.result.expected_value(model.outputs.concentration).shape == (5,)

    # Combined: grow both ensemble and parameter grid in one call
    handle.extend(
        20,
        extra_parameters={model.inputs.traffic_load: np.array([500.0])},
    )
    assert handle.result.expected_value(model.outputs.concentration).shape == (6,)


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — AsyncEvaluationHandle
# ---------------------------------------------------------------------------


def _demo_async_handle() -> None:
    """AsyncEvaluationHandle: non-blocking evaluation via a background thread."""
    scenario = Scenario(model, parameter_axes=[model.inputs.traffic_load])
    ev = Evaluation(scenario)

    async_handle = AsyncEvaluationHandle.evaluate(
        ev, 100,
        parameters={model.inputs.traffic_load: traffic_grid},
    )

    poll_state = async_handle.poll()    # (True, result) if done; (False, None) still running
    assert isinstance(poll_state, tuple)
    result = async_handle.get()         # blocks until complete
    assert result.expected_value(model.outputs.concentration).shape == (3,)


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — EvaluationPlan and scoped abstract indexes
# ---------------------------------------------------------------------------


def _demo_plan() -> None:
    """EvaluationPlan: build and inspect the execution DAG."""
    scenario = Scenario(model, parameter_axes=[model.inputs.traffic_load])
    ev = Evaluation(scenario)

    plan = ev.build_plan(strategy="monolithic")
    assert isinstance(plan, EvaluationPlan)
    assert len(plan.regions) > 0

    # scoped_abstract_indexes groups abstract indexes by region guard chain.
    # For a monolithic plan with no ModelVariant, one entry: guards=() → {variability}
    scoped = plan.scoped_abstract_indexes(scenario)
    all_abstract = {idx for idxs in scoped.values() for idx in idxs}
    assert _is_in(model.inputs.variability, all_abstract)
    assert not _is_in(model.inputs.traffic_load, all_abstract)


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — FrozenEnsemble and BatchDrawable
# ---------------------------------------------------------------------------


def _demo_frozen_ensemble() -> None:
    """FrozenEnsemble: immutable sample snapshot produced by BatchDrawable.draw_batch()."""
    scenario = Scenario(model, parameter_axes=[model.inputs.traffic_load])
    rng = np.random.default_rng(0)

    # DistributionEnsemble implements BatchDrawable
    recipe: BatchDrawable = DistributionEnsemble(scenario, 100)

    frozen_a: FrozenEnsemble = recipe.draw_batch(100, rng)
    frozen_b: FrozenEnsemble = recipe.draw_batch(50, rng)

    # FrozenEnsemble can be merged along its single ENSEMBLE axis
    merged = frozen_a.concat(frozen_b)
    assert merged.ensemble_axes[0].name == frozen_a.ensemble_axes[0].name

    # FrozenEnsemble cannot draw new samples — it is immutable
    try:
        frozen_a.draw_batch(10, rng)
        assert False, "should have raised TypeError"
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — ModelOutput: domain-specific evaluation output
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ConcentrationOutput(ModelOutput):
    """Domain output: expected concentration (scalar or 1-D array)."""

    mean_conc: np.ndarray

    def __post_init__(self) -> None:
        super().__init__()


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — ModelEvaluator: application-level evaluation API
# ---------------------------------------------------------------------------

# For the ModelEvaluator examples, traffic_load is supplied as a concrete
# override so the default evaluate() template works without a custom override.
scenario_fixed = Scenario(model, overrides={
    model.inputs.base_level:   20.0,
    model.inputs.traffic_load: 100.0,
})


class ConcentrationEvaluator(ModelEvaluator[ConcentrationModel, ConcentrationOutput]):
    """Application-level evaluator for ConcentrationModel."""

    def post_process(
        self,
        scenario: Scenario,
        result: EvaluationResult,
    ) -> ConcentrationOutput:
        del scenario  # unused; concentration depends only on result
        return ConcentrationOutput(
            mean_conc=result.expected_value(self._model.outputs.concentration),
        )

    def input_schema(self) -> dict:
        return {
            "base_level":   {"type": "scalar", "default": 15.0, "unit": "µg/m³"},
            "traffic_load": {"type": "scalar", "default": 100.0, "unit": "veh/h"},
        }


def _demo_model_evaluator() -> None:
    """ModelEvaluator lifecycle: evaluate → save → load → resume."""
    evaluator = ConcentrationEvaluator(model)
    config = EvaluationConfig(ensemble_size=200)

    output = evaluator.evaluate(scenario_fixed, config)
    assert isinstance(output, ConcentrationOutput)
    assert output.is_resumable
    assert output.mean_conc.ndim == 0 or output.mean_conc.shape == ()

    # Save and reload via to_dict / from_dict
    data = output.to_dict()
    output2 = ConcentrationOutput.from_dict(data)
    assert output2.is_resumable

    # Resume from saved output — extend the ensemble in a new session
    handle = evaluator.resume(scenario_fixed, output2, config)
    handle.extend(100)
    extended_mean = handle.result.expected_value(model.outputs.concentration)
    assert extended_mean.shape == ()


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — ModelRunHandle: async application-level evaluation
# ---------------------------------------------------------------------------


def _demo_run_async() -> None:
    """ModelRunHandle: non-blocking application-level evaluation."""
    evaluator = ConcentrationEvaluator(model)
    config = EvaluationConfig(ensemble_size=200)

    run_handle: ModelRunHandle[ConcentrationOutput] = evaluator.run_async(scenario_fixed, config)
    poll_state = run_handle.poll()      # (True, output) if done; (False, None) still running
    assert isinstance(poll_state, tuple)
    output = run_handle.get()           # blocks until complete
    assert isinstance(output, ConcentrationOutput)


# ---------------------------------------------------------------------------
# dd-cdt-simulation.md — IncompatibleResultError
# ---------------------------------------------------------------------------


def _demo_incompatible() -> None:
    """IncompatibleResultError is raised when a non-resumable output is resumed."""
    evaluator = ConcentrationEvaluator(model)
    config = EvaluationConfig(ensemble_size=200)

    output = evaluator.evaluate(scenario_fixed, config)
    data = output.to_dict()

    # Strip the resume payload to simulate an incompatible saved file
    data.pop("_resume", None)
    output_stripped = ConcentrationOutput.from_dict(data)
    assert not output_stripped.is_resumable

    try:
        evaluator.resume(scenario_fixed, output_stripped, config)
        assert False, "should have raised IncompatibleResultError"
    except IncompatibleResultError:
        pass


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------

_demo_scenario()
_demo_basic_eval()
_demo_handle()
_demo_async_handle()
_demo_plan()
_demo_frozen_ensemble()
_demo_model_evaluator()
_demo_run_async()
_demo_incompatible()

print("doc_simulation.py: all snippets OK")
