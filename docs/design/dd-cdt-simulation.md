<!-- SPDX-License-Identifier: Apache-2.0 -->

# Scenario Evaluation

|              | Document data                                  |
|--------------| ---------------------------------------------- |
| Author       | [@pistore](https://github.com/pistore)         |
| Last-Updated | 2026-06-21                                     |
| Status       | Draft                                          |
| Approved-By  | N/A                                            |

This guide covers the simulation layer of the civic-digital-twins framework: how to wrap a `Model`
in a `Scenario`, run Monte Carlo evaluations, grow results incrementally with `EvaluationHandle`
or the higher-level `IncrementalRun`, structure execution with `EvaluationPlan`, and build
application-level evaluators with `ModelEvaluator` and `ModelOutput`.

See [dd-cdt-model.md](dd-cdt-model.md) for the `Model`, `ModelVariant`, and index reference.
See [dd-cdt-modularity.md](dd-cdt-modularity.md) for the `@define` / `compute()` pattern.

---

## TL;DR

A `Scenario` wraps a model with optional value overrides and declares which abstract indexes are
PARAMETER axes (deterministic sweep) rather than ENSEMBLE axes (Monte Carlo).  `Evaluation` runs
the model; `EvaluationHandle` lets you grow the ensemble incrementally.

```python
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
```

Run a basic Monte Carlo sweep over three traffic levels:

```python
traffic_grid = np.array([50.0, 100.0, 200.0])
scenario = Scenario(model, parameter_axes=[model.inputs.traffic_load])
ens = DistributionEnsemble(scenario, 100)

result = Evaluation(scenario).evaluate(
    ensemble=ens,
    parameters={model.inputs.traffic_load: traffic_grid},
)

mean_conc = result.expected_value(model.outputs.concentration)
```

`mean_conc` has shape `(3,)` — one expected concentration per traffic level, marginalising
over the 100 Monte Carlo samples of `variability`.

---

## Scenario

A `Scenario` is the canonical first argument to `Evaluation` and all ensemble classes.  It wraps
a model with two optional modifications:

* **Overrides** — shadow one or more index values for a what-if run.
* **Parameter axes** — declare abstract indexes that will be supplied externally as a deterministic
  sweep rather than sampled by the ensemble.

### Overrides

```python
scenario_hi = Scenario(
    model,
    overrides={model.inputs.base_level: 25.0},
)
```

Override compatibility depends on index kind:

```text
Index type                   │ float  str  ndarray  Distribution  dict[str,float]  list[str]
─────────────────────────────┼──────────────────────────────────────────────────────────────
Index                        │  ✓     ✗      ✗         ✗              ✗               ✗
TimeseriesIndex              │  ✗     ✗      ✓(1-D)    ✗              ✗               ✗
ConstIndex / ConstTimeseries │  ✗     ✗      ✗         ✗              ✗               ✗
DistributionIndex            │  ✗     ✗      ✗         ✓              ✗               ✗
CategoricalIndex             │  ✗     ✓*     ✗         ✗              ✓**             ✓***
ConditionalCategoricalIndex  │  ✗     ✓*     ✗         ✗              ✗               ✗

* str must be in idx.support
** dict keys must be a non-empty subset of idx.support, positive probs summing to 1.0
*** list must be a non-empty subset of idx.support; model probabilities are renormalised
```

`ConstIndex` and `ConstTimeseriesIndex` cannot be overridden.

### Parameter axes

Mark abstract indexes that will be provided as a PARAMETER sweep — they are excluded from
`abstract_indexes()` and therefore not sampled by `DistributionEnsemble`:

```python
scenario_param = Scenario(
    model,
    parameter_axes=[model.inputs.traffic_load],
)
```

Supply the PARAMETER axis values explicitly when calling `evaluate()`:

```python
result = Evaluation(scenario).evaluate(
    ensemble=ens,
    parameters={model.inputs.traffic_load: traffic_grid},
)
```

The result has an additional PARAMETER dimension: `result.expected_value(idx)` returns a
`(len(traffic_grid),)` array, one entry per traffic level.

### Combining overrides and parameter axes

```python
scenario_combined = Scenario(
    model,
    overrides={model.inputs.base_level: 20.0},
    parameter_axes=[model.inputs.traffic_load],
)
```

---

## EvaluationHandle: Incremental Evaluation

`EvaluationHandle` is the go-to API when you want to build up a result across multiple
Monte Carlo batches — either to check convergence, to add new parameter values in a
later session, or to avoid allocating a large ensemble up front.

Obtain a handle via the `EvaluationHandle.evaluate()` factory, which builds the plan and
runs the initial ensemble in one call:

```python
scenario = Scenario(model, parameter_axes=[model.inputs.traffic_load])
ev = Evaluation(scenario)

handle = EvaluationHandle.evaluate(
    ev, 100,
    parameters={model.inputs.traffic_load: traffic_grid},
)
```

`handle.result` is the current accumulated `EvaluationResult`.

### Growing the ensemble

Draw additional Monte Carlo samples and merge them with the accumulated result:

```python
handle.extend(50)
```

### Extending the parameter grid

Re-run the **same** frozen sample draws at new PARAMETER values and append the slices:

```python
handle.extend(
    extra_parameters={model.inputs.traffic_load: np.array([300.0, 400.0])},
)
```

After this call `handle.result.expected_value(model.outputs.concentration)` has shape `(5,)`.

### Combined extension

Grow both the ensemble and the parameter grid in a single call:

```python
handle.extend(
    20,
    extra_parameters={model.inputs.traffic_load: np.array([500.0])},
)
```

---

## AsyncEvaluationHandle

`AsyncEvaluationHandle` runs the engine evaluation in a background thread.  Construction and the
API otherwise mirror `EvaluationHandle.evaluate()`:

```python
async_handle = AsyncEvaluationHandle.evaluate(
    ev, 100,
    parameters={model.inputs.traffic_load: traffic_grid},
)

poll_state = async_handle.poll()    # (True, result) if done; (False, None) still running
result = async_handle.get()         # blocks until complete
```

`.poll()` returns `(True, result)` if the evaluation has finished, or `(False, None)` if it is
still running.  `.get()` always blocks until the result is available.

---

## EvaluationPlan

An `EvaluationPlan` captures the compiled computation structure that `EvaluationHandle` reuses
across all `.extend()` calls.  Build one explicitly when you need to inspect it or pass it to
`execute_plan()` directly:

```python
plan = ev.build_plan(strategy="monolithic")
scoped = plan.scoped_abstract_indexes(scenario)
```

`build_plan()` accepts two strategies:

* `"monolithic"` (default) — a single region containing all nodes.
* `"regional"` — splits at `ModelVariant` selector boundaries; enables per-scope sampling in
  `DistributionEnsemble`, aligning the Monte Carlo budget with branch probabilities.

`scoped_abstract_indexes(scenario)` returns a dict mapping each region's guard chain to the
abstract indexes scoped to that region.  For a monolithic plan with no `ModelVariant`, the
result has one entry: `{(): {variability_index}}`.

---

## FrozenEnsemble and BatchDrawable

> **Advanced.** This section covers low-level ensemble mechanics.  Skip it unless you need
> fine-grained control over sample draws or are building custom evaluation pipelines.

`BatchDrawable` is the protocol implemented by live ensemble recipes (`DistributionEnsemble`,
`CrossProductEnsemble`, `PartitionedEnsemble`).  Its single method `draw_batch(size, rng)` returns
a `FrozenEnsemble` — an immutable snapshot of pre-drawn samples that can be replayed without
advancing the RNG.

```python
recipe: BatchDrawable = DistributionEnsemble(scenario, 100)

frozen_a: FrozenEnsemble = recipe.draw_batch(100, rng)
frozen_b: FrozenEnsemble = recipe.draw_batch(50, rng)

merged = frozen_a.concat(frozen_b)
```

`FrozenEnsemble` satisfies the `AxisEnsemble` protocol and can be passed directly to
`execute_plan()`.  It cannot draw new samples — calling `draw_batch()` on it raises `TypeError`:

```python
try:
    frozen_a.draw_batch(10, rng)
    assert False, "should have raised TypeError"
except TypeError:
    pass
```

`EvaluationHandle` stores the frozen snapshot internally so that PARAMETER extension (`.extend(extra_parameters=…)`) can replay the same scenarios without consuming more RNG state — the common-random-numbers guarantee.

---

## ModelOutput and ModelEvaluator

`ModelOutput` and `ModelEvaluator` are abstract base classes that define a stable interface for
scenario evaluation at the application layer.  A domain package subclasses both and wires together
model construction, ensemble setup, post-processing, and persistence.

### ModelOutput

Subclass `ModelOutput` as a `@dataclass` and call `super().__init__()` from `__post_init__`:

```python
@dataclasses.dataclass
class ConcentrationOutput(ModelOutput):
    """Domain output: expected concentration (scalar or 1-D array)."""

    mean_conc: np.ndarray

    def __post_init__(self) -> None:
        super().__init__()
```

The base class provides two serialisation methods:

* `to_dict()` — summary layer (always) + resume payload (when `is_resumable`).
* `to_snapshot()` — summary layer only; suitable for API responses.
* `from_dict(data)` — reconstructs an instance; restores `is_resumable` when the payload is present.

### ModelEvaluator

Subclass `ModelEvaluator[ModelT, OutputT]` and implement `post_process()` and `input_schema()`.
The base class template for `evaluate()` calls `make_ensemble()`, runs `Evaluation`, and delegates
to `post_process()`.  The returned output is **not resumable** — `is_resumable` is always `False`
from `evaluate()`.  Use `start()` (below) for resumable incremental runs:

```python
class ConcentrationEvaluator(ModelEvaluator[ConcentrationModel, ConcentrationOutput]):

    def post_process(
        self,
        scenario: Scenario,
        result: EvaluationResult,
    ) -> ConcentrationOutput:
        return ConcentrationOutput(
            mean_conc=result.expected_value(self._model.outputs.concentration),
        )

    def input_schema(self) -> dict:
        return {
            "base_level":   {"type": "scalar", "default": 15.0, "unit": "µg/m³"},
            "traffic_load": {"type": "scalar", "default": 100.0, "unit": "veh/h"},
        }
```

Override `evaluate()` when the model requires a parameter grid or a non-default ensemble type.
Override `make_ensemble()` for a different ensemble type without changing the rest of the template.

### Evaluation lifecycle

**One-shot evaluation** (no resume payload):

```python
evaluator = ConcentrationEvaluator(model)
config = EvaluationConfig(ensemble_size=200)

output = evaluator.evaluate(scenario_fixed, config)
assert not output.is_resumable
```

**Incremental evaluation** with `start()` / `resume()`:

```python
# Initial run — draws config.ensemble_size samples
run = evaluator.start(scenario_fixed, config)

# Optionally grow the ensemble before snapshotting
run.extend(100)          # draw 100 more samples (explicit)
run.extend()             # draw config.ensemble_size more (default)

# Non-resumable snapshot — for display / analysis only
output = run.snapshot()
assert not output.is_resumable

# Resumable snapshot — embeds the full result for later resume
output = run.snapshot(resumable=True)
assert output.is_resumable

# Save and reload via to_dict / from_dict
data = output.to_dict()
output2 = ConcentrationOutput.from_dict(data)

# Resume from saved output — picks up where the previous run left off
run2 = evaluator.resume(scenario_fixed, output2, config)
run2.extend()            # draw more samples; config.ensemble_size is the default
output3 = run2.snapshot(resumable=True)
```

`evaluator.start()` returns an `IncrementalRun` seeded with the first batch.
`evaluator.resume()` returns an `IncrementalRun` seeded with the saved result.
`run.snapshot(resumable=True)` attaches the full resume payload so the run can be
continued in a later session.

---

## ModelRunHandle: Async Application Evaluation

`evaluator.run_async()` submits the evaluation to a background thread and immediately returns a
`ModelRunHandle`.  Like `evaluate()`, it is a **one-shot** path — the returned output is not
resumable (`is_resumable` is `False`).

```python
run_handle: ModelRunHandle[ConcentrationOutput] = evaluator.run_async(scenario_fixed, config)
poll_state = run_handle.poll()      # (True, output) if done; (False, None) still running
output = run_handle.get()           # blocks until complete
assert not output.is_resumable
```

---

## IncompatibleResultError

`evaluator.resume()` raises `IncompatibleResultError` when the saved output has no resume payload
(e.g. it was produced by an incompatible library version or saved with `to_snapshot()` only):

```python
try:
    evaluator.resume(scenario_fixed, output_stripped, config)
    assert False, "should have raised IncompatibleResultError"
except IncompatibleResultError:
    pass
```

The summary layer (KPIs, timeseries, parameter grids) is still fully readable via the output
fields; only ensemble extension is blocked.

---

## API Reference

### `Scenario`

```python
class Scenario:
    def __init__(
        self,
        model: Model | ModelVariant,
        overrides: dict[GenericIndex, DomainValue] = {},
        parameter_axes: list[GenericIndex] = [],
    ): ...

    def abstract_indexes(self) -> frozenset[GenericIndex]: ...
    def base_substitutions(self) -> dict[graph.Node, Any]: ...
    def effective_distribution(self, idx: GenericIndex) -> Distribution | None: ...
    def effective_outcomes(self, idx: CategoricalIndex) -> dict[str, float]: ...
```

### `EvaluationHandle`

```python
class EvaluationHandle:
    @classmethod
    def evaluate(
        cls,
        evaluation: Evaluation,
        initial_ensemble_size: int,
        nodes_of_interest: list[GenericIndex] | None = None,
        *,
        ensemble_recipe: BatchDrawable | None = None,
        parameters: dict[GenericIndex, Any] | None = None,
        parameter_axes: dict[str, np.ndarray] | None = None,
        strategy: str = "monolithic",
        rng: np.random.Generator | None = None,
    ) -> EvaluationHandle: ...

    @property
    def result(self) -> EvaluationResult: ...

    def extend(
        self,
        ensemble_size: int = 0,
        *,
        extra_ensemble: dict[str, int] | None = None,
        extra_parameters: dict[GenericIndex, np.ndarray] | None = None,
    ) -> EvaluationResult: ...
```

### `FrozenEnsemble`

```python
class FrozenEnsemble:
    def draw_batch(self, size: int, rng: np.random.Generator, *, axis: str | None = None) -> FrozenEnsemble: ...
    def concat(self, other: FrozenEnsemble) -> FrozenEnsemble: ...
    def concat_along(self, axis_name: str, other: FrozenEnsemble) -> FrozenEnsemble: ...
    def with_replaced_axis(self, axis_name: str, other: FrozenEnsemble) -> FrozenEnsemble: ...
```

### `ModelOutput`

```python
class ModelOutput(ABC):
    def to_dict(self) -> dict[str, Any]: ...
    def to_snapshot(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...

    @property
    def is_resumable(self) -> bool: ...
```

### `IncrementalRun`

```python
class IncrementalRun(Generic[OutputT]):
    @property
    def result(self) -> EvaluationResult: ...

    def extend(self, n: int | None = None) -> None: ...
    # n=None → uses config.ensemble_size passed to start() / resume()

    def snapshot(self, *, resumable: bool = False) -> OutputT: ...
    # resumable=False → is_resumable is False on the returned output
    # resumable=True  → attaches full resume payload; is_resumable is True
```

### `ModelEvaluator`

```python
class ModelEvaluator(ABC, Generic[ModelT, OutputT]):
    @abstractmethod
    def input_schema(self) -> dict[str, dict[str, Any]]: ...

    def post_process(self, scenario: Scenario, result: EvaluationResult) -> OutputT: ...
    def make_ensemble(self, scenario: Scenario, config: EvaluationConfig) -> Any: ...

    # One-shot (not resumable):
    def evaluate(self, scenario: Scenario, config: EvaluationConfig) -> OutputT: ...
    def run_async(self, scenario: Scenario, config: EvaluationConfig) -> ModelRunHandle[OutputT]: ...

    # Incremental (resumable via snapshot(resumable=True)):
    def start(self, scenario: Scenario, config: EvaluationConfig) -> IncrementalRun[OutputT]: ...
    def resume(self, scenario: Scenario, output: OutputT, config: EvaluationConfig) -> IncrementalRun[OutputT]: ...

    # Advanced: encode result into output as resume payload
    def attach_resume(self, output: ModelOutput, result: EvaluationResult) -> None: ...
```

### `EvaluationConfig`

```python
# === dataclass — controls Monte Carlo budget for ModelEvaluator ===
@dataclasses.dataclass
class EvaluationConfig:
    ensemble_size: int
```
