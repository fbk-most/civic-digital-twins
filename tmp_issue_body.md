### Background

The current `Model` subclass API requires boilerplate that repeats information already expressed in `@inputs` / `@outputs` / `@expose`. A typical leaf model follows this pattern:

```python
class ParkingModel(Model):

    @inputs
    class Inputs:
        pv_tourists: ConditionalDistributionIndex
        i_c_parking: DistributionIndex

    @outputs
    class Outputs:
        i_u_parking: Index

    def __init__(
        self,
        pv_tourists: ConditionalDistributionIndex,   # ← duplicates Inputs
        i_c_parking: DistributionIndex,              # ← duplicates Inputs
    ) -> None:
        Inputs  = ParkingModel.Inputs                # ← alias boilerplate
        Outputs = ParkingModel.Outputs               # ← alias boilerplate

        inputs = Inputs(                             # ← pure re-packing
            pv_tourists=pv_tourists,
            i_c_parking=i_c_parking,
        )

        i_u_parking = Index("parking usage", inputs.pv_tourists * ...)

        super().__init__(                            # ← wiring boilerplate
            "Parking",
            inputs=inputs,
            outputs=Outputs(i_u_parking=i_u_parking),
        )
```

The domain logic (the `Index(...)` formula lines) is the only part that carries real information. Everything else is mechanical repetition.

---

### Why a pure dataclass approach does not work

A natural instinct is to declare index objects at class level like dataclass fields. This is not viable because `Index` identity is structural: each `Index` carries a unique `graph.Node` (identity-hashed) that the entire evaluation pipeline — `Scenario.overrides`, `EvaluationResult[idx]`, deduplication — tracks by object identity. Class-level index objects would be shared across all instances, making independent model instances impossible.

The correct analogy is not `@dataclass` (where field annotations are types, not values) but the existing `@inputs` / `@outputs` inner classes, which already sit at class level as pure type declarations. Live index objects must be created per-instance at construction time.

---

### Proposed design

Introduce a `@model` decorator and a `compute()` method that together replace the `__init__` boilerplate:

```python
@model("Parking")
class ParkingModel(Model):

    @inputs
    class Inputs:
        pv_tourists: ConditionalDistributionIndex
        i_c_parking: DistributionIndex

    @outputs
    class Outputs:
        i_u_parking: Index

    def compute(self, inp: Inputs) -> Outputs:
        i_u_parking = Index("parking usage", inp.pv_tourists * ...)
        return ParkingModel.Outputs(i_u_parking=i_u_parking)
```

`@model` is responsible for:

1. Reading the `Inputs` fields and generating `__init__(self, inp: Inputs)`.
2. Calling `self.compute(inp)` to obtain the `Outputs` (and optionally `Expose`).
3. Invoking `super().__init__(name, inputs=inp, outputs=result)` automatically.

#### Naming rationale

- **`compute`** over `forward` (PyTorch) or `call` (Keras): the method body constructs formula-graph nodes, not a numeric forward pass. `compute` is domain-neutral and reads naturally as "here is how outputs are computed from inputs."
- **Non-dunder**: `compute` is framework API, not Python data-model protocol. Dunder methods signal Python language machinery (`__iter__`, `__len__`, …); a domain hook should be a plain name, following the same reasoning PyTorch used when naming `forward` instead of `__forward__`.

#### `Expose` and `Functions`

```python
def compute(self, inp: Inputs) -> tuple[Outputs, Expose]:
    ...
    return ParkingModel.Outputs(...), ParkingModel.Expose(...)
```

`Functions` is passed as a keyword argument, matching the existing `@functions` convention:

```python
@model("Traffic")
class TrafficModel(Model):

    @inputs
    class Inputs: ...

    @functions
    class Functions:
        ts_solve: Functor

    @outputs
    class Outputs: ...

    def compute(self, inp: Inputs, *, fns: Functions) -> Outputs:
        ...
```

#### Models that cannot use `compute`

Root / composite models (e.g. `MolvenoModel`) that assign sub-model attributes after `super().__init__()` cannot express this cleanly inside a single `compute` return value. These models remain on the `__init__`-based syntax indefinitely (see deprecation below).

---

### Backwards compatibility

`@model` is purely **opt-in**. A class without the decorator behaves exactly as today. The two forms coexist freely, class by class, in the same codebase.

The decorator must raise `TypeError` at class-definition time if a class defines **both** `compute` and `__init__`, to prevent silent ambiguity.

---

### Deprecation path

`Model.__init_subclass__` can detect the old syntax cleanly, because Python applies class decorators **after** `__init_subclass__` fires. This means that when `__init_subclass__` inspects a new subclass:

- `@model` + `compute` → `__init__` is **not yet** in `cls.__dict__` (decorator hasn't run)
- Old syntax → `__init__` **is** in `cls.__dict__` (user wrote it)

The detection is unambiguous with zero false positives on the new syntax:

```python
class Model:
    def __init_subclass__(cls, *, legacy: bool = False, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__ and not legacy:
            warnings.warn(
                f"{cls.__name__} defines __init__ directly. "
                "Use @model with compute() instead. "
                "Pass legacy=True to suppress this warning.",
                DeprecationWarning,
                stacklevel=2,
            )
```

Models that genuinely cannot migrate use the opt-out:

```python
class MolvenoModel(Model, legacy=True):
    def __init__(self) -> None:
        ...
```

`legacy=True` doubles as a grep-able migration marker. The intended arc:

| Phase | Action |
|---|---|
| This issue | Ship `@model` + `compute`, no warnings yet |
| Next release | Add `__init_subclass__` check → `DeprecationWarning`; migrate all leaf models; tag composite models with `legacy=True` |
| Later | Escalate to `FutureWarning`; work down `legacy=True` list |
| Final | Remove `legacy=` and old `__init__` path |

---

### Implementation notes

- The `_check_inputs_contract` mechanism in `Model.__init__` uses `inspect.currentframe().f_back` to inspect the caller's locals. With a `@model`-generated `__init__`, that frame is framework code. The decorator must perform the contract check itself, or rely on the validation already built into `@inputs` at `Inputs(...)` construction time.
- `@model` should be exported from `civic_digital_twins.dt_model` alongside `@inputs`, `@outputs`, `@expose`, `@functions`.
- The generated `__init__` will not be visible to pyright's call-site type inference. This is a known limitation. Call-site checking for the `compute` method signature itself (argument types, return type) remains fully enforced.

---

### Open questions

- Should `compute` be enforced as an abstract method on classes decorated with `@model`, or left as a plain override?
- Should `legacy=True` be a permanent escape hatch or sunset with a target version?
- Is a `post_compute` hook worth adding for composite models, or is `legacy=True` sufficient for the foreseeable future?
