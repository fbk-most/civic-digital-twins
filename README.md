# Civic-Digital-Twins Modeling Framework

[![Build Status](https://github.com/fbk-most/civic-digital-twins/actions/workflows/ci-release.yml/badge.svg)](https://github.com/fbk-most/civic-digital-twins/actions) [![codecov](https://codecov.io/gh/fbk-most/civic-digital-twins/branch/main/graph/badge.svg)](https://codecov.io/gh/fbk-most/civic-digital-twins) [![PyPI version](https://img.shields.io/pypi/v/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/) [![Python Versions](https://img.shields.io/pypi/pyversions/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/) [![License](https://img.shields.io/pypi/l/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/)

This repository contains a Python package implementing a Civic-Digital-Twins
modeling framework. The framework is designed to support defining digital
twins models and evaluating them in simulated environments with varying
contextual conditions. We develop this package at [@fbk-most](
https://github.com/fbk-most), a research unit at [Fondazione Bruno Kessler](
https://www.fbk.eu/en/).

*Note: this package is currently in an early development stage.*

## Conceptual Overview

The framework is organised in three layers.

### Engine layer

The engine (`civic_digital_twins.dt_model.engine`) is an embedded DSL
compiler.  The programmer builds a *computation graph* (DAG) by composing
typed nodes — constants, placeholders, and operations — using ordinary Python
expressions.  The graph is then linearised by topological sorting and
evaluated by a NumPy-based interpreter that maps each node to the
corresponding `numpy` operation.

```python
import numpy as np

from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor

a = graph.placeholder("a")
b = graph.placeholder("b")
c = a * 2 + b

state = executor.State(values={a: np.asarray(3.0), b: np.asarray(1.0)})
executor.evaluate_nodes(state, *linearize.forest(c))
print(state.get_node_value(c))  # 7.0
```

See [docs/design/dd-cdt-engine.md](docs/design/dd-cdt-engine.md) for a
full description of the engine.

### Model / simulation layer

The model layer (`civic_digital_twins.dt_model`) provides higher-level
abstractions built on top of the engine:

- **`Index`** / **`TimeseriesIndex`** — named wrappers around graph nodes.
  An index can be a constant, a distribution (sampled at evaluation time),
  or a formula.
- **`Model`** — a typed computation unit.  Use the `@define` decorator to
  declare a `Model` subclass via a `compute()` method; `@inputs`, `@outputs`,
  and `@expose` decorators mark the contractual interface.  Sub-models are
  wired via constructor arguments in `compute()`, producing a composable
  pipeline.
- **`ModelVariant`** — selects among pre-constructed `Model` implementations
  sharing the same I/O contract.  The active variant is resolved by a string
  key (static) or a `CategoricalIndex`/graph node (runtime dispatch).
- **`Scenario`** — wraps a model with optional value overrides and parameter
  axes; the canonical first argument to `Evaluation` and all ensemble classes.
- **`Evaluation`** — evaluates a model over a sequence of *weighted
  scenarios*, each of which maps every abstract index to a concrete value.
- **`Ensemble`** / **`WeightedScenario`** — a protocol and type alias that
  define the scenario contract consumed by `Evaluation`.

```python
from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index, Model, define, inputs, outputs

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
```

See [docs/design/dd-cdt-model.md](docs/design/dd-cdt-model.md) for the full
reference: index types, `Model` API, `ModelVariant`, `Evaluation`, and the
domain modeling pattern.

### Usage patterns

The `examples/` directory contains two illustrative examples, distinguished
by whether the model has external categorical context.  Both use the
`@define`/`compute()` API (`@inputs`, `@outputs`, `@expose`, `ModelVariant`) —
see [docs/design/dd-cdt-modularity.md](docs/design/dd-cdt-modularity.md).

**Direct pattern** (`examples/mobility_bologna/`) — no context variables.
Uncertainty enters only through `DistributionIndex` parameters.
`DistributionEnsemble` draws *S* Monte-Carlo samples to produce weighted
scenarios; `Evaluation.evaluate()` runs the engine and returns an
`EvaluationResult`.

**Context-variable pattern** (`examples/overtourism_molveno/`) — the model
has categorical scenario factors outside the modeller's control (season,
weather, …), expressed as `CategoricalIndex`, and quantities whose
distribution depends on context, expressed as
`ConditionalDistributionIndex`.  `CrossProductEnsemble` enumerates the
context combinations into weighted scenarios; presence quantities are
swept over a multi-dimensional grid via
`Evaluation.evaluate(parameters={pv: array, …})`.

## Installation

The package name is `civic-digital-twins` on [PyPi](
https://pypi.org/project/civic-digital-twins/). Install
using `pip`:

```bash
pip install civic-digital-twins
```

or, using `uv`:

```bash
uv add civic-digital-twins
```

The main package name is `civic_digital_twins`:

```python
import civic_digital_twins
```

or

```python
from civic_digital_twins import dt_model
```

## Minimum Python Version

Python 3.12. Tested against Python 3.12, 3.13, and 3.14.

## API Stability Guarantees

The package is currently in an early development stage. We do not
anticipate breaking APIs without a good reason to do so, yet, breaking
changes may occur from time to time. We generally expect subpackages
within the top-level package to change more frequently.

## Development Setup

We use [uv](https://astral.sh/uv) for managing the development environment.

To get started, run:

```bash
git clone https://github.com/fbk-most/civic-digital-twins
cd civic-digital-twins
uv venv
source .venv/bin/activate
uv sync --dev
```

We use [pytest](https://docs.pytest.org/en/stable/) for testing. To run
tests use this command (from inside the virtual environment):

```bash
pytest
```

Pull requests are automatically tested using GitHub Actions. PRs targeting
`dev` run the fast [`CI (dev)`](.github/workflows/ci-dev.yml) workflow
(format, lint, type-check, tests on Python 3.12). PRs targeting `main` run
the full [`CI (release)`](.github/workflows/ci-release.yml) workflow (all
Python versions, doc examples, domain examples, SPDX check, dependency
audit, and build smoke test).

## Updating Dependencies

```bash
uv self update
uv sync --upgrade
```

## Development model

This project follows a simplified GitHub Flow with an explicit `dev` branch:

```
feature/* ──PR─▶ dev ──PR─▶ main ──tag─▶ PyPI
           (CI dev)   (CI release)  (publish)
```

- **Feature work** happens on short-lived branches cut from `dev`.
- **`dev`** is the integration branch. It always carries a `+dev` version
  marker (e.g. `0.11.0+dev`).
- **`main`** contains only released commits. Merging `dev` into `main` is
  always immediately followed by a version tag and a PyPI release.

## Releasing

### Step 1 — Merging a feature PR into `dev`

Before opening the PR, verify locally:

- Tests pass: `uv run pytest`
- Format, lint, and type-check pass: `uv run ruff format --check .`,
  `uv run ruff check .`, `uv run pyright`
- `CHANGELOG.md` `[Unreleased]` section updated (Added / Changed / Removed /
  Fixed; breaking changes flagged).
- Design docs (`docs/design/`) updated if public API or architecture changed.
- SPDX licence header present on any new `.py` or `.md` file.

Open the PR targeting `dev`. The `CI (dev)` workflow runs automatically; merge
once it is green.

### Step 2 — Preparing a release (promoting `dev` to `main`)

Perform the following steps on the `dev` branch before opening the
`dev → main` PR:

1. Set the final version in `pyproject.toml` (remove the `+dev` suffix):
   ```toml
   version = "<version>"
   ```

2. Regenerate the lockfile:
   ```bash
   uv lock
   ```

3. Update `CHANGELOG.md`: promote `[Unreleased]` to `[<version>] - <date>`
   and add the corresponding comparison link at the bottom.

4. Check that documentation `Last-Updated` dates are in sync with actual
   commit dates:
   ```bash
   git log -1 --format="%ai" -- docs/design/dd-cdt-engine.md
   git log -1 --format="%ai" -- docs/design/dd-cdt-model.md
   git log -1 --format="%ai" -- docs/design/dd-cdt-modularity.md
   git log -1 --format="%ai" -- docs/design/dd-cdt-simulation.md
   git log -1 --format="%ai" -- docs/getting-started.md
   ```
   Update any `Last-Updated` fields that are out of date.

5. Verify that the runnable doc scripts are in sync with the documentation
   and execute without errors (also enforced by `CI (release)`):
   ```bash
   uv run python examples/doc/doc_engine.py
   uv run python examples/doc/doc_model.py
   uv run python examples/doc/doc_modularity.py
   uv run python examples/doc/doc_simulation.py
   uv run python examples/doc/doc_getting_started.py
   uv run python examples/doc/doc_overtourism_getting_started.py
   uv run python examples/doc/doc_readme.py
   ```

6. Verify that the full domain examples run end-to-end without errors (also
   enforced by `CI (release)`; output images are written to
   `examples/*/output/`):
   ```bash
   uv run python examples/mobility_bologna/mobility_bologna.py
   uv run python examples/overtourism_molveno/overtourism_molveno.py
   ```

7. Verify that every tracked Python and Markdown file carries an SPDX header
   (also enforced by `CI (release)`):
   ```bash
   # Python files — should print nothing
   git ls-files '*.py' | xargs grep -rL "SPDX-License-Identifier"
   # Markdown files — should print nothing
   git ls-files '*.md' | xargs grep -rL "SPDX-License-Identifier"
   ```
   Add `# SPDX-License-Identifier: Apache-2.0` (Python) or
   `<!-- SPDX-License-Identifier: Apache-2.0 -->` (Markdown) to any file
   that is missing the header.

8. Commit the release preparation:
   ```bash
   git add pyproject.toml uv.lock CHANGELOG.md docs/
   git commit -m "chore: prepare v<version> release"
   git push origin dev
   ```

Open the PR from `dev` to `main`. The `CI (release)` workflow runs the full
verification suite automatically (all Python versions, doc examples, domain
examples, SPDX headers, dependency audit, build smoke test). Merge once it
is green.

### Step 3 — Tagging and publishing

After the `dev → main` PR is merged:

```bash
git checkout main && git pull
git tag v<version> && git push origin main v<version>
```

Go to the repository's **Releases** page, review the auto-created draft, write
release notes, and click **Publish release**. This triggers the
[`publish.yml`](.github/workflows/publish.yml) workflow, which builds the
sdist + wheel, runs `twine check`, and publishes to PyPI via OIDC — no manual
build or upload step is needed.

### Step 4 — Post-release: bump `dev` back to development

After the release is published, switch back to `dev` and prepare it for the
next development cycle:

```bash
git checkout dev && git pull
```

Edit `pyproject.toml` to bump to the next planned version with the `+dev`
marker:
```toml
version = "<next-version>+dev"
```

Then:
```bash
uv lock
```

Add a fresh `[Unreleased]` section at the top of `CHANGELOG.md`:
```markdown
## [Unreleased]
```

Commit and push:
```bash
git add pyproject.toml uv.lock CHANGELOG.md
git commit -m "chore: start v<next-version> development"
git push origin dev
```

### One-time setup

> **PyPI Trusted Publisher:** must be configured before the first release.
> See the [PyPI Trusted Publishers documentation](https://docs.pypi.org/trusted-publishers/).

> **Branch protection:** configure GitHub Rulesets (Settings → Rules →
> Rulesets) to require `CI (dev)` to pass before merging into `dev`, and all
> `CI (release)` jobs to pass before merging into `main`. Direct pushes to
> `main` should be blocked; maintainers should be allowed to bypass `dev`
> protection for post-release bump commits.

## Documentation

| Document | Description |
| -------- | ----------- |
| [Getting Started](docs/getting-started.md) | Step-by-step guide: define a model with `@define`/`compute()`, sample with `DistributionEnsemble`, evaluate with `Evaluation`. |
| [dd-cdt-engine.md](docs/design/dd-cdt-engine.md) | DSL compiler engine — graph nodes, topological sorting, NumPy executor. |
| [dd-cdt-model.md](docs/design/dd-cdt-model.md) | Model layer reference — index types, `@define`/`compute()`, `Model`, `Evaluation`, `EvaluationResult`, and the domain modeling pattern. |
| [dd-cdt-modularity.md](docs/design/dd-cdt-modularity.md) | Model modularity concept guide — `@define`/`compute()`, `ModelVariant`, decomposition patterns, and Bologna worked example. |
| [dd-cdt-simulation.md](docs/design/dd-cdt-simulation.md) | Simulation guide — `Scenario`, `CrossProductEnsemble`, `EvaluationHandle`, incremental evaluation, `ModelEvaluator`. |

## License

```
SPDX-License-Identifier: Apache-2.0
```
