# Civic-Digital-Twins Modeling Framework

[![Build Status](https://github.com/fbk-most/civic-digital-twins/actions/workflows/test.yml/badge.svg)](https://github.com/fbk-most/civic-digital-twins/actions) [![codecov](https://codecov.io/gh/fbk-most/civic-digital-twins/branch/main/graph/badge.svg)](https://codecov.io/gh/fbk-most/civic-digital-twins) [![PyPI version](https://img.shields.io/pypi/v/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/) [![Python Versions](https://img.shields.io/pypi/pyversions/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/) [![License](https://img.shields.io/pypi/l/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/)

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
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
import numpy as np

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
- **`Model`** — a named collection of `Index` objects.  A model is
  *abstract* when some indexes are unbound (distributions or placeholders);
  it becomes concrete once all indexes are resolved.
- **`Evaluation`** — evaluates a model over a sequence of *weighted
  scenarios*, each of which maps every abstract index to a concrete value.
- **`Ensemble`** / **`WeightedScenario`** — a protocol and type alias that
  define the scenario contract consumed by `Evaluation`.

```python
from civic_digital_twins.dt_model import Evaluation, Model
from civic_digital_twins.dt_model.model.index import Index, UniformDistIndex

# Two distribution-backed indexes
x = UniformDistIndex("x", loc=0.0, scale=1.0)
y = UniformDistIndex("y", loc=0.0, scale=1.0)
result = Index("result", x + y)

model = Model("example", [x, y, result])
```

### Usage patterns

Two concrete usage patterns are illustrated in the `examples/` directory.

**Direct pattern** (`examples/mobility_bologna/`) — the model consists
entirely of `Index` and `TimeseriesIndex` objects;
`DistributionEnsemble` samples each distribution-backed index to produce
weighted scenarios; `Evaluation.evaluate()` runs the engine and returns an
`EvaluationResult`.

**Vertical extension pattern** (`examples/overtourism_molveno/`) — the
domain-specific layer introduces `ContextVariable`, `PresenceVariable`,
`Constraint`, and `OvertourismModel` on top of the core types.
`OvertourismEnsemble` samples context variables and produces weighted
scenarios.  `Evaluation.evaluate(axes={pv: array, …})` evaluates the model
on a multi-dimensional grid, returning arrays of shape `(N₀, …, Nₖ, S)`
where `S` is the number of scenarios and each `Nᵢ` corresponds to one
presence axis.

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

```Python
import civic_digital_twins
```

or

```Python
from civic_digital_twins import dt_model
```

## Minimum Python Version

Python 3.11.

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

Each pull request is automatically tested using GitHub Actions. The workflow
is defined in [`.github/workflows/test.yml`](.github/workflows/test.yml).

## Updating Dependencies

```bash
uv self update
uv sync --upgrade
```

## Releasing

1. Make sure the version number in `pyproject.toml` is correct.

2. Make sure you are outside the virtual environment.

3. Make sure `python3-hatchling` is installed (`sudo apt install python3-hatchling`).

4. Make sure `twine` is installed (`sudo apt install twine`).

5. Build the package using `python3 -m hatchling build`.

6. Check whether the package is okay using `twine check dist/*`.

7. Upload the package to PyPI using `twine upload dist/*`.

## Documentation

| Document | Description |
| -------- | ----------- |
| [Getting Started](docs/getting-started.md) | Step-by-step guide covering the direct and vertical extension usage patterns. |
| [dd-cdt-engine.md](docs/design/dd-cdt-engine.md) | DSL compiler engine — graph nodes, topological sorting, NumPy executor. |
| [dd-cdt-model.md](docs/design/dd-cdt-model.md) | Model / simulation layer — `Model`, `Evaluation`, `WeightedScenario`, and the vertical extension pattern. |

## License

```
SPDX-License-Identifier: Apache-2.0
```
