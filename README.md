# Civic-Digital-Twins Modeling Framework

[![Build Status](https://github.com/fbk-most/dt-model/actions/workflows/test.yml/badge.svg)](https://github.com/fbk-most/dt-model/actions) [![codecov](https://codecov.io/gh/fbk-most/dt-model/branch/main/graph/badge.svg)](https://codecov.io/gh/fbk-most/dt-model) [![PyPI version](https://img.shields.io/pypi/v/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/) [![Python Versions](https://img.shields.io/pypi/pyversions/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/) [![License](https://img.shields.io/pypi/l/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/)

This repository contains a Python package implementing a Civic-Digital-Twins
modeling framework. The framework is designed to support defining digital
twins models and evaluating them in simulated environment with varying
contextual conditions. We develop this package at [@fbk-most](
https://github.com/fbk-most), a research unit at [Fondazione Bruno Kessler](
https://www.fbk.eu/en/).

*Note: this package is currently in an early development stage.*

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

The installed package name is currently named `dt_model` because
that was the package name used ahead of releasing. We will eventually
transition onto a more consistent package name, and deprecate the
old package name. In the meanwhile, use as follows:

```Python
import dt_model
```

or

```Python
from dt_model import Model  # and other classes...
```

## API Stability Guarantees

The package is currently in an early development stage. We do not
anticipate breaking APIs without a good reason to do so, yet, breaking
changes may occur from time to time. We generally expect subpackages
within the top-level package to change more frequently.

## Development Setup

We use [uv](https://astral.sh/uv) for managing the development environment.

To get started, run:

```bash
git clone https://github.com/fbk-most/dt-model
cd dt-model
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

## License

```
SPDX-License-Identifier: Apache-2.0
```
