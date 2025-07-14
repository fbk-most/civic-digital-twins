# Civic-Digital-Twins Modeling Framework

[![Build Status](https://github.com/fbk-most/civic-digital-twins/actions/workflows/test.yml/badge.svg)](https://github.com/fbk-most/civic-digital-twins/actions) [![codecov](https://codecov.io/gh/fbk-most/civic-digital-twins/branch/main/graph/badge.svg)](https://codecov.io/gh/fbk-most/civic-digital-twins) [![PyPI version](https://img.shields.io/pypi/v/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/) [![Python Versions](https://img.shields.io/pypi/pyversions/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/) [![License](https://img.shields.io/pypi/l/civic-digital-twins.svg)](https://pypi.org/project/civic-digital-twins/)

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

## License

```
SPDX-License-Identifier: Apache-2.0
```
