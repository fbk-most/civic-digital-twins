<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agent Guidance for civic-digital-twins

## Development Commands

All commands should be run via `uv` (use `uv run` for python scripts, or prefix bin commands with `uv run`).

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync --dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/dt_model/engine/frontend/test_graph.py

# Lint and format check
uv run ruff check .
uv run ruff format --check .

# Type check
uv run pyright

# Run doc examples (must pass for releases; must stay verbatim-aligned with docs/)
uv run python examples/doc/doc_engine.py
uv run python examples/doc/doc_model.py
uv run python examples/doc/doc_modularity.py
uv run python examples/doc/doc_simulation.py
uv run python examples/doc/doc_getting_started.py
uv run python examples/doc/doc_overtourism_getting_started.py
uv run python examples/doc/doc_readme.py
```

## Branch model

This project follows a simplified GitHub Flow with an explicit `dev` branch:

```
feature/* ──PR──▶ dev ──PR──▶ main ──tag──▶ PyPI
            (CI dev)   (CI release)   (publish)
```

- Feature branches are cut from `dev` and merged back into `dev` via PR.
- `dev` always carries a `+dev` version marker (e.g. `0.11.0+dev`).
- `main` contains only released commits; every merge to `main` is immediately
  tagged and published to PyPI.
- Post-release, `dev` is bumped to `<next>+dev` with a direct push by a
  maintainer (bypassing the branch protection PR requirement).

## Dev PR checklist

Follow the checklist in the **"Step 1 — Merging a feature PR into `dev`"**
section of `README.md`.

## Release checklist (dev → main)

Follow the steps in the **"Releasing"** section of `README.md` (Steps 2, 3,
and 4).

## Important Details

- **Package structure**: `civic_digital_twins/dt_model/` contains the main code with subpackages: `engine/` (DSL compiler), `model/` (indexes, models, contracts), `simulation/` (scenarios, ensembles, evaluation, runners). Top-level modules follow the module-role convention (see README “Conceptual Overview”): `axes.py` is the canonical home of the cross-cutting axis vocabulary; `graph.py` is a curated user façade over `engine/frontend/graph.py`.
- **Python path**: `pyproject.toml` sets `pythonpath = ["examples"]` so tests can import example packages like `mobility_bologna` and `overtourism_molveno`.
- **Pyright config**: `pyproject.toml` includes `examples` in both `include` (type-checked) and `extraPaths` (import resolution).
- **CI uses `--locked`**: Both `ci-dev.yml` and `ci-release.yml` run `uv sync --locked`, which fails if `uv.lock` is out of sync with `pyproject.toml`. Always commit an updated `uv.lock` alongside any `pyproject.toml` change.
- **Doc/example alignment**: Every script under `examples/doc/` must be kept verbatim-aligned with its counterpart in `docs/`. Enforced by `CI (release)` (`doc-examples` job).
- **SPDX headers**: Every `.py` file must open with `# SPDX-License-Identifier: Apache-2.0`; every `.md` file must open with `<!-- SPDX-License-Identifier: Apache-2.0 -->`. Add the header whenever you create a new file. Enforced by `CI (release)` (`spdx-check` job).
- **Minimum Python**: 3.12 (defined in pyproject.toml).
