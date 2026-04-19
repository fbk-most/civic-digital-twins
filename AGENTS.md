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
uv run python examples/doc/doc_getting_started.py
uv run python examples/doc/doc_overtourism_getting_started.py
uv run python examples/doc/doc_readme.py
```

## Important Details

- **Package structure**: `civic_digital_twins/dt_model/` contains the main code with subpackages: `engine/` (DSL compiler), `model/` (higher-level abstractions), `symbols/` (Index, TimeseriesIndex).
- **Python path**: `pyproject.toml` sets `pythonpath = ["examples"]` so tests can import example packages like `mobility_bologna` and `overtourism_molveno`.
- **Pyright config**: `pyproject.toml` includes `examples` in both `include` (type-checked) and `extraPaths` (import resolution).
- **Release process**: Update version in `pyproject.toml`, run `uv lock`, update `CHANGELOG.md`, verify doc examples, then tag and push.
- **Doc/example alignment**: Every script under `examples/doc/` must be kept verbatim-aligned with its counterpart in `docs/`. This is enforced as a release checklist step — run the doc examples and confirm no drift.
- **SPDX headers**: Every `.py` file must open with `# SPDX-License-Identifier: Apache-2.0`; every `.md` file must open with `<!-- SPDX-License-Identifier: Apache-2.0 -->`. Add the header whenever you create a new file. Pre-release, verify with `git ls-files '*.py' '*.md' | xargs grep -rL "SPDX-License-Identifier"` (should print nothing).
- **Minimum Python**: 3.12 (defined in pyproject.toml).
