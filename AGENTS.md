# Repository Guidelines

## Project Structure & Module Organization

CastFactory is a recipe-centric framework for CPT, SFT, and RLVR in LLM-driven time series forecasting.

- `castfactory/` contains source code.
- `tests/` contains unit tests grouped by module area.
- `docs/CastFactory_Library_Architecture.md` is the architecture and MVP scope reference.
- `runs/`, `checkpoints/`, and `datasets/` are ignored local artifact directories.

## Build, Test, and Development Commands

Use the standard-library test runner in the current local environment:

```bash
python3 -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/castfactory_pycache python3 -m compileall castfactory
python3 -m castfactory.cli.run /tmp/castfactory_cli_recipe.yaml
```

After installing dev dependencies, use:

```bash
python -m pytest tests -q
python -m ruff check castfactory tests
```

## Coding Style & Naming Conventions

Use Python 3.9+ compatible code, 4-space indentation, and type hints for public APIs. Prefer small modules with one responsibility.

- `snake_case` for functions, modules, and variables.
- `PascalCase` for classes such as `ForecastSample`, `RecipeConfig`, and `JSONForecastParser`.
- Registry keys use lowercase snake case, e.g. `textual_summary`, `forecast_json_v1`.

## Testing Guidelines

Write tests under `tests/<module>/test_*.py`. Test behavior, not implementation details. Prioritize leakage checks, parser fallback behavior, recipe validation, metric correctness, and reward scoring.

## Commit & Pull Request Guidelines

Use Conventional Commits:

```bash
feat: add standard evaluator
test: cover parser fallbacks
docs: add architecture guide
```

Pull requests should include a short summary, test results, and any architecture impact. For data safety, parser, recipe schema, or reward changes, include a minimal before/after example.

## Agent-Specific Instructions

Read `docs/CastFactory_Library_Architecture.md` before coding. Keep MVP work focused on single-turn CPT/SFT/RLVR infrastructure. Do not introduce multi-turn agent workflows unless explicitly requested.
