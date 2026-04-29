# CastFactory

CastFactory is a recipe-centric research framework for LLM-driven time series forecasting. The current implementation provides the early framework scaffold for leakage-safe data handling, representation adapters, forecast parsing, evaluation, trace artifacts, and verifiable rewards.

## Current Scope

Implemented foundation modules:

- `castfactory.core`: recipe loading, experiment shell, and registries
- `castfactory.data`: `TSRecord`, `ForecastSample`, CSV reader, timestamp split, window builder, scaler, leakage checks
- `castfactory.representation`: statistics, context, textual summary, and hybrid representations
- `castfactory.parsers`: JSON and array forecast parsers with fallback behavior
- `castfactory.evaluation`: point metrics and standard evaluator
- `castfactory.trace`: run artifact store
- `castfactory.rewards`: accuracy, format, and composite rewards
- `castfactory.training`: minimal SFT dataset formatter

The architecture plan is in [docs/CastFactory_Library_Architecture.md](docs/CastFactory_Library_Architecture.md).

## Development

```bash
python3 -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/castfactory_pycache python3 -m compileall castfactory
python3 -m castfactory.cli.run /tmp/castfactory_cli_recipe.yaml
```

Optional dev tooling is declared in `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m ruff check castfactory tests
```
