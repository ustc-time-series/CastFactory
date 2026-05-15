from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from castfactory import Experiment


def parse_dotlist_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        key, raw_value = item.split("=", 1)
        path = tuple(part for part in key.split(".") if part)
        if not path:
            raise ValueError(f"Override key cannot be empty: {item}")
        _deep_set(overrides, path, yaml.safe_load(raw_value))
    return overrides


def apply_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = {} if not value else apply_overrides(existing, value)
        else:
            merged[key] = value
    return merged


def _deep_set(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for part in path[:-1]:
        existing = current.setdefault(part, {})
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set nested override through scalar key: {part}")
        current = existing
    current[path[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(prog="castfactory")
    parser.add_argument("recipe", help="Path to a CastFactory recipe YAML file")
    parser.add_argument(
        "--mode",
        choices=("load", "fit", "evaluate", "report"),
        default="load",
        help="Operation to run for the recipe",
    )
    args, overrides = parser.parse_known_args()
    recipe_path = Path(args.recipe)
    if overrides:
        loaded = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("recipe file must contain a YAML mapping")
        mapping = apply_overrides(loaded, parse_dotlist_overrides(overrides))
        experiment = Experiment.from_mapping(mapping)
        experiment.recipe_path = recipe_path
    else:
        experiment = Experiment.from_recipe(recipe_path)
    if args.mode == "load":
        print(f"Loaded CastFactory experiment: {experiment.name}")
        return
    if args.mode == "fit":
        result = experiment.fit()
    elif args.mode == "evaluate":
        result = experiment.evaluate()
    else:
        result = {"report": str(experiment.report())}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
