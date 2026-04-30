from __future__ import annotations

import argparse
import json

from castfactory import Experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="castfactory")
    parser.add_argument("recipe", help="Path to a CastFactory recipe YAML file")
    parser.add_argument(
        "--mode",
        choices=("load", "fit", "evaluate", "report"),
        default="load",
        help="Operation to run for the recipe",
    )
    args = parser.parse_args()
    experiment = Experiment.from_recipe(args.recipe)
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
