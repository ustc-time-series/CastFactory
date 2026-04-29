from __future__ import annotations

import argparse

from castfactory import Experiment


def main() -> None:
    parser = argparse.ArgumentParser(prog="castfactory")
    parser.add_argument("recipe", help="Path to a CastFactory recipe YAML file")
    args = parser.parse_args()
    experiment = Experiment.from_recipe(args.recipe)
    print(f"Loaded CastFactory experiment: {experiment.name}")


if __name__ == "__main__":
    main()
