#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python -m castfactory.cli.run "$ROOT_DIR/examples/rlvr/etth1_qwen_grpo.yaml" --mode fit
