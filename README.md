# CastFactory

CastFactory is a recipe-centric research framework for LLM-driven time series forecasting.
It is designed to organize CPT, SFT, and RLVR experiments with explicit recipes, leakage-aware
data processing, structured forecast parsing, verifiable rewards, and reproducible trace
artifacts.

The current codebase is an active research prototype. The main implemented path targets
single-turn forecasting workflows on timestamped time-series data, with ETTh1 examples for
Qwen-style causal language models.

## Highlights

- **Recipe-first experiments**: data, representation, model, training stage, reward, evaluation,
  and trace behavior are described in YAML recipes.
- **Three-stage training flow**: `cpt`, `sft`, and `rlvr` are first-class experiment stages in
  `Experiment.fit()`.
- **Leakage-aware data layer**: CSV reading, timestamp or ratio split, rolling windows, train-only
  normalization, and visibility-separated `ForecastSample` objects.
- **Time-series representations**: textual statistics, context summaries, discrete tokens,
  numerical patches, hybrid prompts, and Markdown-table prompts.
- **Forecast parsers**: JSON, array, and `<think>/<answer>` parsers with fallback behavior.
- **Verifiable rewards**: format, accuracy, calibration, reasoning, and MSE-based rewards, with
  extensible reward composition for RLVR.
- **verl integration**: RLVR recipes can export rollout datasets, verl config files, reward
  entrypoints, and launch commands for GRPO/vLLM training.
- **Trace artifacts**: runs can store recipe snapshots, predictions, parsed results, metrics,
  errors, reports, and stage metadata.

## Repository Layout

```text
castfactory/
  cli/                 # command-line recipe runner
  core/                # RecipeConfig, Experiment, registries
  data/                # records, readers, splits, windows, leakage checks
  evaluation/          # point metrics and evaluation protocols
  models/              # backbone, bridge, head, and adapter skeletons
  parsers/             # forecast output parsers
  representation/      # time-series to LLM input representations
  rewards/             # verifiable reward functions
  trace/               # run artifact storage
  training/            # CPT/SFT/RLVR datasets, trainers, and backends

examples/
  cpt/                 # ETTh1 CPT recipes
  sft/                 # ETTh1 SFT recipes
  rlvr/                # ETTh1 GRPO/RLVR recipes and prompt template

scripts/               # 4-GPU server launch helpers
tests/                 # unit tests by module area
```

## Installation

Install the package in editable mode:

```bash
python -m pip install -e .
```

Optional HuggingFace model loading:

```bash
python -m pip install -e ".[hf]"
```

Optional local training stack:

```bash
python -m pip install -e ".[train]"
```

Development tools:

```bash
python -m pip install -e ".[dev]"
```

For RLVR with verl/vLLM, install the corresponding external runtime in the target training
environment. CastFactory prepares verl-compatible artifacts and reward entrypoints, but does not
vendor the full verl runtime.

## Quick Start

Load a recipe:

```bash
python -m castfactory.cli.run examples/sft/etth1_qwen_sft.yaml
```

Run a training stage:

```bash
python -m castfactory.cli.run examples/cpt/etth1_qwen_cpt.yaml --mode fit
python -m castfactory.cli.run examples/sft/etth1_qwen_sft.yaml --mode fit
python -m castfactory.cli.run examples/rlvr/etth1_qwen_grpo.yaml --mode fit
```

The example pipeline is checkpoint-chained:

```text
examples/cpt/etth1_qwen_cpt.yaml
  -> ./checkpoints/etth1_ot_qwen_cpt

examples/sft/etth1_qwen_sft.yaml
  -> init_checkpoint: ./checkpoints/etth1_ot_qwen_cpt
  -> ./checkpoints/etth1_ot_qwen_sft

examples/rlvr/etth1_qwen_grpo.yaml
  -> init_checkpoint: ./checkpoints/etth1_ot_qwen_sft
  -> ./checkpoints/etth1_ot_qwen_grpo
```

Command-line dotlist overrides are supported:

```bash
python -m castfactory.cli.run examples/sft/etth1_qwen_sft.yaml --mode fit \
  model.backbone.model_name=Qwen/Qwen2.5-1.5B \
  training.args.num_train_epochs=1 \
  training.args.learning_rate=2.0e-4
```

## CPT

CPT converts forecasting windows into causal language modeling text streams. The built-in
`CPTDataset` serializes observed time-series values with channel, domain, and unit metadata when
available. `TransformersCPTBackend` then tokenizes the text and trains a causal LM objective where
input tokens are also labels.

Minimal recipe shape:

```yaml
experiment:
  name: etth1_ot_qwen_cpt
  stage: cpt

data:
  reader:
    name: csv
    path: ./castfactory/dataset/ETTh1/ETTh1.csv
    timestamp_col: date
    target_channels: [OT]
  split:
    type: ratio
    ratios: [0.7, 0.1, 0.2]
  window:
    context_length: 96
    prediction_length: 96
    stride: 96

model:
  backbone:
    name: hf_causal_lm
    model_name: Qwen/Qwen2.5-1.5B

training:
  checkpoint_dir: ./checkpoints/etth1_ot_qwen_cpt
  backend:
    name: transformers
```

## SFT

SFT builds single-turn forecasting instruction examples. The dataset formats each row as an
instruction plus visible time-series context, and the output is a JSON forecast target. The
Transformers backend masks prompt tokens with `-100`, so loss is computed on response tokens.

Supported prompt inputs include representation text, future-known covariates, cutoff time,
prediction length, dataset metadata, target channel names, and template-defined placeholders.

The SFT backend exposes separate prompt and response token budgets:

```yaml
training:
  init_checkpoint: ./checkpoints/etth1_ot_qwen_cpt
  checkpoint_dir: ./checkpoints/etth1_ot_qwen_sft
  backend:
    name: transformers
    max_prompt_length: 2048
    max_response_length: 1024
```

## RLVR

RLVR is implemented as a single-turn rollout workflow. CastFactory builds rollout rows from
forecasting samples, exports them for verl, and connects model outputs back to task-specific
reward computation.

The current ETTh1 GRPO example uses:

- `MarkdownTableRepresentation` for historical context;
- an external instruction template file;
- `<think>...</think>` and `<answer>...</answer>` output structure;
- format checking before accuracy-style reward computation;
- MSE-based bounded reward;
- verl GRPO config generation with vLLM rollout settings.

Running an RLVR recipe prepares artifacts under the configured checkpoint directory:

```text
checkpoints/.../rlvr/
  rollout_dataset.jsonl
  verl_config.yaml
  launch_command.txt
```

The generated reward entrypoint is:

```text
castfactory.training.verl_reward_adapter:compute_score
```

### Prompt, Parser, and Reward Design

The RLVR path strengthens the connection between model output format and verifiable reward
calculation.

Prompt construction is template-driven. A recipe can point to an external `.txt` instruction
template through `training.instruction_template_file`, keeping prompt design separate from Python
code. Templates can reference forecasting context variables such as:

- `prediction_length`
- `cutoff_time`
- `dataset_name`
- `attr_meaning`
- `target_channels`
- `covariate_channels`
- `data_lookback`

Structured parsing is handled by `ThinkAnswerForecastParser`. It expects the model to separate
reasoning and final prediction:

````text
<think>
...
</think>
<answer>
```
...
```
</answer>
````

`FormatReward` can validate both parse success and required structural blocks. If the response
does not satisfy the expected reasoning/answer format, the reward adapter can short-circuit the
sample and penalize invalid outputs before computing numerical rewards.

For prediction correctness, CastFactory currently includes accuracy-oriented rewards such as
`AccuracyReward`, `CalibrationReward`, and `MSEReward`. `MSEReward` maps unbounded MSE into a
bounded scalar through a temperature parameter, which is useful for stabilizing RL training.

The reward interface is intentionally extensible. The planned RLVR reward family also includes
trend-aware and seasonality-aware decomposition rewards, so that future experiments can evaluate
not only pointwise error, but also whether a model captures trend direction and periodic
structure. These decomposition rewards should plug into the same reward adapter path as the
existing format and numerical rewards.

## 4-GPU Server Examples

Static 4-GPU recipes are provided for Qwen3-1.7B-style local checkpoints:

```bash
bash scripts/server_cpt_4gpu.sh
bash scripts/server_sft_4gpu.sh
bash scripts/server_rlvr_4gpu.sh
```

The scripts use CLI overrides for machine-specific paths and training settings. The RLVR server
script first asks CastFactory to prepare verl artifacts, then executes the generated
`launch_command.txt`.

## Testing

Run the standard-library test suite:

```bash
python3 -m unittest discover -s tests -v
```

Compile-check the package:

```bash
PYTHONPYCACHEPREFIX=/tmp/castfactory_pycache python3 -m compileall castfactory
```

With development dependencies installed:

```bash
python -m pytest tests -q
python -m ruff check castfactory tests
```

## Current Boundaries

CastFactory currently focuses on single-turn CPT/SFT/RLVR infrastructure for LLM-driven time
series forecasting. It does not aim to provide a broad model zoo, a web UI, AutoML, or multi-turn
agent workflows in the current MVP scope.
