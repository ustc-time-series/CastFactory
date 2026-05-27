# CastFactory

CastFactory is a recipe-centric research framework for LLM-driven time series forecasting.
It is designed to organize CPT, SFT, and RLVR experiments with explicit recipes, leakage-aware
data processing, structured forecast parsing, verifiable rewards, and reproducible trace
artifacts.

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

Optional time-series agentic RL helpers:

```bash
python -m pip install -e ".[agentic]"
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

RLVR supports both the default single-turn rollout workflow and a `time_series_agent` workflow
that uses verl native AgentLoop for multi-turn tool use. CastFactory builds rollout rows from
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
  agent_loop_config.yaml  # only for rollout.workflow.name: time_series_agent
  launch_command.txt
```

The generated reward entrypoint is:

```text
castfactory.training.verl_reward_adapter:compute_score
```

For Cast-R1 style agentic training, add a workflow section to an RLVR recipe:

```yaml
rollout:
  workflow:
    name: time_series_agent
    max_steps: 3
    max_parallel_calls: 5
    tool_parser_format: hermes
    model_service_url: http://localhost:8994
    prediction_models: [chronos2, arima, patchtst, itransformer]
    local_fallback: arima_then_last_value
```

This path is currently univariate. It writes raw chat prompts, `agent_name:
time_series_forecast_agent`, Cast-R1 style timestamp/value ground truth, and a native verl
`agent_loop_config.yaml`. The `predict_time_series` tool tries the configured HTTP model service
first and falls back locally to ARIMA, then last-value forecasting.

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
