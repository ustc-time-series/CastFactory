# CastFactory 开源库架构设计方案

> 版本：v0.2  
> 日期：2026-04-29  
> 仓库：https://github.com/ustc-time-series/CastFactory  
> 定位：面向 LLM-driven Time Series Forecasting 的 recipe-centric 训练、评测与复现实验框架。

---

## 1. 项目定位

CastFactory 是一个面向 LLM 时代时间序列预测研究的开源库。它不应被设计成另一个 model zoo，而应成为一个统一的 research harness：用同一套 recipe、数据安全机制、模型组合接口、训练阶段、评测协议和实验 trace，支撑研究者系统地探索 LLM-driven forecasting 方法。

CastFactory 的核心关注点是三类训练阶段：

1. **CPT, Continual Pre-Training**：在大规模时序语料上继续预训练 LLM，让其获得时序数值、统计模式、领域文本和事件上下文之间的语义理解能力。
2. **SFT, Supervised Fine-Tuning**：通过单轮 forecasting instruction，让模型学会根据历史窗口、任务指令和上下文信息输出符合格式要求的预测结果。
3. **RLVR, Reinforcement Learning with Verifiable Rewards**：用可验证的预测精度、校准性和格式合规奖励，训练模型更可靠地进行时间序列预测推理。

这里的"多模态"主要指时间序列预测中的异构输入模态，包括数值序列、时间戳、变量描述、统计摘要、日历事件、领域文本、协变量和结构化元数据，而不是以图像/视频为中心的通用多模态任务。

---

## 2. 设计原则

### 2.1 Recipe First

实验方法必须首先能被 recipe 描述。研究者不应在多个训练脚本中硬编码数据窗口、prompt、tokenizer、backbone、loss、reward 和评测方式。

一次实验应由 recipe 完整恢复，包括：

- 数据集和 temporal split（含切分时间戳，而非仅比例）
- lookback / horizon / rolling 策略
- 输入表征方式
- LLM backbone 或 TS foundation model
- bridge、adapter 和 output head
- CPT / SFT / RLVR 阶段配置
- 评测协议与指标
- 随机种子、checkpoint、prompt、response、reward 和预测结果

### 2.2 Stage Explicit

CPT、SFT、RLVR 是一等公民，而不是藏在 `train.py` 里的 flag。每个阶段有独立 trainer、dataset、objective、checkpoint contract 和评测入口，同时支持阶段间串联。

### 2.3 Leakage-Safe

时间序列预测框架最容易出现信息泄漏。CastFactory 必须在数据层显式区分：

- **observed context**：预测时真实可见的历史信息。
- **future-known context**：预测时提前可知的信息，例如节假日、计划排产、已发布天气预报。
- **future-unknown context**：预测时不可见的信息，只能作为 label 或评测真值。
- **derived context**：统计特征、文本摘要、检索片段等派生信息，必须只基于 cutoff time 之前的数据生成。

所有 normalization、窗口生成、prompt 构造、统计摘要和检索上下文都必须遵守 cutoff。

**切分策略**：时间切分必须以 timestamp 为锚点，而非序列比例。ratio-based split 在滑动窗口场景下无法保证窗口不跨越切分边界，LeakageChecker 默认拒绝比例切分模式，详见 §5.2。

### 2.4 Backend Agnostic

CastFactory 不绑定单一模型或训练框架。它应该能够适配：

- HuggingFace causal LM，例如 Qwen、LLaMA、GPT-style models
- TS foundation model，例如 Chronos、TimesFM、Moirai
- PEFT 方法，例如 LoRA、QLoRA、prefix tuning
- 分布式训练后端，例如 Accelerate、DeepSpeed、FSDP
- RL 后端，例如 verl、TRL 或轻量 GRPO 实现
- 推理加速引擎，例如 vLLM（RLVR rollout 阶段必须）

### 2.5 Traceable

LLM-driven forecasting 的复现不只依赖 checkpoint。CastFactory 需要保存 prompt、response、reasoning output、解析后的 forecast、reward breakdown、metric breakdown 和完整 recipe snapshot。

---

## 3. 总体架构

CastFactory 采用 recipe-centric layered architecture：

```text
Recipe / Config Layer
        |
Experiment Orchestrator
        |
+-------+----------------+---------+----------+---------+
| Data  | Representation | Model   | Training | Parsers |
+-------+----------------+---------+----------+---------+
        |                                   |
        +-----------> Rewards <-------------+
        |
Evaluation / Trace
        |
Reports / Leaderboard / Artifacts
```

**架构说明**：Rewards 同时被 Training（RLVR 阶段）和 Evaluation（指标计算）引用，是跨层共享组件；Parsers 独立于 Evaluation，负责把 LLM 文本输出转为结构化预测，是整个 LLM-driven forecasting 管道的关键路径。

各层职责如下：

| 层 | 职责 |
|---|---|
| Recipe / Config | 描述实验，不承载业务逻辑 |
| Experiment | 解析 recipe，构建组件，串联训练和评测 |
| Data | 读取数据、时间切分、窗口构造、防泄漏检查 |
| Representation | 将时序、统计、文本、上下文转换为 embedding 或 token id |
| Model | 组合 backbone、bridge（将 embedding 注入 LLM）、head |
| Parsers | 从 LLM 文本输出中稳定解析结构化预测，含 fallback 策略 |
| Training | 提供 CPT / SFT / RLVR 阶段训练器 |
| Evaluation | 标准化评测协议、指标和 leaderboard |
| Rewards | 可验证 reward 函数库，被 Training 和 Evaluation 共用 |
| Trace | 保存实验复现所需的全部输入、输出和中间产物 |

---

## 4. 推荐目录结构

```text
CastFactory/
├── castfactory/
│   ├── core/
│   │   ├── experiment.py
│   │   ├── recipe.py
│   │   ├── config.py
│   │   ├── registry.py
│   │   └── seed.py
│   │
│   ├── data/
│   │   ├── records.py          # TSRecord, ForecastSample
│   │   ├── readers/
│   │   ├── splits/             # TimestampSplitter（默认）, RatioSplitter（需显式启用）
│   │   ├── windows/
│   │   ├── transforms/
│   │   ├── covariates.py
│   │   └── leakage.py
│   │
│   ├── representation/
│   │   ├── base.py             # RepresentationAdapter 抽象基类
│   │   ├── numerical_patch.py  # 连续 patch embedding（无离散化）
│   │   ├── discrete_token.py   # 数值离散化 → token id
│   │   ├── textual_summary.py
│   │   ├── statistics.py
│   │   ├── context.py
│   │   └── hybrid.py
│   │
│   ├── parsers/                # 独立 Parser 模块（关键路径）
│   │   ├── base.py             # ForecastParser 抽象基类
│   │   ├── json_parser.py      # JSON schema 解析
│   │   ├── array_parser.py     # 纯数字列表解析
│   │   ├── markdown_parser.py  # Markdown 表格解析
│   │   └── registry.py
│   │
│   ├── models/
│   │   ├── backbones/
│   │   │   ├── hf_causal_lm.py
│   │   │   ├── chronos.py
│   │   │   ├── timesfm.py
│   │   │   └── moirai.py
│   │   ├── bridges/            # 将 Representation 输出注入 LLM
│   │   │   ├── projector.py        # Linear / MLP projector
│   │   │   ├── soft_prompt.py      # Soft prompt prepend
│   │   │   ├── cross_attention.py  # Cross-attention fusion
│   │   │   └── text_concat.py      # 文本拼接（无额外参数）
│   │   ├── heads/
│   │   │   ├── point_forecast.py
│   │   │   ├── quantile_forecast.py
│   │   │   ├── distribution.py
│   │   │   └── text_generation.py
│   │   └── adapters/
│   │       ├── peft.py
│   │       └── checkpoint.py
│   │
│   ├── training/
│   │   ├── base_trainer.py
│   │   ├── cpt_trainer.py
│   │   ├── sft_trainer.py
│   │   ├── rlvr_trainer.py
│   │   ├── objectives/
│   │   └── callbacks/
│   │
│   ├── rewards/
│   │   ├── base.py             # RewardFunction 抽象基类
│   │   ├── accuracy.py
│   │   ├── calibration.py
│   │   ├── format.py
│   │   ├── reasoning.py        # 实验性，MVP 阶段为可选
│   │   └── composite.py
│   │
│   ├── evaluation/
│   │   ├── protocols/
│   │   ├── metrics/
│   │   ├── leaderboard.py
│   │   └── reports.py
│   │
│   ├── trace/
│   │   ├── logger.py
│   │   ├── artifacts.py
│   │   ├── prompt_store.py
│   │   └── run_store.py
│   │
│   └── cli/
│       ├── run.py
│       ├── evaluate.py
│       └── report.py
│
├── recipes/
│   ├── cpt/
│   ├── sft/
│   ├── rlvr/
│   └── baselines/
│
├── benchmarks/
├── examples/
├── tests/
├── docs/
└── pyproject.toml
```

---

## 5. 核心模块边界

### 5.1 Core

`core` 负责实验编排，不直接实现模型或数据逻辑。

核心对象：

```python
class Experiment:
    @classmethod
    def from_recipe(cls, path: str) -> "Experiment": ...
    def fit(self) -> None: ...
    def evaluate(self) -> dict: ...
    def predict(self, inputs: dict) -> dict: ...
    def report(self) -> None: ...
```

`Experiment` 只做组件构建和生命周期管理，具体能力通过 registry 注入。

**注意**：所有 `registry.register_xxx()` 调用必须在 `Experiment.from_recipe()` 之前完成，不可在 DataLoader worker 进程中注册（multiprocessing fork 不会继承主进程的注册表状态）。

### 5.2 Data

`data` 负责从原始数据到 `ForecastSample` 的转换。

#### 数据结构设计

```python
@dataclass
class TSRecord:
    """完整时序记录，存储一段原始时序数据，不区分可见性。
    values shape: (T, C)，C=1 表示单变量，C>1 表示多变量。
    """
    values: np.ndarray              # shape: (T, C)
    timestamps: pd.DatetimeIndex    # length: T
    channel_names: list[str]        # length: C，变量名
    target_channels: list[str]      # 需要预测的变量名子集
    covariate_channels: list[str]   # 协变量列名子集
    static_context: dict            # 不随时间变化的元信息（领域、单位等）
    metadata: dict                  # 其他附加字段


@dataclass
class ForecastSample:
    """单个预测样本，明确区分三类可见性窗口。
    
    这是数据层对外暴露的最终格式，Representation / Trainer / Reward 均以此为输入。
    ForecastSample 在 WindowBuilder 阶段由 TSRecord 切分生成，
    切分后 future_unknown 字段中的值对模型不可见（仅用于 label 和评测）。
    """
    observed_window: TSRecord       # cutoff 之前，模型可见
    future_known_window: TSRecord   # cutoff 之后，但提前已知（如节假日）
    future_unknown_window: TSRecord # cutoff 之后，不可见，作为预测 label
    cutoff_time: pd.Timestamp
    prediction_length: int
    metadata: dict


@dataclass
class ForecastResult:
    point_forecast: np.ndarray      # shape: (H,) 或 (H, C)
    quantile_forecast: dict         # {"q10": array, "q90": array}
    raw_response: str               # LLM 原始文本输出
    parse_success: bool
    metadata: dict
```

**多变量说明**：`values` 统一使用 `(T, C)` 形状。`C=1` 退化为单变量。Representation 层需在 recipe 中通过 `multivariate_mode` 声明处理方式（`channel_independent` 或 `channel_joint`），不可隐式假设。

#### 切分策略

**首选：timestamp-based split**（唯一保证无窗口泄漏的方式）：

```yaml
split:
  type: timestamp
  train_end:  "2022-12-31 23:00"
  val_end:    "2023-06-30 23:00"
  test_end:   "2024-10-31 23:00"
```

**次选：ratio-based split**（需显式声明，LeakageChecker 会额外检查窗口边界）：

```yaml
split:
  type: ratio
  ratios: [0.7, 0.1, 0.2]
  warn_on_window_overlap: true   # 必须为 true，否则报错
```

ratio 模式下，LeakageChecker 会检查所有 val/test 窗口的 `observed_window` 是否与 train 集时间范围重叠，发现重叠时抛出 `LeakageWarning` 并记录到 `errors.jsonl`。

#### 关键边界

- Reader 只负责读原始数据，输出 `TSRecord`，不做切分。
- Splitter 只负责按时间边界切分，输出 train/val/test 三段 `TSRecord`，不做 normalization。
- WindowBuilder 以 `TSRecord` 为输入，按 `context_length` 和 `prediction_length` 构造 `ForecastSample` 列表，`stride` 默认为 `prediction_length`（非重叠），需大数据集时可设为 1，但会显著增加样本数。
- Transform（normalization）必须声明 `fit_scope: train_only`，基于 train 集统计量对 val/test 做 transform。
- LeakageChecker 检查 Transform、Representation、derived context 是否越过 `cutoff_time`。

### 5.3 Representation

`representation` 是 CastFactory 的关键差异化模块。它负责把 `ForecastSample` 的 `observed_window`（以及可见的 `future_known_window`）转换为 LLM 可消费的 embedding 或 token id。

**职责边界**：Representation 负责数据→模型输入格式的转换（embedding tensor 或 token id sequence）。如何把这些输出注入 LLM 是 Bridge 的职责，两者不重叠。

核心接口：

```python
class RepresentationAdapter:
    """
    encode 输出 ModelInput，包含：
      - embeddings: Optional[Tensor]   # shape: (T', D)，连续表征
      - token_ids:  Optional[Tensor]   # shape: (T',)，离散 token id
      - text_prompt: Optional[str]     # 文本拼接表征
    """
    def encode(self, sample: ForecastSample) -> "ModelInput": ...
```

**注意**：Representation 不提供 `decode` 方法——解码（从 LLM 输出恢复预测值）由独立的 Parser 模块负责（见 §5.5）。

内置 representation：

| 表征 | 输出类型 | 多变量处理 | 说明 |
|---|---|---|---|
| NumericalPatchRepresentation | embeddings | channel_independent / joint | 按 patch_size 切分，线性投影为 embedding |
| DiscreteTokenRepresentation | token_ids | channel_independent | 数值离散化为 token id，加入 LLM 词表 |
| TextualSummaryRepresentation | text_prompt | joint（统一描述） | 趋势、周期、异常、统计特征转文本 |
| StatisticsRepresentation | text_prompt | 逐通道描述 | 均值、方差、ACF、频域、趋势强度 |
| ContextRepresentation | text_prompt | — | 变量描述、领域信息、事件、日历、协变量 |
| HybridRepresentation | 组合 | 取决于子组件 | 组合多种表征，合并为统一 ModelInput |

设计重点是允许研究者比较不同表征，而不是默认"文本化时序"就是最优解。

### 5.4 Model

`models` 采用 **Backbone + Bridge + Head** 的组合方式，三者职责严格分离。

```text
输入流：
  ForecastSample
      ↓ Representation
  ModelInput（embeddings / token_ids / text_prompt）
      ↓ Bridge（将 Representation 输出注入 LLM）
  LLM hidden states
      ↓ Head
  LLM 原始输出（logits / 文本）
      ↓ Parser（见 §5.5）
  ForecastResult
```

**Bridge 职责**：将 Representation 的输出（embedding 或 text）拼接或注入 LLM 的输入空间，不做数值离散化（那是 DiscreteTokenRepresentation 的职责）。

```text
Backbone: Qwen / LLaMA / GPT2 / Chronos / TimesFM / Moirai

Bridge（按注入方式分类）:
  text_concat      — 直接将文本表征拼接到 prompt（无额外参数，适合 TextualSummary / Statistics）
  projector        — Linear / MLP 将 embedding 投影到 LLM 维度（适合 NumericalPatch）
  soft_prompt      — 可学习 soft token 拼接（适合少参数微调实验）
  cross_attention  — 时序 embedding 通过 cross-attention 与 LLM 交互

Head:
  point_forecast   — 回归头，输出连续预测值
  quantile_forecast — 多分位数头
  distribution     — 参数化分布头（Gaussian / Student-t）
  text_generation  — 依赖 Parser 解析 LLM 文本输出
```

这种拆分可以表达多种 LLM-driven TSF 路线：

```text
路线 A: DiscreteToken + text_concat + LLM + text_generation head + JSON Parser
路线 B: Statistics/TextualSummary + text_concat + LLM + text_generation head + Parser
路线 C: NumericalPatch + projector + LLM + point_forecast head（无需 Parser）
路线 D: TS foundation model + unified evaluation adapter（zero-shot baseline）
```

### 5.5 Parsers（独立模块）

Parser 负责从 LLM 文本输出中稳定提取结构化预测，是 text_generation head 路线的关键路径。Parse 失败会导致整条样本无法计算 metric 和 reward，因此 Parser 需要内置 fallback 策略，而不是直接抛异常。

核心接口：

```python
class ForecastParser:
    def parse(self, raw_text: str, context: "ParseContext") -> "ParseResult": ...

@dataclass
class ParseContext:
    prediction_length: int
    num_channels: int
    output_schema: str          # 注册的 schema 名称
    channel_names: list[str]

@dataclass
class ParseResult:
    success: bool
    point_forecast: Optional[np.ndarray]    # shape: (H,) 或 (H, C)
    quantile_forecast: Optional[dict]
    parse_error: Optional[str]
    fallback_used: bool
    fallback_strategy: Optional[str]        # "last_value" / "mean" / "zero"
```

内置 Parser：

| Parser | 目标格式 | 说明 |
|---|---|---|
| JSONForecastParser | `{"forecast": [v1, v2, ...]}` | 支持嵌套 schema，含量化区间 |
| ArrayForecastParser | `[v1, v2, ...]` | 纯数字列表，正则提取 |
| MarkdownTableParser | Markdown 表格 | 逐行解析 |

Fallback 策略（parse 失败时触发，按优先级）：
1. 提取输出中所有数字，截取前 `prediction_length` 个
2. 使用 `observed_window` 最后一个值重复填充（last_value）
3. 使用 `observed_window` 均值填充（mean）
4. 全零填充（zero，最终保底）

每次 fallback 都记录到 `errors.jsonl`，并在 metric 汇总中统计 `ParseSuccessRate`。

### 5.6 Training

`training` 提供三个显式阶段：

| Trainer | 输入 | 目标 | 输出 |
|---|---|---|---|
| CPTTrainer | 大规模 TSRecord + token sequence | next-token prediction on time-series tokens | CPT checkpoint |
| SFTTrainer | ForecastSample + instruction template | 指令遵循、格式约束、预测生成 | SFT checkpoint |
| RLVRTrainer | rollout prompt + generated response + verifiable label | 可验证推理能力训练 | RLVR checkpoint |

训练层支持的 objective（按阶段归属）：

**CPT objective**：
- Next-token prediction（NTP）on time-series tokens（**MVP 核心**）
- Masked span reconstruction（Research Extension）
- Time-text contrastive alignment（Research Extension）

**SFT objective**：
- Output-only causal LM loss（输出 token 上的 NLL，忽略 input token）
- Forecasting head supervised loss（适用于路线 C）
- 混合 loss（LM loss + forecasting loss）

**RLVR objective**：
- GRPO（推荐首选，无需 value model）
- PPO（需额外 critic，资源消耗高）
- DPO-style offline preference（适合有 preference pair 数据）

**注意**：Preference / ranking loss 属于 RLVR 阶段，不适用于标准 SFT。

### 5.7 Rewards

`rewards` 只处理可验证奖励，不处理多轮工具环境。Reward 函数同时被 `RLVRTrainer`（训练期计算）和 `Evaluation`（评测期统计）调用。

内置 reward：

```text
R_total =
  alpha * R_accuracy       (核心，MVP 必选)
+ beta  * R_calibration    (概率预测时启用)
+ gamma * R_format         (核心，MVP 必选)
+ delta * R_reasoning      (实验性，MVP 可选)
```

| Reward | 类型 | 说明 |
|---|---|---|
| AccuracyReward | 核心 | 基于 MAE、MSE、MAPE、SMAPE、MASE，与真实 label 比较 |
| CalibrationReward | 核心 | 基于 coverage、interval sharpness、CRPS |
| FormatReward | 核心 | 规则检查：JSON 合法性、数组长度 = prediction_length、字段名称、数值范围合理性 |
| ReasoningReward | 实验性 | 见下方说明 |
| CompositeReward | 工具 | 按权重组合多个 reward，归一化到 [-1, 1] |

**ReasoningReward（实验性组件，MVP 阶段为可选）**

ReasoningReward 检查模型的推理文本是否符合时间语义规则。MVP 阶段只实现可程序验证的规则子集：

- `horizon_length_check`：推理中提及的预测步数是否与 `prediction_length` 一致（正则提取数字）
- `no_future_timestamp`：推理中是否出现 cutoff 之后的时间戳（正则匹配日期格式）
- `channel_name_validity`：推理中引用的变量名是否在 `channel_names` 范围内

高级规则（如"是否使用了可见证据"、"推理是否符合时间顺序"）依赖 NLP 语义解析，**不在 MVP 范围内**，留作 Research Extension。

### 5.8 Evaluation

评测层必须标准化协议，而不是只提供 metric 函数。

内置协议：

| Protocol | 说明 |
|---|---|
| StandardEval | 固定 timestamp train/val/test split |
| RollingEval | 按时间滚动预测，降低偶然切分影响 |
| ZeroShotEval | 不在目标数据集上训练，直接预测 |
| FewShotEval | 少量目标序列样本适配或 in-context 示例 |
| CrossDomainEval | 跨领域泛化，例如 energy train 到 weather test |
| StressEval | 缺失、异常、长 horizon 压力测试（MVP 后阶段实现） |

**StressEval 定义**（防止歧义）：
- `missing`：随机 mask 5%/15%/30% context 值
- `anomaly`：注入合成异常点（3σ 之外）
- `long_horizon`：prediction_length 扩展到训练时的 2x / 4x

指标体系：

- 点预测：MAE、MSE、RMSE、MAPE、SMAPE、MASE
- 概率预测：CRPS、Winkler Score、Coverage、Interval Sharpness
- LLM 输出：FormatValidRate、ParseSuccessRate（一等公民，parse 失败会影响所有其他指标）
- 推理质量（可选）：HorizonLengthAccuracy、NoFutureTimestampRate
- 成本：tokens_per_sample、latency_ms、gpu_memory_gb

### 5.9 Trace

Trace 是 LLM-driven forecasting 可复现的核心。

每次 run 应保存：

```text
runs/{run_id}/
├── recipe.yaml                 # 原始 recipe（未解析）
├── resolved_config.yaml        # 完全展开后的配置（含所有默认值）
├── environment.json            # Python 版本、关键包版本、CUDA 版本、hostname
├── checkpoints/
├── predictions.parquet         # 列：sample_id, cutoff_time, channel, step, pred, target
├── metrics.json                # 所有协议 × 所有指标
├── leaderboard.md
├── prompts.jsonl               # 每条样本的完整 prompt
├── responses.jsonl             # 每条样本的原始 LLM 输出
├── parsed.jsonl                # 解析后的 ForecastResult（含 parse_success, fallback_used）
├── rewards.jsonl               # RLVR 阶段各 reward breakdown
├── errors.jsonl                # parse 失败、leakage 警告、异常
└── report.md
```

---

## 6. 三阶段训练设计

### 6.1 CPT：时序语言预训练

CPT 的目标是在现有 LLM checkpoint 基础上**继续预训练**（不从头训练），让模型学习时序的"语言"：数值变化、周期性、趋势、异常、领域语义和统计结构之间的关系。CPT 的输入是大规模时序语料（如 Monash、M-datasets、行业数据），不依赖 instruction-response 格式。

**MVP 核心目标（Phase 4 实现）**：next-token prediction on discrete time-series tokens

其他 CPT 任务（Research Extension，非 MVP 范围）：
- Masked span reconstruction
- Time-text contrastive alignment（series-to-summary 对齐）
- Context-conditioned continuation

**`context-conditioned continuation`** 定义：给定领域 tag（如 `[ENERGY]`）和部分历史，预测序列延续，是 NTP 的一种结构化变体，可通过 domain tag prefix 实现。

输入可以包括：

- 离散化后的数值 token（DiscreteTokenRepresentation 输出）
- 变量名、单位、频率、领域标签（作为 prefix token）
- 数据质量描述，例如缺失率、采样频率（可选 context）

核心接口：

```python
class CPTTrainer(BaseTrainer):
    def build_dataset(self) -> "CPTDataset": ...
    def build_objective(self) -> "Objective": ...
    def train_step(self, batch: dict) -> dict: ...
    def save_checkpoint(self, path: str) -> None: ...
```

研究者插拔点（按优先级）：

1. 时序 tokenizer（离散化粒度：`num_bins`、`tokenization_strategy`）
2. domain tag 和 context 模板（领域信息编码方式）
3. curriculum 策略（短序列→长序列、领域均衡采样）
4. alignment objective（Research Extension）
5. backbone 和 PEFT 策略

### 6.2 SFT：单轮指令微调

SFT 的目标是训练模型在单轮交互中理解 forecast instruction，并输出可解析、可评测的预测结果。

推荐样本格式（`context_length=3`，`prediction_length=24` 为示例，实际值由 recipe 控制）：

```json
{
  "input": {
    "context_values": [[1.0], [1.2], [1.1]],
    "timestamps": ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"],
    "metadata": {"domain": "energy", "freq": "1H", "unit": "MW", "channel": "load"},
    "future_known_context": {"hour_of_day": [3, 4, 5, ...], "is_holiday": [false, false, false, ...]},
    "instruction": "Predict the next 24 steps. Return JSON: {\"forecast\": [v1, ..., v24]}"
  },
  "output": {
    "forecast": ["<v1>", "...", "<v24>"],
    "confidence": {"type": "quantile", "q10": ["<q10_1>", "..."], "q90": ["<q90_1>", "..."]},
    "rationale": "（可选）The recent trend is upward with daily seasonality."
  }
}
```

**关于 `rationale` 字段**：SFT 样本中 `rationale` 为可选字段，由 recipe 中的 `output_schema` 控制是否包含。包含 rationale 会增加生成长度和训练时间，但可为 RLVR 的 `ReasoningReward` 提供训练信号。MVP 阶段 `rationale` 默认关闭。

训练策略（仅训练期行为）：

- output-only causal LM loss（只在 output tokens 上计算 NLL，input tokens 不参与梯度）
- forecast head supervised loss（适用于路线 C：NumericalPatch + projector）
- LoRA / QLoRA fine-tuning

**注意**：format-constrained decoding 属于**推理期**策略（强制 LLM 按特定格式输出），不是训练策略，在 SFT 训练中不涉及。推理期约束由 `inference` 配置段控制。

核心接口：

```python
class SFTTrainer(BaseTrainer):
    def build_dataset(self) -> "SFTDataset": ...
    def apply_peft(self) -> None: ...
    def compute_loss(self, batch: dict) -> dict: ...
    def evaluate(self) -> dict: ...
```

研究者插拔点：

- instruction template（prompt 格式设计）
- output schema（JSON / 纯数组 / Markdown）
- representation mix（哪些 Representation 组件）
- parser（对应 output schema 的 Parser）
- PEFT 策略（LoRA rank、target_modules、quantization）
- supervised loss 组合（LM only / LM + forecast head）

### 6.3 RLVR：可验证推理能力训练

RLVR 的目标是在 SFT checkpoint 基础上，让模型通过可验证 reward 学会更可靠的预测推理。这里的 RLVR 是**单轮 prompt-response 训练**，不涉及工具调用环境或多轮 agent 状态。

单轮 RLVR 流程：

```text
Rollout Dataset（历史窗口 + 任务指令，无 label）
      ↓ 推理引擎（vLLM / HF generate）× num_generations
多条 Response（推理摘要 + 结构化预测）
      ↓ Parser → ForecastResult
      ↓ Verifier（AccuracyReward + FormatReward + CalibrationReward）
RewardResult（per-response scalar）
      ↓ GRPO / PPO update
Updated Policy
```

核心 reward（按可实现难度排序）：

| Reward | 难度 | 说明 |
|---|---|---|
| FormatReward | 低 | 规则检查，0/1 信号 |
| AccuracyReward | 低 | 与真实 label 比较，连续信号 |
| CalibrationReward | 中 | 需要模型输出分位数 |
| ReasoningReward | 高（实验性）| 需要文本解析 |

核心接口：

```python
class RLVRTrainer(BaseTrainer):
    def generate(self, batch: dict) -> list["ModelOutput"]: ...
    def verify(self, outputs: list["ModelOutput"], batch: dict) -> list["RewardResult"]: ...
    def update_policy(self, outputs: list["ModelOutput"], rewards: list["RewardResult"]) -> dict: ...
```

RLVR 后端抽象为 adapter，**显式暴露推理引擎选项**：

```python
class RLBackendAdapter:
    def __init__(
        self,
        algorithm: str = "grpo",            # "grpo" | "ppo" | "dpo"
        inference_engine: str = "hf",       # "hf" | "vllm"
        num_generations: int = 8,
    ): ...
    def rollout(self, model, batch: dict) -> list: ...
    def step(self, rollouts: list, rewards: list) -> dict: ...
```

#### 计算开销说明

GRPO 每个训练 step 需要对同一 prompt 采样 `num_generations`（默认 8）条 response。对时序预测任务（预测 96 步 ≈ 96 个输出 token），单卡 HF generate 模式的吞吐约为 SFT 的 1/6～1/4。**强烈建议**使用 vLLM 作为 rollout 推理引擎：

```yaml
training:
  trainer: rlvr
  backend:
    algorithm: grpo
    inference_engine: vllm      # 推荐，rollout 吞吐提升 4-8x
    num_generations: 8
    vllm_config:
      gpu_memory_utilization: 0.4   # 留余量给 training forward
      tensor_parallel_size: 1
```

无 vLLM 时，`inference_engine: hf` 模式可运行但速度慢，建议仅在 debug 和小规模实验中使用。

第一版优先支持轻量 GRPO + vLLM adapter，不重写完整 RL 基础设施。

#### RLVR 的 Rollout Dataset

RLVR 使用专用的 rollout dataset，与 SFT dataset 格式不同——**不包含 output 字段**（output 由模型在线生成），但包含 label 供 Verifier 计算 reward。recipe 中需明确指定：

```yaml
data:
  rollout_dataset:
    source: sft_train       # 复用 SFT 训练集的 prompt（无 output）
    label_field: future_unknown_window   # 用于 reward 计算的真实 label
```

---

## 7. Recipe Schema

### 7.1 完整 SFT Recipe 示例

```yaml
experiment:
  name: qwen_ts_sft_etth1
  seed: 42
  task: forecasting
  stage: sft

data:
  reader:
    name: ett
    path: ./datasets/ETT-small/ETTh1.csv
    target_channels: [OT]           # 预测目标列
    covariate_channels: [HUFL, HULL, MUFL, MULL, LUFL, LULL]
  split:
    type: timestamp                 # 首选 timestamp 模式
    train_end:  "2017-06-30 23:00"
    val_end:    "2017-10-31 23:00"
    test_end:   "2018-02-28 23:00"
  window:
    context_length: 512
    prediction_length: 96
    stride: 96                      # 默认非重叠，与 prediction_length 相等
    multivariate_mode: channel_independent
  covariates:
    future_known: [hour_of_day, day_of_week, is_holiday]
    future_unknown: []
  leakage_policy:
    enforce_cutoff: true
    normalize_scope: train_only
    derived_context_scope: past_only

representation:
  name: hybrid
  components:
    - name: statistics
      features: [mean, std, trend_strength, seasonality_strength, acf_lag1]
    - name: textual_summary
      template: default_ts_summary_v1
    - name: context
      include_domain: true
      include_calendar: true

model:
  backbone:
    type: hf_causal_lm
    name: Qwen/Qwen2.5-1.5B
  bridge:
    type: text_concat               # TextualSummary + Statistics → 文本拼接，无额外参数
  head:
    type: text_generation
    output_schema: forecast_json_v1
  peft:
    method: lora
    r: 16
    alpha: 32
    target_modules: [q_proj, v_proj]

training:
  trainer: sft
  objective:
    name: output_only_causal_lm
  epochs: 10
  batch_size: 32
  lr: 2.0e-4
  scheduler: cosine
  warmup_ratio: 0.05
  checkpoint_dir: ./checkpoints/qwen_ts_sft_etth1

inference:
  parser: forecast_json_v1
  fallback_strategy: last_value
  constrained_decoding: false       # 推理期格式约束，训练期无关

evaluation:
  protocols: [standard, rolling]
  metrics: [mae, mse, smape, mase, parse_success_rate, format_valid_rate]
  save_predictions: true

trace:
  save_recipe: true
  save_prompt: true
  save_response: true
  save_parsed: true
  save_artifacts: true
```

### 7.2 RLVR Recipe 示例（在 SFT Recipe 基础上替换 training + data 段）

```yaml
experiment:
  name: qwen_ts_rlvr_etth1
  seed: 42
  task: forecasting
  stage: rlvr

# data 段需额外指定 rollout_dataset
data:
  reader:
    name: ett
    path: ./datasets/ETT-small/ETTh1.csv
    target_channels: [OT]
  split:
    type: timestamp
    train_end:  "2017-06-30 23:00"
    val_end:    "2017-10-31 23:00"
    test_end:   "2018-02-28 23:00"
  window:
    context_length: 512
    prediction_length: 96
    stride: 96
    multivariate_mode: channel_independent
  rollout_dataset:
    source: sft_train               # 从 SFT 训练集抽取 rollout prompt
    label_field: future_unknown_window
    max_rollout_samples: 5000       # 限制每 epoch rollout 样本数

# representation / model 与 SFT recipe 保持一致（省略）

training:
  trainer: rlvr
  init_checkpoint: ./checkpoints/qwen_ts_sft_etth1/best
  backend:
    algorithm: grpo
    inference_engine: vllm
    num_generations: 8
    vllm_config:
      gpu_memory_utilization: 0.4
  rewards:
    - name: format
      weight: 0.3
    - name: accuracy
      metric: smape
      weight: 0.5
    - name: calibration
      metric: coverage
      weight: 0.2
    # ReasoningReward 为实验性可选，默认关闭
    # - name: reasoning
    #   checks: [horizon_length_check, no_future_timestamp, channel_name_validity]
    #   weight: 0.1
  epochs: 3
  batch_size: 16
  lr: 5.0e-6
  checkpoint_dir: ./checkpoints/qwen_ts_rlvr_etth1
```

---

## 8. 外层 API

### 8.1 Python API

```python
from castfactory import Experiment

exp = Experiment.from_recipe("recipes/sft/qwen_ts_sft_etth1.yaml")
exp.fit()
metrics = exp.evaluate()
exp.report()
```

单轮预测：

```python
result = exp.predict({
    "context_values": values,        # shape: (T, C)
    "timestamps": timestamps,
    "instruction": "Predict the next 96 steps. Return JSON: {\"forecast\": [...]}",
    "metadata": {"domain": "energy", "freq": "1H"}
})
# result.point_forecast: np.ndarray, shape (96, C)
# result.parse_success: bool
# result.fallback_used: bool
```

### 8.2 CLI

```bash
castfactory run   recipes/sft/qwen_ts_sft_etth1.yaml
castfactory eval  runs/qwen_ts_sft_etth1
castfactory report runs/qwen_ts_sft_etth1
```

### 8.3 Registry API

**注意**：所有注册调用必须在 `Experiment.from_recipe()` 之前完成，且在主进程中执行（不可在 DataLoader worker 进程中注册）。

```python
from castfactory import registry

registry.register_reader("my_reader", MyReader)
registry.register_representation("my_repr", MyRepresentation)
registry.register_backbone("my_backbone", MyBackbone)
registry.register_bridge("my_bridge", MyBridge)
registry.register_head("my_head", MyHead)
registry.register_trainer("my_trainer", MyTrainer)
registry.register_parser("my_parser", MyForecastParser)
registry.register_metric("my_metric", my_metric_fn)
registry.register_reward("my_reward", MyRewardFunction)
```

---

## 9. 扩展点设计

| 扩展点 | 研究问题 | 注册方法 |
|---|---|---|
| Reader | 如何接入新数据集或行业数据 | `register_reader` |
| Splitter | 如何设计更严格的时间切分 | `register_splitter` |
| Representation | 时序数值、统计、文本、上下文如何组合 | `register_representation` |
| Tokenizer | 连续值如何离散化（num_bins、strategy） | `register_tokenizer` |
| Backbone | 使用哪类 LLM 或 TS foundation model | `register_backbone` |
| Bridge | 时序表征如何注入 LLM | `register_bridge` |
| Head | 输出点预测、分位数、分布还是结构化文本 | `register_head` |
| Parser | 如何从 LLM 输出中稳定解析 forecast | `register_parser` |
| Objective | 如何组合 forecasting、LM、alignment loss | `register_objective` |
| Reward | RLVR 如何定义可验证反馈 | `register_reward` |
| Metric | 如何评估精度、校准、格式和推理质量 | `register_metric` |
| Protocol | 如何做 zero-shot、rolling、cross-domain 测试 | `register_protocol` |

---

## 10. MVP 路线图

### Phase 0：Package Scaffold（1 周）

目标：建立最小可运行开源库骨架。

- `pyproject.toml` 和 package layout
- `Experiment`、`Recipe` parser、`Registry`
- `TSRecord`、`ForecastSample`、`ForecastResult`
- 最小 CLI：`castfactory run`
- 单元测试框架（pytest）
- CI/CD 配置（GitHub Actions）

### Phase 1：Leakage-Safe Data Layer（2 周）

目标：把时间序列实验中最容易出错的数据流程固定下来。

- CSV / Parquet reader
- TimestampSplitter（默认）、RatioSplitter（需显式启用）
- Rolling window builder（`stride` 默认等于 `prediction_length`）
- Train-only normalization（fit_scope 强制声明）
- Future-known / future-unknown covariate schema
- LeakageChecker（检查 transform、representation、derived context）
- ETTh1 / ETTm1 example recipe
- 数据层单元测试（含泄漏检测测试）

### Phase 2a：Representation + SFT Data（2 周）

目标：打通 ForecastSample → ModelInput → SFTDataset 的数据流。

- TextualSummaryRepresentation
- StatisticsRepresentation（单变量 + 多变量逐通道模式）
- ContextRepresentation
- HybridRepresentation
- SFTDataset（ForecastSample → instruction-following JSON）
- JSONForecastParser（核心 schema + fallback 策略）
- ArrayForecastParser
- Parser 单元测试（覆盖 parse failure 各 fallback 路径）

### Phase 2b：SFT Training + First Baseline（2 周）

目标：跑出第一个有科研价值的可复现 baseline。

- HuggingFace causal LM adapter（Qwen2.5-1.5B）
- text_concat bridge
- text_generation head
- SFTTrainer（output-only causal LM loss）
- LoRA / QLoRA via PEFT
- **Baseline**：Qwen2.5-1.5B LoRA SFT on ETTh1，预测 96 / 192 / 336 / 720 步

### Phase 3：Evaluation and Trace（2 周）

目标：让结果可复现、可比较、可写进论文。

- StandardEval、RollingEval、ZeroShotEval
- 点预测指标：MAE / MSE / SMAPE / MASE
- LLM 指标：ParseSuccessRate、FormatValidRate（一等公民）
- Trace：`predictions.parquet`、`metrics.json`、`prompts.jsonl`、`responses.jsonl`、`parsed.jsonl`、`errors.jsonl`
- `leaderboard.md` 自动生成
- `report.md` 自动生成

### Phase 4：CPT Path（3 周）

目标：支持时序语言预训练。

- Mean-scale quantile tokenizer（`num_bins` 可配置）
- DiscreteTokenRepresentation
- CPTDataset（context packing、domain tag prefix）
- CPTTrainer（NTP objective）
- 大规模时序语料接入（Monash ≥ 20 子集）
- CPT checkpoint → SFT checkpoint 的传递机制
- CPT → SFT 流水线 recipe 示例

### Phase 5：RLVR Path（3 周）

目标：支持单轮可验证推理能力训练。

- RLVRDataset（rollout prompt 构造，无 output 字段）
- RLVRTrainer
- RewardRegistry（支持运行时注册）
- AccuracyReward、CalibrationReward、FormatReward、CompositeReward
- ReasoningReward（实验性，仅 `horizon_length_check`、`no_future_timestamp`）
- GRPO backend（轻量实现或 verl adapter）
- vLLM inference engine adapter
- SFT checkpoint → RLVR checkpoint 的传递机制

### Phase 6：Foundation Model Adapter（2 周）

目标：把 Chronos、TimesFM、Moirai 等作为强 baseline 和统一评测对象。

- Chronos adapter（zero-shot + fine-tune 模式）
- TimesFM adapter
- Moirai adapter
- Zero-shot evaluation recipe
- 与 SFT / RLVR 方法共用同一 evaluation harness

---

## 11. 第一版非目标

为了保持 MVP 清晰，第一版不做：

- 大规模 model zoo（不复制 Time-Series-Library 路线）
- 多轮 agent environment 和工具调用工作流
- AutoML 或自动调参平台
- Web UI
- 重写完整 verl（通过 adapter 对接）
- 一次性覆盖所有公开 benchmark（MVP 覆盖 ETTh1/h2/m1/m2、Weather、Traffic）
- ReasoningReward 的高级语义规则（NLP 解析类）

这些方向可以作为未来扩展，但不应影响第一版的核心：CPT / SFT / RLVR 三阶段、严格防泄漏数据层、稳定 Parser、可复现实验 trace。

---

## 12. 与现有生态的关系

**CastFactory 与 Time-Series-Library 的关系**：

- Time-Series-Library 更偏 DL 模型统一训练和评测（random init + scratch train）。
- CastFactory 更偏 LLM-driven forecasting 的 recipe、表征、指令微调、RLVR 和 trace（pretrained checkpoint + CPT/SFT/RLVR）。
- CastFactory 可以复用 Time-Series-Library 中成熟的评测数据集，但不复制其 model zoo 路线。

**CastFactory 与 Chronos / TimesFM / Moirai 的关系**：

- 这些模型应作为 foundation model adapter 接入（Phase 6）。
- CastFactory 不替代它们，而是提供统一评测、微调和对比环境。

**CastFactory 与 verl / TRL 的关系**：

- CastFactory 定义 forecasting-specific prompt、reward、verifier、parser、metric 和 trace。
- verl / TRL 可作为 RL backend，通过 `RLBackendAdapter` 对接。
- 第一版通过 adapter 对接，而不是重写底层 RL 系统。

**CastFactory 与 vLLM 的关系**：

- vLLM 作为 RLVR rollout 阶段的推理引擎，通过 `inference_engine: vllm` 配置项启用。
- SFT 训练阶段不依赖 vLLM，仅 RLVR rollout 时需要。

---

## 13. 总结

CastFactory 的核心定位可以概括为：

> A recipe-centric training and evaluation framework for CPT, SFT, and RLVR of LLM-driven time series forecasting models, with leakage-safe data pipelines, stable forecast parsing, and full experiment traceability.

它的差异化不在于"接入更多模型"，而在于提供一套严谨、可扩展、可复现的研究框架，让研究者能够系统比较：

- 不同时间序列表征方式（patch / discrete token / text / hybrid）
- 不同 LLM backbone
- 不同 instruction template 和 output schema
- 不同 CPT 数据配比与 objective
- 不同 SFT 监督信号和 PEFT 策略
- 不同 RLVR reward 设计
- 不同评测协议下的泛化、校准、格式遵循和推理可靠性

**实施优先级**：第一版按 Phase 0 → 1 → 2a → 2b → 3 顺序推进，先打通 SFT baseline 并建立可信的评测体系，再补 CPT（Phase 4）和 RLVR（Phase 5）。只要 recipe、data safety、Parser 和 trace 的底座设计正确，后续模型、数据集和训练范式都可以稳定扩展。

---

*文档版本：v0.2 | 修订日期：2026-04-29*
