from __future__ import annotations

import importlib.util
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml


class VerlBackend:
    def __init__(
        self,
        run_dir: str | Path,
        training_config: Mapping[str, Any] | None = None,
        model_config: Mapping[str, Any] | None = None,
        reward_config: Mapping[str, Any] | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        workflow_config: Mapping[str, Any] | None = None,
        python_executable: str = "python",
        verl_module: str = "verl.trainer.main_ppo",
        execute: bool = False,
        command_runner: Callable[[str], Mapping[str, Any] | subprocess.CompletedProcess] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.training_config = dict(training_config or {})
        self.model_config = dict(model_config or {})
        self.reward_config = dict(reward_config or {})
        self.config_overrides = _normalize_config_overrides(dict(config_overrides or {}))
        self.workflow_config = dict(workflow_config or {})
        self.python_executable = python_executable
        self.verl_module = verl_module
        self.execute = execute
        self.command_runner = command_runner

    def fit(self, dataset, rewards: Iterable[Any]) -> dict:
        rlvr_dir = self.run_dir / "rlvr"
        rlvr_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = rlvr_dir / "rollout_dataset.jsonl"
        config_path = rlvr_dir / "verl_config.yaml"
        agent_loop_config_path = (
            rlvr_dir / "agent_loop_config.yaml" if self._workflow_name() == "time_series_agent" else None
        )

        self._write_dataset(dataset_path, self._rows_for_verl_export(dataset.rows_for_export()))
        if agent_loop_config_path is not None:
            agent_loop_config_path.write_text(
                yaml.safe_dump(self._build_agent_loop_config(), sort_keys=False),
                encoding="utf-8",
            )
        config = self._build_config(
            dataset_path,
            rewards,
            agent_loop_config_path=agent_loop_config_path,
        )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        launch_command = self.build_launch_command(config_path, config=config)
        launch_command_path = rlvr_dir / "launch_command.txt"
        launch_command_path.write_text(launch_command + "\n", encoding="utf-8")
        result = {
            "status": "prepared",
            "reason": "verl runtime is not executed locally",
            "dataset_path": str(dataset_path),
            "config_path": str(config_path),
            "reward_entrypoint": "castfactory.training.verl_reward_adapter:compute_score",
            "launch_command": launch_command,
            "launch_command_path": str(launch_command_path),
        }
        if agent_loop_config_path is not None:
            result["agent_loop_config_path"] = str(agent_loop_config_path)
        if self.execute:
            execution = self._run_command(launch_command)
            result.update(execution)
            result["status"] = "trained" if int(execution.get("returncode", 1)) == 0 else "failed"
            result["reason"] = "verl training command executed"
        elif self._verl_runtime_available():
            result["reason"] = "verl runtime detected; artifacts are ready for server execution"
        return result

    def build_launch_command(self, config_path: str | Path, config: Mapping[str, Any] | None = None) -> str:
        config_path = Path(config_path)
        if config is None and config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        overrides = _hydra_overrides(config or {})
        override_text = " ".join(overrides)
        return f"{self.python_executable} -m {self.verl_module} {override_text}".strip()

    def _write_dataset(self, dataset_path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        with dataset_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _build_config(
        self,
        dataset_path: Path,
        rewards: Iterable[Any],
        *,
        agent_loop_config_path: Path | None = None,
    ) -> dict:
        training = dict(self.training_config)
        model = dict(self.model_config)
        reward_specs = [_reward_to_spec(reward) for reward in rewards]
        algorithm = training.get("algorithm", "grpo")
        rollout_name = training.get("inference_engine", training.get("rollout_name", "vllm"))
        num_generations = int(training.get("num_generations", 1))
        dataset_path_text = _path_for_verl(dataset_path)
        val_files = training.get("val_files")
        base_config = {
            "data": {
                "train_files": dataset_path_text,
                "val_files": _path_for_verl(val_files) if val_files else dataset_path_text,
                "prompt_key": "prompt",
                "reward_fn_key": "data_source",
                "train_batch_size": int(training.get("batch_size", 1)),
                "max_prompt_length": int(training.get("max_prompt_length", 4096)),
                "max_response_length": int(training.get("max_response_length", 8192)),
                "filter_overlong_prompts": bool(training.get("filter_overlong_prompts", False)),
                "truncation": training.get("truncation", "error"),
            },
            "algorithm": {
                "adv_estimator": algorithm,
                "use_kl_in_reward": bool(training.get("use_kl_in_reward", False)),
            },
            "trainer": {
                "total_epochs": training.get("epochs", 1),
                "project_name": training.get("project_name", "castfactory"),
                "experiment_name": training.get("experiment_name", self.run_dir.name),
                "default_local_dir": _path_for_verl(self.run_dir),
            },
            "actor_rollout_ref": {
                "model": {
                    "path": _model_path_for_verl(model.get("model_path") or model.get("path") or ""),
                },
                "rollout": {
                    "name": rollout_name,
                    "n": num_generations,
                },
            },
            "reward": {
                "custom_reward_function": {
                    "path": "pkg://castfactory.training.verl_reward_adapter",
                    "name": "compute_score",
                    "reward_kwargs": {
                        "reward_specs": reward_specs,
                        **dict(self.reward_config),
                    },
                },
                "reward_model": {
                    "enable": False,
                },
            },
        }
        if agent_loop_config_path is not None:
            workflow = self.workflow_config
            agentic_config = {
                "data": {
                    "return_raw_chat": True,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "multi_turn": {
                            "enable": True,
                            "format": workflow.get("tool_parser_format", "hermes"),
                            "max_parallel_calls": int(workflow.get("max_parallel_calls", 5)),
                        },
                        "agent": {
                            "default_agent_loop": "time_series_forecast_agent",
                            "agent_loop_config_path": _path_for_verl(agent_loop_config_path),
                        },
                    },
                },
            }
            base_config = _deep_merge_dicts(base_config, agentic_config)
        return _deep_merge_dicts(base_config, self.config_overrides)

    def _build_agent_loop_config(self) -> list[dict]:
        workflow = self.workflow_config
        return [
            {
                "name": "time_series_forecast_agent",
                "_target_": (
                    "castfactory.training.agentic.time_series_agent_loop."
                    "TimeSeriesForecastAgentLoop"
                ),
                "max_steps": int(workflow.get("max_steps", 3)),
                "max_parallel_calls": int(workflow.get("max_parallel_calls", 5)),
                "tool_parser_format": workflow.get("tool_parser_format", "hermes"),
                "model_service_url": workflow.get("model_service_url", "http://localhost:8994"),
                "prediction_models": list(
                    workflow.get(
                        "prediction_models",
                        ["chronos2", "arima", "patchtst", "itransformer"],
                    )
                ),
                "local_fallback": workflow.get("local_fallback", "arima_then_last_value"),
            }
        ]

    def _workflow_name(self) -> str:
        return str(self.workflow_config.get("name", "single_turn")).lower()

    def _verl_runtime_available(self) -> bool:
        if importlib.util.find_spec("verl") is not None:
            return True
        vendored_root = self.run_dir.parent / "castfactory" / "verl"
        return vendored_root.exists()

    def _rows_for_verl_export(self, rows: Iterable[Mapping[str, Any]]) -> list[dict]:
        exported = []
        for row in rows:
            prompt = row["prompt"]
            if isinstance(prompt, str):
                prompt_messages = [{"role": "user", "content": prompt}]
            else:
                prompt_messages = prompt
            reward_model = row.get("reward_model")
            if reward_model is None:
                reward_model = {
                    "style": "rule",
                    "ground_truth": {
                        "label": row["label"],
                    },
                }
            extra_info = dict(row.get("extra_info", {}) or {})
            extra_info.setdefault("index", row.get("sample_id"))
            extra_info.setdefault("sample_id", row.get("sample_id"))
            extra_info.setdefault("prediction_length", row["prediction_length"])
            extra_info.setdefault("channel_names", row["channel_names"])
            extra_info.setdefault("observed_values", row["observed_values"])
            extra_info.setdefault("cutoff_time", row["cutoff_time"])
            extra_info.setdefault("model_input_text", row.get("model_input_text", ""))
            extra_info.setdefault("metadata", row.get("metadata", {}))
            exported_row = {
                "data_source": row.get("data_source", "castfactory_rlvr"),
                "prompt": prompt_messages,
                "ability": row.get("ability", "forecasting"),
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
            if "agent_name" in row:
                exported_row["agent_name"] = row["agent_name"]
            exported.append(
                exported_row
            )
        return exported

    def _run_command(self, command: str) -> dict:
        runner = self.command_runner or _default_command_runner
        completed = runner(command)
        if isinstance(completed, Mapping):
            return {
                "returncode": int(completed.get("returncode", 0)),
                "stdout": str(completed.get("stdout", "")),
                "stderr": str(completed.get("stderr", "")),
            }
        return {
            "returncode": int(completed.returncode),
            "stdout": completed.stdout or "",
            "stderr": completed.stderr or "",
        }


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _path_for_verl(path: str | Path) -> str:
    return _resolve_local_path(path).as_posix()


def _model_path_for_verl(path: str | Path) -> str:
    if not path:
        return ""
    if _looks_like_local_path(path):
        return _resolve_local_path(path).as_posix()
    return str(path)


def _resolve_local_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _looks_like_local_path(path: str | Path) -> bool:
    if isinstance(path, Path):
        return True
    text = str(path)
    if "://" in text:
        return False
    return (
        text.startswith("./")
        or text.startswith("../")
        or text.startswith("~/")
        or Path(text).is_absolute()
        or "\\" in text
    )


def _normalize_config_overrides(overrides: Mapping[str, Any]) -> dict:
    overrides = dict(overrides)
    legacy = {}
    remaining = dict(overrides)

    trainer = dict(remaining.pop("trainer", {}) or {})
    if "total_epochs" in trainer:
        _deep_set(legacy, ("trainer", "total_epochs"), trainer.pop("total_epochs"))
    if "train_batch_size" in trainer:
        _deep_set(legacy, ("data", "train_batch_size"), trainer.pop("train_batch_size"))
    if "num_generations" in trainer:
        _deep_set(legacy, ("actor_rollout_ref", "rollout", "n"), trainer.pop("num_generations"))
    if trainer:
        _deep_set(legacy, ("trainer",), trainer)

    rollout = dict(remaining.pop("rollout", {}) or {})
    if "inference_engine" in rollout:
        _deep_set(legacy, ("actor_rollout_ref", "rollout", "name"), rollout.pop("inference_engine"))
    if rollout:
        _deep_set(legacy, ("actor_rollout_ref", "rollout"), rollout)

    custom_reward = dict(remaining.pop("custom_reward_function", {}) or {})
    if custom_reward:
        config = custom_reward.pop("config", None)
        if config:
            custom_reward["reward_kwargs"] = _deep_merge_dicts(
                dict(custom_reward.get("reward_kwargs", {})),
                dict(config),
            )
        _deep_set(legacy, ("reward", "custom_reward_function"), custom_reward)

    return _deep_merge_dicts(legacy, remaining)


def _deep_set(target: dict, path: tuple[str, ...], value: Any) -> None:
    current = target
    for part in path[:-1]:
        current = current.setdefault(part, {})
    final = path[-1]
    if isinstance(value, Mapping) and isinstance(current.get(final), dict):
        current[final] = _deep_merge_dicts(current[final], value)
    else:
        current[final] = value


def _reward_to_spec(reward: Any) -> dict:
    if isinstance(reward, Mapping):
        return dict(reward)
    name = getattr(reward, "name", reward.__class__.__name__).lower()
    spec = {"name": name}
    for attr in (
        "prediction_length",
        "num_channels",
        "metric",
        "temperature",
        "lower_key",
        "upper_key",
        "max_horizon_steps",
    ):
        if hasattr(reward, attr):
            value = getattr(reward, attr)
            if value is not None:
                spec[attr] = value
    return spec


def _default_command_runner(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )


def _hydra_overrides(config: Mapping[str, Any]) -> list[str]:
    overrides = []
    for key, value in _flatten_config(config):
        prefix = "+" if key.startswith("reward.custom_reward_function.reward_kwargs") else ""
        overrides.append(f"{prefix}{key}={_format_hydra_value(value)}")
    return overrides


def _flatten_config(config: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    flattened = []
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.extend(_flatten_config(value, path))
        else:
            flattened.append((path, value))
    return flattened


def _format_hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return shlex.quote(_format_hydra_container_value(value))
    return shlex.quote(str(value))


def _format_hydra_container_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(_format_hydra_container_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = (
            f"{_format_hydra_dict_key(key)}:{_format_hydra_container_value(item)}"
            for key, item in value.items()
        )
        return "{" + ",".join(items) + "}"
    return json.dumps(str(value), ensure_ascii=True, separators=(",", ":"))


_HYDRA_SIMPLE_DICT_KEY = re.compile(r"^[A-Za-z0-9_$%*@?./+-]+$")


def _format_hydra_dict_key(key: Any) -> str:
    text = str(key)
    if _HYDRA_SIMPLE_DICT_KEY.fullmatch(text):
        return text
    escaped = text.replace("\\", "\\\\")
    for char in " \t,{}[]:=()":
        escaped = escaped.replace(char, "\\" + char)
    return escaped
