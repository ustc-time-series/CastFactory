import tempfile
import unittest
import json
import shlex
from pathlib import Path

import yaml


class VerlBackendTests(unittest.TestCase):
    def test_verl_backend_exports_dataset_and_config(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            backend = VerlBackend(
                run_dir=tmp_path,
                training_config={"algorithm": "grpo", "epochs": 1, "batch_size": 2},
                model_config={"model_path": "checkpoint"},
            )

            result = backend.fit(dataset, rewards=[])

            self.assertIn(result["status"], {"prepared", "skipped"})
            self.assertTrue((tmp_path / "rlvr" / "rollout_dataset.jsonl").exists())
            self.assertTrue((tmp_path / "rlvr" / "verl_config.yaml").exists())
            self.assertIn("python -m verl.trainer.main_ppo", result["launch_command"])

    def test_verl_backend_writes_launch_command_file(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = VerlBackend(run_dir=tmp_path).fit(dataset, rewards=[])
            launch_path = tmp_path / "rlvr" / "launch_command.txt"

            self.assertTrue(launch_path.exists())
            self.assertIn("verl.trainer.main_ppo", launch_path.read_text(encoding="utf-8"))
            self.assertEqual(result["launch_command_path"], str(launch_path))

    def test_verl_backend_exports_native_verl_config_schema(self):
        from castfactory.rewards import FormatReward, MSEReward
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            backend = VerlBackend(
                run_dir=tmp_path,
                training_config={
                    "algorithm": "grpo",
                    "epochs": 2,
                    "batch_size": 4,
                    "num_generations": 8,
                    "max_prompt_length": 2048,
                    "max_response_length": 512,
                },
                model_config={"model_path": "checkpoint"},
                config_overrides={
                    "actor_rollout_ref": {"rollout": {"name": "vllm", "gpu_memory_utilization": 0.4}},
                    "trainer": {"n_gpus_per_node": 1},
                },
            )

            backend.fit(
                dataset,
                rewards=[FormatReward(prediction_length=1, num_channels=1), MSEReward(temperature=3.0)],
            )
            config = yaml.safe_load((tmp_path / "rlvr" / "verl_config.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["algorithm"]["adv_estimator"], "grpo")
        self.assertEqual(config["data"]["prompt_key"], "prompt")
        self.assertEqual(config["data"]["train_batch_size"], 4)
        self.assertEqual(config["data"]["max_prompt_length"], 2048)
        self.assertEqual(config["data"]["max_response_length"], 512)
        self.assertEqual(config["actor_rollout_ref"]["model"]["path"], "checkpoint")
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["name"], "vllm")
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["n"], 8)
        self.assertEqual(config["reward"]["custom_reward_function"]["path"], "pkg://castfactory.training.verl_reward_adapter")
        self.assertEqual(config["reward"]["custom_reward_function"]["name"], "compute_score")
        self.assertEqual(
            config["reward"]["custom_reward_function"]["reward_kwargs"]["reward_specs"],
            [
                {"name": "format", "prediction_length": 1, "num_channels": 1},
                {"name": "mse", "temperature": 3.0},
            ],
        )

    def test_verl_backend_formats_reward_specs_as_hydra_dict_list_override(self):
        from castfactory.rewards import FormatReward, MSEReward
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = VerlBackend(run_dir=tmp).fit(
                dataset,
                rewards=[FormatReward(prediction_length=1, num_channels=1), MSEReward(temperature=3.0)],
            )

        reward_specs_arg = next(
            arg
            for arg in shlex.split(result["launch_command"])
            if arg.startswith("+reward.custom_reward_function.reward_kwargs.reward_specs=")
        )
        self.assertIn("[{name:", reward_specs_arg)
        self.assertNotIn('{"name"', reward_specs_arg)

    def test_verl_backend_defaults_to_long_context_budgets(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            VerlBackend(run_dir=tmp_path).fit(dataset, rewards=[])
            config = yaml.safe_load((tmp_path / "rlvr" / "verl_config.yaml").read_text(encoding="utf-8"))

        self.assertEqual(config["data"]["max_prompt_length"], 4096)
        self.assertEqual(config["data"]["max_response_length"], 8192)

    def test_verl_backend_writes_posix_paths_for_server_execution(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = VerlBackend(
                run_dir=tmp_path / "nested" / "run",
                training_config={"val_files": tmp_path / "validation.jsonl"},
            ).fit(dataset, rewards=[])
            config = yaml.safe_load(
                (tmp_path / "nested" / "run" / "rlvr" / "verl_config.yaml").read_text(encoding="utf-8")
            )

        self.assertNotIn("\\", config["data"]["train_files"])
        self.assertNotIn("\\", config["data"]["val_files"])
        self.assertNotIn("\\", config["trainer"]["default_local_dir"])
        self.assertIn("data.train_files=", result["launch_command"])

    def test_verl_backend_expands_relative_local_model_path_for_ray_workers(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            backend = VerlBackend(
                run_dir=tmp_path,
                model_config={"model_path": "./checkpoints/sft"},
            )

            result = backend.fit(dataset, rewards=[])
            config = yaml.safe_load((tmp_path / "rlvr" / "verl_config.yaml").read_text(encoding="utf-8"))

        model_path = config["actor_rollout_ref"]["model"]["path"]
        self.assertTrue(Path(model_path).is_absolute())
        self.assertNotIn("actor_rollout_ref.model.path=./checkpoints/sft", result["launch_command"])

    def test_verl_backend_exports_verl_compatible_dataset_rows(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            VerlBackend(run_dir=tmp_path).fit(dataset, rewards=[])
            exported = json.loads(
                (tmp_path / "rlvr" / "rollout_dataset.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(exported["data_source"], "castfactory_rlvr")
        self.assertEqual(exported["ability"], "forecasting")
        self.assertEqual(exported["prompt"], [{"role": "user", "content": "forecast"}])
        self.assertEqual(exported["reward_model"]["style"], "rule")
        self.assertEqual(exported["reward_model"]["ground_truth"]["label"], [[1.0]])
        self.assertEqual(exported["extra_info"]["prediction_length"], 1)
        self.assertEqual(exported["extra_info"]["channel_names"], ["OT"])

    def test_verl_reward_adapter_compute_score_uses_verl_signature(self):
        from castfactory.training.verl_reward_adapter import compute_score

        ground_truth = {
            "label": [[1.0], [2.0]],
        }
        extra_info = {
            "prediction_length": 2,
            "channel_names": ["OT"],
            "observed_values": [[0.0], [1.0]],
            "cutoff_time": "2022-01-01 00:00",
            "sample_id": "row-1",
        }

        result = compute_score(
            data_source="castfactory_rlvr",
            solution_str=(
                "<think>\nuse recent values\n</think>\n"
                "<answer>\n```\n1.0\n2.0\n```\n</answer>"
            ),
            ground_truth=ground_truth,
            extra_info=extra_info,
            reward_specs=[
                {"name": "format", "prediction_length": 2, "num_channels": 1},
                {"name": "mse", "temperature": 1.0},
            ],
        )

        self.assertEqual(result["parse_success"], True)
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["format"], 1.0)
        self.assertAlmostEqual(result["mse"], 0.5)

    def test_verl_reward_adapter_does_not_return_verl_reserved_extra_info_fields(self):
        from castfactory.training.verl_reward_adapter import compute_score

        result = compute_score(
            data_source="castfactory_rlvr",
            solution_str="<think>\nreason\n</think>\n<answer>\n```\n1.0\n```\n</answer>",
            ground_truth={"label": [[1.0]]},
            extra_info={
                "prediction_length": 1,
                "channel_names": ["OT"],
                "observed_values": [[0.0]],
            },
            reward_specs=[],
        )

        self.assertNotIn("data_source", result)
        self.assertNotIn("reward", result)

    def test_verl_reward_adapter_returns_only_metric_compatible_extra_info(self):
        from castfactory.training.verl_reward_adapter import compute_score

        result = compute_score(
            data_source="castfactory_rlvr",
            solution_str="bad format",
            ground_truth={"label": [[1.0], [2.0]]},
            extra_info={
                "prediction_length": 2,
                "channel_names": ["OT"],
                "observed_values": [[0.0], [1.0]],
            },
            reward_specs=[
                {"name": "format", "prediction_length": 2, "num_channels": 1},
                {"name": "mse", "temperature": 1.0},
            ],
        )

        for value in result.values():
            self.assertIsInstance(value, (bool, int, float))

    def test_verl_reward_adapter_returns_stable_reward_keys_when_format_invalid(self):
        from castfactory.training.verl_reward_adapter import compute_score

        reward_specs = [
            {"name": "format", "prediction_length": 2, "num_channels": 1},
            {"name": "mse", "temperature": 1.0},
        ]
        payload = {
            "data_source": "castfactory_rlvr",
            "ground_truth": {"label": [[1.0], [2.0]]},
            "extra_info": {
                "prediction_length": 2,
                "channel_names": ["OT"],
                "observed_values": [[0.0], [1.0]],
            },
            "reward_specs": reward_specs,
        }

        valid = compute_score(
            solution_str="<think>\nreason\n</think>\n<answer>\n```\n1.0\n2.0\n```\n</answer>",
            **payload,
        )
        invalid = compute_score(solution_str="bad format", **payload)

        self.assertEqual(set(valid), set(invalid))
        self.assertEqual(invalid["mse"], -1.0)

    def test_verl_reward_adapter_short_circuits_when_format_is_invalid(self):
        from castfactory.rewards import FormatReward, MSEReward
        from castfactory.training.verl_reward_adapter import score_response

        row = {
            "label": [[1.0], [2.0]],
            "prediction_length": 2,
            "channel_names": ["OT"],
            "observed_values": [[0.0], [1.0]],
            "cutoff_time": "2022-01-01 00:00",
            "sample_id": "row-1",
        }

        result = score_response(
            response="1.0\n2.0",
            row=row,
            rewards=[
                FormatReward(prediction_length=2, num_channels=1),
                MSEReward(temperature=1.0),
            ],
        )

        self.assertEqual(result["parse_success"], False)
        self.assertEqual(result["reward"], -1.0)
        self.assertEqual(result["details"]["format"], -1.0)
        self.assertEqual(result["details"]["mse"], "skipped")

    def test_verl_backend_soft_fails_without_runtime_but_keeps_launch_artifacts(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            backend = VerlBackend(
                run_dir=tmp_path,
                training_config={"algorithm": "grpo"},
                model_config={"model_path": "checkpoint"},
            )

            result = backend.fit(dataset, rewards=[])

            self.assertEqual(result["status"], "prepared")
            self.assertIn("launch_command", result)
            self.assertTrue(Path(result["dataset_path"]).exists())
            self.assertTrue(Path(result["config_path"]).exists())

    def test_verl_backend_deep_merges_training_backend_verl_overrides(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            backend = VerlBackend(
                run_dir=tmp_path,
                training_config={"algorithm": "grpo", "epochs": 1, "batch_size": 2},
                model_config={"model_path": "checkpoint", "init_checkpoint": "ckpt"},
                config_overrides={
                    "trainer": {"total_epochs": 3},
                    "rollout": {"temperature": 0.8},
                    "custom_reward_function": {"config": {"strict_format": True}},
                },
            )

            backend.fit(dataset, rewards=[])

            config_text = (tmp_path / "rlvr" / "verl_config.yaml").read_text(encoding="utf-8")

        self.assertIn("total_epochs: 3", config_text)
        self.assertIn("train_batch_size: 2", config_text)
        self.assertIn("temperature: 0.8", config_text)
        self.assertIn("strict_format: true", config_text)

    def test_verl_backend_normalizes_castfactory_verl_tree_to_native_paths(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            backend = VerlBackend(
                run_dir=tmp_path,
                config_overrides={
                    "trainer": {
                        "total_epochs": 3,
                        "train_batch_size": 16,
                        "num_generations": 8,
                    },
                    "rollout": {
                        "inference_engine": "vllm",
                        "gpu_memory_utilization": 0.4,
                    },
                },
            )

            backend.fit(dataset, rewards=[])
            config = yaml.safe_load((tmp_path / "rlvr" / "verl_config.yaml").read_text(encoding="utf-8"))

        self.assertNotIn("rollout", config)
        self.assertEqual(config["trainer"]["total_epochs"], 3)
        self.assertEqual(config["data"]["train_batch_size"], 16)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["n"], 8)
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["name"], "vllm")
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"], 0.4)

    def test_verl_backend_execute_runs_launch_command_with_command_runner(self):
        from castfactory.training import RLVRDataset
        from castfactory.training.backends import VerlBackend

        calls = []

        def command_runner(command):
            calls.append(command)
            return {"returncode": 0, "stdout": "trained", "stderr": ""}

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0]],
                    "prediction_length": 1,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = VerlBackend(
                run_dir=tmp,
                execute=True,
                command_runner=command_runner,
            ).fit(dataset, rewards=[])

        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("verl.trainer.main_ppo", calls[0])


if __name__ == "__main__":
    unittest.main()
