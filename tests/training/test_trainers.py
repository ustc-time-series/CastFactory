import tempfile
import unittest


class TrainerTests(unittest.TestCase):
    def test_base_trainer_creates_checkpoint_directory(self):
        from castfactory.training import BaseTrainer

        with tempfile.TemporaryDirectory() as tmp:
            trainer = BaseTrainer(checkpoint_dir=f"{tmp}/checkpoints")

            self.assertTrue(trainer.checkpoint_dir.exists())

    def test_sft_trainer_requires_train_dataset_for_fit(self):
        from castfactory.training import SFTTrainer

        with tempfile.TemporaryDirectory() as tmp:
            trainer = SFTTrainer(train_dataset=None, checkpoint_dir=f"{tmp}/sft")

            with self.assertRaisesRegex(ValueError, "train_dataset"):
                trainer.fit()

    def test_sft_trainer_delegates_to_backend(self):
        from castfactory.training import SFTTrainer

        class Backend:
            def __init__(self):
                self.received = None

            def fit(self, **kwargs):
                self.received = kwargs
                return {"status": "trained", "steps": 3}

        backend = Backend()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SFTTrainer(
                train_dataset=[{"input": "x", "output": "y"}],
                checkpoint_dir=f"{tmp}/sft",
                backend=backend,
                train_args={"max_steps": 3},
            )

            result = trainer.fit()

        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["steps"], 3)
        self.assertEqual(backend.received["train_args"], {"max_steps": 3})
        self.assertEqual(len(backend.received["train_dataset"]), 1)

    def test_transformers_sft_backend_tokenizes_prompt_and_output(self):
        from castfactory.training import TransformersSFTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            eos_token_id = None
            pad_token = None

            def __call__(self, text, truncation, max_length=None):
                ids = [ord(char) for char in text]
                input_ids = ids[:max_length] if max_length else ids
                return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

        backend = TransformersSFTBackend(
            tokenizer=FakeTokenizer(),
            max_prompt_length=32,
            max_response_length=32,
            max_length=64,
        )

        rows = backend.prepare_dataset([{"input": "Q:", "output": "A"}])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_ids"], [81, 58, 65, 60, 101, 111, 115, 62])
        self.assertEqual(rows[0]["labels"][:2], [-100, -100])
        self.assertEqual(rows[0]["labels"][2], 65)

    def test_transformers_sft_backend_uses_prompt_and_response_token_budgets(self):
        from castfactory.training import TransformersSFTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            eos_token_id = 0
            pad_token = None

            def __call__(self, text, truncation, max_length=None):
                ids = [ord(char) for char in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        backend = TransformersSFTBackend(
            tokenizer=FakeTokenizer(),
            max_prompt_length=4,
            max_response_length=4,
            max_length=8,
            truncation_side="left",
        )

        rows = backend.prepare_dataset([{"input": "abcdef", "output": "WXYZ"}])

        self.assertEqual(rows[0]["input_ids"], [99, 100, 101, 102, 87, 88, 89, 0])
        self.assertEqual(rows[0]["labels"], [-100, -100, -100, -100, 87, 88, 89, 0])

    def test_transformers_sft_backend_rejects_response_overflow(self):
        from castfactory.training import TransformersSFTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            eos_token_id = 0
            pad_token = None

            def __call__(self, text, truncation, max_length=None):
                ids = [ord(char) for char in text]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

        backend = TransformersSFTBackend(
            tokenizer=FakeTokenizer(),
            max_prompt_length=4,
            max_response_length=8,
            max_length=4,
        )

        with self.assertRaisesRegex(ValueError, "response tokens exceed max_length"):
            backend.prepare_dataset([{"input": "ab", "output": "WXYZ"}])

    def test_transformers_sft_backend_sets_pad_token_and_passes_collator(self):
        from castfactory.training import TransformersSFTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = None

            def __call__(self, text, truncation, max_length):
                return {"input_ids": [1, 2], "attention_mask": [1, 1]}

        class FakeCollator:
            def __init__(self, tokenizer, model=None, padding=True):
                self.tokenizer = tokenizer
                self.model = model
                self.padding = padding

        class FakeTrainingArgs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeTrainer:
            received_collator = None

            def __init__(self, model, args, train_dataset, data_collator):
                FakeTrainer.received_collator = data_collator

            def train(self):
                return type("Result", (), {"metrics": {"loss": 0.0}})()

        tokenizer = FakeTokenizer()
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            trainer_cls=FakeTrainer,
            training_args_cls=FakeTrainingArgs,
            data_collator_cls=FakeCollator,
        )
        result = backend.fit([{"input": "x", "output": "y"}], "/tmp/out", model=object())

        self.assertEqual(result["status"], "trained")
        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertIs(FakeTrainer.received_collator.tokenizer, tokenizer)

    def test_transformers_sft_backend_saves_tokenizer_with_model_checkpoint(self):
        from castfactory.training import TransformersSFTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = None
            saved_path = None

            def __call__(self, text, truncation, max_length=None):
                return {"input_ids": [1], "attention_mask": [1]}

            def save_pretrained(self, path):
                self.saved_path = path

        class FakeCollator:
            def __init__(self, tokenizer, model=None, padding=True):
                pass

        class FakeTrainingArgs:
            def __init__(self, **kwargs):
                pass

        class FakeTrainer:
            def __init__(self, model, args, train_dataset, data_collator):
                pass

            def train(self):
                return type("Result", (), {"metrics": {}})()

            def save_model(self, path):
                self.saved_model_path = path

        tokenizer = FakeTokenizer()
        backend = TransformersSFTBackend(
            tokenizer=tokenizer,
            trainer_cls=FakeTrainer,
            training_args_cls=FakeTrainingArgs,
            data_collator_cls=FakeCollator,
        )
        with tempfile.TemporaryDirectory() as tmp:
            backend.fit([{"input": "x", "output": "y"}], tmp, model=object())

            self.assertEqual(tokenizer.saved_path, tmp)

    def test_transformers_sft_backend_reports_missing_accelerate_dependency(self):
        from castfactory.training import TransformersSFTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = None

            def __call__(self, text, truncation, max_length=None):
                return {"input_ids": [1], "attention_mask": [1]}

        class FakeCollator:
            def __init__(self, tokenizer, model=None, padding=True):
                pass

        class FakeTrainingArgs:
            def __init__(self, **kwargs):
                raise ImportError("Using the Trainer with PyTorch requires accelerate>=0.26.0")

        class FakeTrainer:
            def __init__(self, model, args, train_dataset, data_collator):
                pass

        backend = TransformersSFTBackend(
            tokenizer=FakeTokenizer(),
            trainer_cls=FakeTrainer,
            training_args_cls=FakeTrainingArgs,
            data_collator_cls=FakeCollator,
        )

        with self.assertRaisesRegex(ImportError, r"castfactory\[train\].*accelerate>=0.26.0"):
            backend.fit([{"input": "x", "output": "y"}], "/tmp/out", model=object())

    def test_transformers_sft_backend_strips_metadata_before_collation(self):
        from castfactory.training import TransformersSFTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = None

            def __call__(self, text, truncation, max_length=None):
                return {"input_ids": [1], "attention_mask": [1]}

        class StrictCollator:
            def __init__(self, tokenizer, model=None, padding=True):
                pass

            def __call__(self, features):
                for feature in features:
                    if "metadata" in feature:
                        raise AssertionError("metadata leaked into training collator")
                return {}

        class FakeTrainingArgs:
            def __init__(self, **kwargs):
                pass

        class FakeTrainer:
            def __init__(self, model, args, train_dataset, data_collator):
                self.train_dataset = train_dataset
                self.data_collator = data_collator

            def train(self):
                self.data_collator(list(self.train_dataset))
                return type("Result", (), {"metrics": {}})()

        backend = TransformersSFTBackend(
            tokenizer=FakeTokenizer(),
            trainer_cls=FakeTrainer,
            training_args_cls=FakeTrainingArgs,
            data_collator_cls=StrictCollator,
        )

        result = backend.fit(
            [{"input": "x", "output": "y", "metadata": {"sample_id": "row-1"}}],
            "/tmp/out",
            model=object(),
        )

        self.assertEqual(result["status"], "trained")

    def test_cpt_dataset_builds_text_stream_from_forecast_samples(self):
        import numpy as np
        import pandas as pd

        from castfactory.data.records import ForecastSample, TSRecord
        from castfactory.training import CPTDataset

        observed = TSRecord(
            values=np.array([[1.0], [2.0]]),
            timestamps=pd.DatetimeIndex(["2022-01-01 00:00", "2022-01-01 01:00"]),
            channel_names=["OT"],
            target_channels=["OT"],
            covariate_channels=[],
            static_context={"domain": "energy", "unit": "MW"},
            metadata={},
        )
        future = TSRecord(
            values=np.array([[3.0]]),
            timestamps=pd.DatetimeIndex(["2022-01-01 02:00"]),
            channel_names=["OT"],
            target_channels=["OT"],
            covariate_channels=[],
            static_context={},
            metadata={},
        )
        sample = ForecastSample(
            observed_window=observed,
            future_known_window=None,
            future_unknown_window=future,
            cutoff_time=pd.Timestamp("2022-01-01 01:00"),
            prediction_length=1,
            metadata={"sample_id": "row-1"},
        )

        dataset = CPTDataset([sample])
        row = dataset[0]

        self.assertIn("<domain> energy", row["text"])
        self.assertIn("<unit> MW", row["text"])
        self.assertIn("<channel> OT", row["text"])
        self.assertIn("2022-01-01 00:00:00=1.0", row["text"])
        self.assertEqual(row["metadata"]["sample_id"], "row-1")

    def test_transformers_cpt_backend_uses_causal_lm_labels(self):
        from castfactory.training import TransformersCPTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = None

            def __call__(self, text, truncation=False, max_length=None):
                return {"input_ids": [ord(char) for char in text]}

        backend = TransformersCPTBackend(tokenizer=FakeTokenizer(), max_length=4)

        rows = backend.prepare_dataset([{"text": "abcdef", "metadata": {"id": 1}}])

        self.assertEqual(rows[0]["input_ids"], [97, 98, 99, 100])
        self.assertEqual(rows[0]["labels"], [97, 98, 99, 100])
        self.assertEqual(rows[0]["metadata"], {"id": 1})

    def test_transformers_cpt_backend_strips_metadata_before_collation(self):
        from castfactory.training import TransformersCPTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = None

            def __call__(self, text, truncation=False, max_length=None):
                return {"input_ids": [1, 2]}

        class StrictCollator:
            def __init__(self, tokenizer, model=None, padding=True):
                pass

            def __call__(self, features):
                for feature in features:
                    if "metadata" in feature:
                        raise AssertionError("metadata leaked into training collator")
                return {}

        class FakeTrainingArgs:
            def __init__(self, **kwargs):
                pass

        class FakeTrainer:
            def __init__(self, model, args, train_dataset, data_collator):
                self.train_dataset = train_dataset
                self.data_collator = data_collator

            def train(self):
                self.data_collator(list(self.train_dataset))
                return type("Result", (), {"metrics": {}})()

        backend = TransformersCPTBackend(
            tokenizer=FakeTokenizer(),
            trainer_cls=FakeTrainer,
            training_args_cls=FakeTrainingArgs,
            data_collator_cls=StrictCollator,
        )

        result = backend.fit(
            [{"text": "series", "metadata": {"sample_id": "row-1"}}],
            "/tmp/out",
            model=object(),
        )

        self.assertEqual(result["status"], "trained")

    def test_transformers_cpt_backend_reports_missing_accelerate_dependency(self):
        from castfactory.training import TransformersCPTBackend

        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = None

            def __call__(self, text, truncation=False, max_length=None):
                return {"input_ids": [1]}

        class FakeCollator:
            def __init__(self, tokenizer, model=None, padding=True):
                pass

        class FakeTrainingArgs:
            def __init__(self, **kwargs):
                raise ImportError("Using the Trainer with PyTorch requires accelerate>=0.26.0")

        class FakeTrainer:
            def __init__(self, model, args, train_dataset, data_collator):
                pass

        backend = TransformersCPTBackend(
            tokenizer=FakeTokenizer(),
            trainer_cls=FakeTrainer,
            training_args_cls=FakeTrainingArgs,
            data_collator_cls=FakeCollator,
        )

        with self.assertRaisesRegex(ImportError, r"castfactory\[train\].*accelerate>=0.26.0"):
            backend.fit([{"text": "series"}], "/tmp/out", model=object())

    def test_cpt_trainer_delegates_to_backend(self):
        from castfactory.training import CPTTrainer

        class Backend:
            def __init__(self):
                self.received = None

            def fit(self, **kwargs):
                self.received = kwargs
                return {"status": "trained", "num_examples": len(kwargs["train_dataset"])}

        backend = Backend()
        trainer = CPTTrainer(
            train_dataset=[{"text": "x"}],
            checkpoint_dir="/tmp/cpt",
            backend=backend,
            train_args={"max_steps": 1},
        )

        result = trainer.fit()

        self.assertEqual(result["status"], "trained")
        self.assertEqual(result["num_examples"], 1)
        self.assertEqual(backend.received["train_args"], {"max_steps": 1})

    def test_rlvr_dataset_and_trainer_delegate_rollouts(self):
        from castfactory.training import RLVRDataset, RLVRTrainer

        class Backend:
            def fit(self, dataset, rewards):
                return {
                    "status": "prepared",
                    "num_rows": len(dataset),
                    "num_rewards": len(rewards),
                }

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
        trainer = RLVRTrainer(dataset=dataset, rewards=[object()], backend=Backend())

        result = trainer.fit()

        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["num_rows"], 1)
        self.assertEqual(result["num_rewards"], 1)

    def test_rlvr_dataset_preserves_structured_rollout_rows(self):
        from castfactory.training import RLVRDataset

        dataset = RLVRDataset(
            [
                {
                    "prompt": "forecast",
                    "label": [[1.0], [2.0]],
                    "prediction_length": 2,
                    "channel_names": ["OT"],
                    "observed_values": [[0.0], [1.0]],
                    "cutoff_time": "2022-01-01 00:00",
                    "sample_id": "row-1",
                }
            ]
        )

        row = dataset[0]

        self.assertEqual(row["sample_id"], "row-1")
        self.assertEqual(dataset.prompts(), ["forecast"])
        self.assertEqual(dataset.rows_for_export()[0]["prediction_length"], 2)

    def test_rlvr_trainer_delegates_dataset_rows_and_rewards_to_backend(self):
        from castfactory.training import RLVRDataset, RLVRTrainer

        class Backend:
            def __init__(self):
                self.received = None

            def fit(self, dataset, rewards):
                self.received = (dataset, rewards)
                return {"status": "prepared", "num_rows": len(dataset)}

        backend = Backend()
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

        trainer = RLVRTrainer(dataset=dataset, rewards=[object()], backend=backend)
        result = trainer.fit()

        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["num_rows"], 1)
        self.assertIs(backend.received[0], dataset)
        self.assertEqual(len(backend.received[1]), 1)


if __name__ == "__main__":
    unittest.main()
