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
            pad_token = None

            def __call__(self, text, truncation, max_length):
                ids = [ord(char) for char in text]
                input_ids = ids[:max_length]
                return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

        backend = TransformersSFTBackend(tokenizer=FakeTokenizer(), max_length=32)

        rows = backend.prepare_dataset([{"input": "Q:", "output": "A"}])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_ids"], [81, 58, 65, 60, 101, 111, 115, 62])
        self.assertEqual(rows[0]["labels"][:2], [-100, -100])
        self.assertEqual(rows[0]["labels"][2], 65)

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

    def test_rlvr_dataset_and_trainer_delegate_rollouts(self):
        from castfactory.training import RLVRDataset, RLVRTrainer

        class Backend:
            def run(self, prompts, rewards):
                return {
                    "status": "trained",
                    "num_prompts": len(prompts),
                    "num_rewards": len(rewards),
                }

        dataset = RLVRDataset([{"prompt": "forecast", "target": [1.0]}])
        trainer = RLVRTrainer(dataset=dataset, rewards=[object()], backend=Backend())

        result = trainer.fit()

        self.assertEqual(result["num_prompts"], 1)
        self.assertEqual(result["num_rewards"], 1)


if __name__ == "__main__":
    unittest.main()
