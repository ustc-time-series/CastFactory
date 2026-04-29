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


if __name__ == "__main__":
    unittest.main()
