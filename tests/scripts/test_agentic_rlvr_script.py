import shutil
import subprocess
import unittest
from pathlib import Path


class AgenticRLVRScriptTests(unittest.TestCase):
    def setUp(self):
        self.script = Path("scripts/agentic_rlvr.sh")

    def test_agentic_rlvr_script_documents_cast_r1_agent_loop_training(self):
        self.assertTrue(self.script.exists())
        text = self.script.read_text(encoding="utf-8")

        self.assertIn("export CUDA_VISIBLE_DEVICES=0,1,2,3", text)
        self.assertNotIn("conda activate", text)
        self.assertNotIn("CONDA_ENV", text)
        self.assertNotIn("--dry-run", text)
        self.assertNotIn("--skip-verl", text)
        self.assertIn("examples/rlvr/etth1_qwen3_1_7b_agentic_grpo_4gpu.yaml", text)
        self.assertIn("castfactory/dataset/ETTh1/ETTh1.csv", text)
        self.assertIn("rollout.workflow.name=time_series_agent", text)
        self.assertIn("time_series_forecast_agent", text)
        self.assertIn("launch_command.txt", text)
        self.assertIn("bash -lc \"$(cat \"$LAUNCH_FILE\")\"", text)

    def test_agentic_rlvr_script_prepares_then_runs_verl_launch_command(self):
        text = self.script.read_text(encoding="utf-8")

        prepare_index = text.index("python -m castfactory.cli.run")
        run_index = text.index("bash -lc \"$(cat \"$LAUNCH_FILE\")\"")

        self.assertLess(prepare_index, run_index)
        self.assertIn("training.backend.verl.data.return_raw_chat=true", text)
        self.assertIn("training.backend.verl.actor_rollout_ref.rollout.multi_turn.enable=true", text)
        self.assertIn(
            "training.backend.verl.actor_rollout_ref.rollout.agent.default_agent_loop=time_series_forecast_agent",
            text,
        )

    @unittest.skipUnless(shutil.which("bash"), "bash is required for shell script checks")
    def test_agentic_rlvr_script_has_valid_bash_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(self.script)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash is required for shell script execution")
    def test_agentic_rlvr_script_rejects_removed_dry_run_option(self):
        result = subprocess.run(
            ["bash", str(self.script), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Override must use key=value syntax", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
