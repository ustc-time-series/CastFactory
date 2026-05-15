import unittest
from pathlib import Path
import re

import yaml


class ExampleAssetsTests(unittest.TestCase):
    def test_etth1_rlvr_example_shell_script_exists_and_targets_recipe(self):
        script_path = Path("examples/rlvr/run_etth1_qwen_grpo.sh")

        self.assertTrue(script_path.exists())
        content = script_path.read_text(encoding="utf-8")
        self.assertIn("python -m castfactory.cli.run", content)
        self.assertIn("examples/rlvr/etth1_qwen_grpo.yaml", content)

    def test_etth1_rlvr_example_allocates_prompt_and_response_budget(self):
        recipe_path = Path("examples/rlvr/etth1_qwen_grpo.yaml")
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))

        data_config = recipe["training"]["backend"]["verl"]["data"]

        self.assertGreaterEqual(data_config["max_prompt_length"], 4096)
        self.assertGreaterEqual(data_config["max_response_length"], 8192)

    def test_basic_cpt_sft_rlvr_examples_exist_and_chain_checkpoints(self):
        cpt = Path("examples/cpt/etth1_qwen_cpt.yaml")
        sft = Path("examples/sft/etth1_qwen_sft.yaml")
        rlvr = Path("examples/rlvr/etth1_qwen_grpo.yaml")

        self.assertTrue(cpt.exists())
        self.assertTrue(sft.exists())
        self.assertTrue(rlvr.exists())

        cpt_recipe = yaml.safe_load(cpt.read_text(encoding="utf-8"))
        sft_recipe = yaml.safe_load(sft.read_text(encoding="utf-8"))
        rlvr_recipe = yaml.safe_load(rlvr.read_text(encoding="utf-8"))

        self.assertEqual(cpt_recipe["experiment"]["stage"], "cpt")
        self.assertEqual(sft_recipe["experiment"]["stage"], "sft")
        self.assertEqual(rlvr_recipe["experiment"]["stage"], "rlvr")
        self.assertEqual(
            sft_recipe["training"]["init_checkpoint"],
            cpt_recipe["training"]["checkpoint_dir"],
        )
        self.assertEqual(
            rlvr_recipe["training"]["init_checkpoint"],
            sft_recipe["training"]["checkpoint_dir"],
        )

    def test_4gpu_examples_target_qwen3_local_model_without_server_suffix(self):
        expected_model = "/home/zyt/LLM/Qwen3-1.7B"
        recipes = [
            Path("examples/cpt/etth1_qwen3_1_7b_cpt_4gpu.yaml"),
            Path("examples/sft/etth1_qwen3_1_7b_sft_4gpu.yaml"),
            Path("examples/rlvr/etth1_qwen3_1_7b_grpo_4gpu.yaml"),
        ]

        for recipe_path in recipes:
            with self.subTest(recipe=str(recipe_path)):
                self.assertTrue(recipe_path.exists())
                recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
                self.assertEqual(recipe["model"]["backbone"]["model_name"], expected_model)
                self.assertEqual(recipe["trace"]["run_root"], "./runs/etth1_qwen3_1_7b_4gpu")
                self.assertNotIn("server", recipe_path.name)
                self.assertNotIn("server", recipe["experiment"]["name"])

    def test_server_scripts_use_static_recipes_and_fixed_dotlist_presets(self):
        scripts = {
            Path("scripts/server_cpt_4gpu.sh"): "examples/cpt/etth1_qwen3_1_7b_cpt_4gpu.yaml",
            Path("scripts/server_sft_4gpu.sh"): "examples/sft/etth1_qwen3_1_7b_sft_4gpu.yaml",
            Path("scripts/server_rlvr_4gpu.sh"): "examples/rlvr/etth1_qwen3_1_7b_grpo_4gpu.yaml",
        }

        for script_path, recipe in scripts.items():
            with self.subTest(script=str(script_path)):
                self.assertTrue(script_path.exists())
                content = script_path.read_text(encoding="utf-8")
                self.assertIn(recipe, content)
                self.assertIn("model.backbone.model_name=/home/zyt/LLM/Qwen3-1.7B", content)
                self.assertIn('"$@"', content)
                if script_path.name == "server_rlvr_4gpu.sh":
                    self.assertIn("training.backend.verl.$arg", content)
                    self.assertIn('"${EXTRA_ARGS[@]}"', content)
                self.assertIsNone(re.search(r"\$\{[A-Z0-9_]+:-", content))
                self.assertNotIn("cat > ", content)
                self.assertNotIn(".server_tmp", content)

    def test_4gpu_rlvr_example_sets_common_verl_grpo_parameters(self):
        recipe_path = Path("examples/rlvr/etth1_qwen3_1_7b_grpo_4gpu.yaml")
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        verl = recipe["training"]["backend"]["verl"]

        self.assertEqual(verl["actor_rollout_ref"]["actor"]["optim"]["lr"], 1.0e-6)
        self.assertEqual(verl["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"], 16)
        self.assertEqual(verl["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"], 1)
        self.assertEqual(verl["actor_rollout_ref"]["actor"]["kl_loss_coef"], 0.001)
        self.assertEqual(verl["actor_rollout_ref"]["rollout"]["name"], "vllm")
        self.assertEqual(verl["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"], 0.6)
        self.assertEqual(verl["trainer"]["n_gpus_per_node"], 4)
        self.assertEqual(verl["trainer"]["nnodes"], 1)
        self.assertEqual(verl["trainer"]["logger"], ["console"])

    def test_basic_pipeline_guide_documents_commands(self):
        guide = Path("docs/CPT_SFT_RLVR_Basic_Pipeline.md")

        self.assertTrue(guide.exists())
        text = guide.read_text(encoding="utf-8")
        self.assertIn("examples/cpt/etth1_qwen_cpt.yaml", text)
        self.assertIn("examples/sft/etth1_qwen_sft.yaml", text)
        self.assertIn("examples/rlvr/etth1_qwen_grpo.yaml", text)
        self.assertIn("launch_command.txt", text)

if __name__ == "__main__":
    unittest.main()
