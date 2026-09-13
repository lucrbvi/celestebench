import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from celestebench import catalog, cost

PRICES = {"openai": {"models": {"gpt-5.6-sol": {
    "id": "gpt-5.6-sol",
    "cost": {"input": 4, "output": 20, "cache_read": 0.4, "cache_write": 5}}}}}


class ProducerTests(unittest.TestCase):
    def test_producer_matches_the_labs_we_run(self):
        self.assertEqual(catalog.producer("gpt-5.6-sol"), "openai")
        self.assertEqual(catalog.producer("glm-5.3-flash"), "zai")
        self.assertEqual(catalog.producer("kimi-k3"), "moonshotai")
        self.assertEqual(catalog.producer("mimo-v2.5"), "xiaomi")
        self.assertEqual(catalog.producer("muse-spark-1.3-contributor"), "meta")
        self.assertEqual(catalog.producer("qwen3.8-flash-next"), "alibaba")
        self.assertEqual(catalog.producer("deepseek-v4-flash-vision-exp"), "deepseek")
        self.assertIsNone(catalog.producer("dots-studio/dots-3-note-preview"))
        self.assertEqual(catalog.producer_name("zai"), "Z.AI")

    def test_price_reads_the_first_party_provider(self):
        with patch.object(catalog, "_models_dev", return_value=PRICES):
            self.assertEqual(catalog.price("openai", "gpt-5.6-sol")["output"], 20)


class CostTests(unittest.TestCase):
    def tau(self, folder, usage):
        (folder / "messages.jsonl").write_text(
            json.dumps({"role": "assistant", "usage": usage}) + "\n" + "{truncated\n")

    def test_tau_cost_uses_each_token_bucket(self):
        with TemporaryDirectory() as tmp, patch.object(catalog, "_models_dev", return_value=PRICES):
            folder = Path(tmp)
            self.tau(folder, {"input": 1000, "output": 2000, "cacheRead": 4000, "cacheWrite": 5000})
            # (1000*4 + 2000*20 + 4000*.4 + 5000*5) / 1e6
            self.assertAlmostEqual(cost.rollout_cost(folder / "messages.jsonl", "gpt-5.6-sol"),
                                   0.0706, places=6)

    def test_codex_cost_subtracts_the_cached_part_from_input(self):
        with TemporaryDirectory() as tmp, patch.object(catalog, "_models_dev", return_value=PRICES):
            folder = Path(tmp)
            (folder / "codex.jsonl").write_text(json.dumps({
                "type": "turn.completed",
                "usage": {"input_tokens": 5000, "cached_input_tokens": 4000,
                          "output_tokens": 1000, "cache_write_input_tokens": 0}}) + "\n")
            # ((5000-4000)*4 + 1000*20 + 4000*.4) / 1e6
            self.assertAlmostEqual(cost.rollout_cost(folder / "codex.jsonl", "gpt-5.6-sol",
                                                     codex=True), 0.0256, places=6)

    def test_context_tiers_raise_the_rate(self):
        tiers = {"openai": {"models": {"gpt-5.6-sol": {"id": "gpt-5.6-sol", "cost": {
            "input": 4, "output": 20,
            "tiers": [{"input": 8, "output": 30, "tier": {"type": "context", "size": 100}}]}}}}}
        with TemporaryDirectory() as tmp, patch.object(catalog, "_models_dev", return_value=tiers):
            folder = Path(tmp)
            self.tau(folder, {"input": 50, "output": 10})
            small = cost.rollout_cost(folder / "messages.jsonl", "gpt-5.6-sol")
            self.tau(folder, {"input": 200, "output": 10})
            large = cost.rollout_cost(folder / "messages.jsonl", "gpt-5.6-sol")
        self.assertAlmostEqual(small, (50 * 4 + 10 * 20) / 1_000_000, places=6)
        self.assertAlmostEqual(large, (200 * 8 + 10 * 30) / 1_000_000, places=6)

    def test_unpriceable_or_usage_free_runs_stay_unknown(self):
        with TemporaryDirectory() as tmp, patch.object(catalog, "_models_dev", return_value=PRICES):
            folder = Path(tmp)
            self.assertIsNone(cost.rollout_cost(folder / "messages.jsonl", "gpt-5.6-sol"))
            self.tau(folder, {"input": 1})
            self.assertIsNone(cost.rollout_cost(folder / "messages.jsonl", "dots-studio/x"))


if __name__ == "__main__":
    unittest.main()
