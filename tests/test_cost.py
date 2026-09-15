import json
from unittest.mock import patch

import pytest

from celestebench import catalog, cost

PRICES = {"openai": {"models": {"gpt-5.6-sol": {
    "id": "gpt-5.6-sol",
    "cost": {"input": 4, "output": 20, "cache_read": 0.4, "cache_write": 5}}}}}


def test_producer_matches_the_labs_we_run():
    assert catalog.producer("gpt-5.6-sol") == "openai"
    assert catalog.producer("glm-5.3-flash") == "zai"
    assert catalog.producer("kimi-k3") == "moonshotai"
    assert catalog.producer("mimo-v2.5") == "xiaomi"
    assert catalog.producer("muse-spark-1.3-contributor") == "meta"
    assert catalog.producer("qwen3.8-flash-next") == "alibaba"
    assert catalog.producer("deepseek-v4-flash-vision-exp") == "deepseek"
    assert catalog.producer("dots-studio/dots-3-note-preview") is None
    assert catalog.producer_name("zai") == "Z.AI"


def test_price_reads_the_first_party_provider():
    with patch.object(catalog, "_models_dev", return_value=PRICES):
        assert catalog.price("openai", "gpt-5.6-sol")["output"] == 20


def test_price_translates_a_gateway_name_to_the_lab_model():
    prices = {
        "mistral": {"models": {"mistral-medium-2604": {
            "id": "mistral-medium-2604", "name": "Mistral Medium 3.5",
            "cost": {"input": 1.5, "output": 7.5}}}},
        "openrouter": {"models": {"mistralai/mistral-medium-3-5": {
            "id": "mistralai/mistral-medium-3-5", "name": "Mistral Medium 3.5",
            "cost": {"input": 2, "output": 8}}}},
    }
    with patch.object(catalog, "_models_dev", return_value=prices):
        assert catalog.price("mistral", "mistralai/mistral-medium-3-5")["input"] == 1.5


def tau(folder, usage):
    (folder / "messages.jsonl").write_text(
        json.dumps({"role": "assistant", "usage": usage}) + "\n" + "{truncated\n")


def test_tau_cost_uses_each_token_bucket(tmp_path):
    with patch.object(catalog, "_models_dev", return_value=PRICES):
        tau(tmp_path, {"input": 1000, "output": 2000, "cacheRead": 4000, "cacheWrite": 5000})
        # (1000*4 + 2000*20 + 4000*.4 + 5000*5) / 1e6
        assert cost.rollout_cost(tmp_path / "messages.jsonl", "gpt-5.6-sol") == pytest.approx(0.0706, abs=1e-6)


def test_codex_cost_subtracts_the_cached_part_from_input(tmp_path):
    with patch.object(catalog, "_models_dev", return_value=PRICES):
        (tmp_path / "codex.jsonl").write_text(json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 5000, "cached_input_tokens": 4000,
                      "output_tokens": 1000, "cache_write_input_tokens": 0}}) + "\n")
        # ((5000-4000)*4 + 1000*20 + 4000*.4) / 1e6
        assert cost.rollout_cost(tmp_path / "codex.jsonl", "gpt-5.6-sol",
                                 codex=True) == pytest.approx(0.0256, abs=1e-6)


def test_context_tiers_raise_the_rate(tmp_path):
    tiers = {"openai": {"models": {"gpt-5.6-sol": {"id": "gpt-5.6-sol", "cost": {
        "input": 4, "output": 20,
        "tiers": [{"input": 8, "output": 30, "tier": {"type": "context", "size": 100}}]}}}}}
    with patch.object(catalog, "_models_dev", return_value=tiers):
        tau(tmp_path, {"input": 50, "output": 10})
        small = cost.rollout_cost(tmp_path / "messages.jsonl", "gpt-5.6-sol")
        tau(tmp_path, {"input": 200, "output": 10})
        large = cost.rollout_cost(tmp_path / "messages.jsonl", "gpt-5.6-sol")
    assert small == pytest.approx((50 * 4 + 10 * 20) / 1_000_000, abs=1e-6)
    assert large == pytest.approx((200 * 8 + 10 * 30) / 1_000_000, abs=1e-6)


def test_unpriceable_or_usage_free_runs_stay_unknown(tmp_path):
    with patch.object(catalog, "_models_dev", return_value=PRICES):
        assert cost.rollout_cost(tmp_path / "messages.jsonl", "gpt-5.6-sol") is None
        tau(tmp_path, {"input": 1})
        assert cost.rollout_cost(tmp_path / "messages.jsonl", "dots-studio/x") is None
