import json
import os
from unittest.mock import patch

from celestebench import auth


def test_opencode_reads_the_data_home_auth_file(tmp_path):
    data = tmp_path / "opencode"
    data.mkdir()
    (data / "auth.json").write_text(json.dumps({
        "openai": {"type": "oauth"},
        "opencode-go": {"type": "api"},
    }))
    with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}):
        assert auth.provider("opencode", "openai/gpt-5") == "openai"
        assert auth.oauth("opencode", "openai/gpt-5")
        assert not auth.oauth("opencode", "opencode-go/deepseek")
        # An unknown provider carries no OAuth subscription.
        assert not auth.oauth("opencode", "openrouter/llama")


def test_pi_resolves_a_bare_model_through_its_default_provider(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "auth.json").write_text(json.dumps({"openai-codex": {"type": "oauth"}}))
    (agent / "settings.json").write_text(json.dumps({"defaultProvider": "openai-codex"}))
    with patch.dict(os.environ, {"PI_CODING_AGENT_DIR": str(agent)}):
        assert auth.oauth("pi", "gpt-5.6-sol")
        assert auth.oauth("pi", "openai-codex/gpt-5.6-sol")


def test_missing_credentials_are_not_oauth(tmp_path):
    with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path), "PI_CODING_AGENT_DIR": str(tmp_path)}):
        assert not auth.oauth("opencode", "openai/gpt-5")
        assert not auth.oauth("pi", "openai-codex/gpt-5.6-sol")
        assert not auth.oauth("unknown", "x/y")
