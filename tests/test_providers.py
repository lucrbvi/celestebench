from unittest.mock import patch

import pytest

try:
    from tau_agent import TextContent, ToolResultMessage
    from tau_ai import AnthropicProvider, OpenAICompatibleProvider, openai_compatible

    from celestebench import providers
except ImportError:
    pytest.skip("celestebench[llm] is not installed", allow_module_level=True)

# models.dev data as the gateway serves it: one wire protocol per model.
GO_MODELS = {"opencode-go": {"models": {
    "qwen3.8-flash": {"provider": {"npm": "@ai-sdk/anthropic"}},
    "qwen3.8-max": {"provider": {"npm": "@ai-sdk/anthropic"}},
    "glm-5.3": {"provider": {"npm": "@ai-sdk/openai-compatible"}},
    "gpt-5.6-luna": {"provider": {"npm": "@ai-sdk/openai"}},
}}}


def test_gateway_models_take_their_wire_protocol_from_models_dev():
    with patch.object(providers, "_models_dev", return_value=GO_MODELS):
        config = providers.resolve("opencode-go", "qwen3.8-flash")
    assert config.model_metadata["qwen3.8-flash"].api == "anthropic-messages"


def test_catalog_metadata_survives_the_protocol_injection():
    with patch.object(providers, "_models_dev", return_value=GO_MODELS):
        config = providers.resolve("opencode-go", "qwen3.8-max")
    assert config.model_metadata["qwen3.8-max"].api == "anthropic-messages"
    assert config.model_metadata["qwen3.8-max"].context_window > 0


def test_unknown_models_keep_the_vision_baseline():
    with patch.object(providers, "_models_dev", return_value=GO_MODELS):
        config = providers.resolve("opencode-go", "brand-new-model")
    metadata = config.model_metadata["brand-new-model"]
    assert metadata.input == ("text", "image")
    assert metadata.api is None


def test_create_picks_the_provider_class_from_the_protocol():
    with patch.object(providers, "_models_dev", return_value=GO_MODELS), \
            patch.dict("os.environ", {"CELESTEBENCH_API_KEY": "s"}):
        anthropic = providers.create("opencode-go", "qwen3.8-flash", thinking_level="low")
        completions = providers.create("opencode-go", "glm-5.3", api_key="s",
                                       thinking_level="low")
    assert isinstance(anthropic, AnthropicProvider)
    assert anthropic._config.base_url == "https://opencode.ai/zen/go/v1"
    assert anthropic._config.thinking_budget_tokens == 2048
    assert isinstance(completions, OpenAICompatibleProvider)


def test_official_anthropic_goes_through_taus_runtime():
    with patch.object(providers, "_models_dev", return_value=GO_MODELS):
        provider = providers.create("anthropic", "claude-sonnet-4-6", api_key="s",
                                    thinking_level="low")
    assert isinstance(provider, AnthropicProvider)
    assert provider._config.base_url == "https://api.anthropic.com/v1"


def test_unknown_provider_and_thinking_levels_fail_loudly():
    with pytest.raises(ValueError, match="Unknown provider"):
        providers.key_env("grpc")
    with pytest.raises(ValueError, match="Unknown thinking mode"):
        providers.create("opencode-go", "glm-5.3", api_key="s", thinking_level="much")


def test_tool_messages_drop_the_nonstandard_name_field():
    message = ToolResultMessage(tool_call_id="call_1", tool_name="play",
                                content=[TextContent(text="ok")])
    converted = openai_compatible._messages_to_openai_chat([message], supports_images=False)
    assert "name" not in converted[0]
    assert converted[0]["role"] == "tool"
