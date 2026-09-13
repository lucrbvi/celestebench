import unittest
from unittest.mock import patch

from tau_ai import AnthropicProvider, OpenAICompatibleProvider

from celestebench import providers

# models.dev data as the gateway serves it: one wire protocol per model.
GO_MODELS = {"opencode-go": {"models": {
    "qwen3.8-flash": {"provider": {"npm": "@ai-sdk/anthropic"}},
    "qwen3.8-max": {"provider": {"npm": "@ai-sdk/anthropic"}},
    "glm-5.3": {"provider": {"npm": "@ai-sdk/openai-compatible"}},
    "gpt-5.6-luna": {"provider": {"npm": "@ai-sdk/openai"}},
}}}


class ProviderResolutionTests(unittest.TestCase):
    def test_gateway_models_take_their_wire_protocol_from_models_dev(self):
        with patch.object(providers, "_models_dev", return_value=GO_MODELS):
            config = providers.resolve("opencode-go", "qwen3.8-flash")
        self.assertEqual(config.model_metadata["qwen3.8-flash"].api, "anthropic-messages")

    def test_catalog_metadata_survives_the_protocol_injection(self):
        with patch.object(providers, "_models_dev", return_value=GO_MODELS):
            config = providers.resolve("opencode-go", "qwen3.8-max")
        self.assertEqual(config.model_metadata["qwen3.8-max"].api, "anthropic-messages")
        self.assertGreater(config.model_metadata["qwen3.8-max"].context_window, 0)

    def test_unknown_models_keep_the_vision_baseline(self):
        with patch.object(providers, "_models_dev", return_value=GO_MODELS):
            config = providers.resolve("opencode-go", "brand-new-model")
        metadata = config.model_metadata["brand-new-model"]
        self.assertEqual(metadata.input, ("text", "image"))
        self.assertIsNone(metadata.api)

    def test_create_picks_the_provider_class_from_the_protocol(self):
        with patch.object(providers, "_models_dev", return_value=GO_MODELS), \
                patch.dict("os.environ", {"CELESTEBENCH_API_KEY": "s"}):
            anthropic = providers.create("opencode-go", "qwen3.8-flash", thinking_level="low")
            completions = providers.create("opencode-go", "glm-5.3", api_key="s",
                                           thinking_level="low")
        self.assertIsInstance(anthropic, AnthropicProvider)
        self.assertEqual(anthropic._config.base_url, "https://opencode.ai/zen/go/v1")
        self.assertEqual(anthropic._config.thinking_budget_tokens, 2048)
        self.assertIsInstance(completions, OpenAICompatibleProvider)

    def test_official_anthropic_goes_through_taus_runtime(self):
        with patch.object(providers, "_models_dev", return_value=GO_MODELS):
            provider = providers.create("anthropic", "claude-sonnet-4-6", api_key="s",
                                        thinking_level="low")
        self.assertIsInstance(provider, AnthropicProvider)
        self.assertEqual(provider._config.base_url, "https://api.anthropic.com/v1")

    def test_unknown_provider_and_thinking_levels_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "Unknown provider"):
            providers.key_env("grpc")
        with self.assertRaisesRegex(ValueError, "Unknown thinking mode"):
            providers.create("opencode-go", "glm-5.3", api_key="s", thinking_level="much")


if __name__ == "__main__":
    unittest.main()
