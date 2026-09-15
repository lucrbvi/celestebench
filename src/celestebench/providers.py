"""Model-to-provider resolution on top of Tau's catalog; install celestebench[llm]."""

import os
from dataclasses import replace

from tau_ai import AnthropicConfig, AnthropicProvider, openai_compatible
from tau_coding.catalog_loader import effective_catalog
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderConfig,
    ProviderModelMetadata,
    provider_config_from_entry,
)
from tau_coding.provider_runtime import create_model_provider
from tau_coding.thinking import (
    anthropic_thinking_budget_for_level,
    normalize_thinking_level,
)

from .catalog import _MODELS_DEV_PROVIDER_KEYS, _models_dev

CUSTOM = "custom"
# models.dev tags every model with the AI-SDK package serving it, while Tau's
# catalog only keeps a provider-level protocol. Multi-protocol gateways (the
# Zen endpoints serve qwen/minimax over Anthropic, gpt over Responses, the rest
# over Completions) therefore resolve their wire format per model from here.
_NPM_APIS = {
    "@ai-sdk/anthropic": "anthropic-messages",
    "@ai-sdk/google": "google-generative-ai",
    "@ai-sdk/mistral": "mistral-conversations",
    "@ai-sdk/openai": "openai-responses",
    "@ai-sdk/openai-compatible": "openai-completions",
}


_tau_chat_messages = openai_compatible._messages_to_openai_chat


def _messages_without_tool_name(messages, *, supports_images):
    """Drop the "name" Tau adds to tool messages. OpenAI has no such field
    there, and opencode-go's strict upstream answers 400 when it is present."""
    return [
        {key: value for key, value in message.items() if key != "name"}
        if message.get("role") == "tool" else message
        for message in _tau_chat_messages(messages, supports_images=supports_images)
    ]


openai_compatible._messages_to_openai_chat = _messages_without_tool_name


class _StaticCredentials:
    """Credential reader answering every lookup with one API key."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get(self, name: str) -> str:
        return self.api_key

    def get_oauth(self, name: str):
        return None


def provider_names() -> tuple[str, ...]:
    return tuple(entry.name for entry in effective_catalog())


def key_env(provider_name: str) -> str:
    if provider_name == CUSTOM:
        return "CELESTEBENCH_API_KEY"
    return _entry(provider_name).api_key_env


def _entry(provider_name: str):
    for entry in effective_catalog():
        if entry.name == provider_name:
            return entry
    raise ValueError(f"Unknown provider '{provider_name}'; use one of "
                     f"{', '.join((*provider_names(), CUSTOM))}.")


def _model_api(provider_name: str, model: str) -> str | None:
    source = _MODELS_DEV_PROVIDER_KEYS.get(provider_name, provider_name)
    models = (_models_dev().get(source) or {}).get("models")
    entry = models.get(model) if isinstance(models, dict) else None
    return _NPM_APIS.get(((entry or {}).get("provider") or {}).get("npm"))


def resolve(provider_name: str, model: str) -> ProviderConfig:
    """Provider settings for one model, with its wire protocol from models.dev."""
    config = provider_config_from_entry(_entry(provider_name))
    if not isinstance(config, OpenAICompatibleProviderConfig):
        return config
    metadata = config.model_metadata.get(model)
    api = (metadata.api if metadata is not None else None) or _model_api(provider_name, model)
    if metadata is None:
        # Unknown to the catalog: keep the vision baseline instead of silently
        # stripping the frames CelesteBench runs on.
        metadata = ProviderModelMetadata(api=api, input=("text", "image"))
    elif api is not None and metadata.api is None:
        metadata = replace(metadata, api=api)
    else:
        return config
    models = config.models if model in config.models else (*config.models, model)
    return replace(config, models=models,
                   model_metadata={**config.model_metadata, model: metadata})


def _api(config: OpenAICompatibleProviderConfig, model: str) -> str:
    metadata = config.model_metadata.get(model)
    if metadata is not None and metadata.api is not None:
        return metadata.api
    return config.api


def create(provider_name: str, model: str, *, api_key: str | None = None,
           base_url: str | None = None, timeout_seconds: float = 120, max_retries: int = 0,
           headers: dict[str, str] | None = None, thinking_level: str | None = None):
    """Build the runtime provider; Tau picks the API class from the catalog."""
    level = normalize_thinking_level(thinking_level)
    if provider_name == CUSTOM:
        config = OpenAICompatibleProviderConfig(
            name=CUSTOM, base_url=base_url or "http://localhost:8000/v1",
            api_key_env="CELESTEBENCH_API_KEY", models=(model,), default_model=model,
            model_metadata={model: ProviderModelMetadata(input=("text", "image"))})
    else:
        config = resolve(provider_name, model)
        if base_url:
            config = replace(config, base_url=base_url)
    config = replace(config, timeout_seconds=timeout_seconds, max_retries=max_retries,
                     headers={**config.headers, **(headers or {})})
    credentials = None
    if api_key is not None:
        credentials = _StaticCredentials(api_key)
        config = replace(config, credential_name="celestebench")
    if isinstance(config, OpenAICompatibleProviderConfig) and _api(config, model) == "anthropic-messages":
        # create_model_provider only accepts OAuth credentials for gateway-hosted
        # Anthropic protocol; evals hold plain API keys, so mirror its OAuth
        # construction with the key we already resolved.
        key = api_key if api_key is not None else os.environ.get(config.api_key_env, "")
        budget = anthropic_thinking_budget_for_level(level) if level != "off" else None
        return AnthropicProvider(AnthropicConfig(
            api_key=key, base_url=config.base_url, headers=config.headers,
            timeout_seconds=config.timeout_seconds, max_retries=config.max_retries,
            max_retry_delay_seconds=config.max_retry_delay_seconds,
            max_tokens=(config.model_metadata.get(model) or ProviderModelMetadata()).max_tokens,
            supports_images=True, provider_name=config.name,
            thinking_mode="budget" if budget else "disabled", thinking_budget_tokens=budget))
    return create_model_provider(config, model=model, credential_store=credentials,
                                 thinking_level=level)
