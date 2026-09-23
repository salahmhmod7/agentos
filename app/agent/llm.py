"""LLM abstraction layer with pluggable providers.

Supports:
  - Ollama (local, free, slower)
  - Groq   (cloud, free tier, ultra fast)
  - Gemini (cloud, OpenAI-compatible endpoint)

The active provider is chosen via LLM_PROVIDER (or EVAL_LLM_PROVIDER) in .env.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ollama import Client as OllamaClient
from groq import Groq as GroqClient
from openai import OpenAI as OpenAIClient

from app.core.config import settings


# ---------------------------------------------------------------------------
# Normalized response types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A normalized tool call from any provider."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str = ""
    model: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_thought_tags(text: str) -> str:
    """Remove <thought>...</thought> blocks that some models emit."""
    return re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class LLMProvider:
    """Base class for LLM providers."""

    name: str = "base"
    model: str = ""

    def chat(self, messages: list[dict], temperature: float = 0.7) -> LLMResponse:
        raise NotImplementedError

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
    ) -> LLMResponse:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """Local Ollama provider."""

    name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.ollama_model
        self.client = OllamaClient(host=base_url or settings.ollama_base_url)

    def chat(self, messages, temperature=0.7):
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature},
        )
        msg = resp["message"]
        content = _strip_thought_tags(msg.get("content", "") or "")
        return LLMResponse(
            content=content,
            model=resp.get("model", self.model),
            tool_calls=[],
            raw_message=dict(msg),
            raw=dict(resp),
        )

    def chat_with_tools(self, messages, tools, temperature=0.7):
        resp = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options={"temperature": temperature},
        )
        msg = resp["message"]
        raw_calls = msg.get("tool_calls", []) or []

        normalized: list[ToolCall] = []
        for i, call in enumerate(raw_calls):
            fn = call.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            normalized.append(
                ToolCall(
                    id=call.get("id") or f"ollama_call_{i}",
                    name=fn.get("name", ""),
                    arguments=args or {},
                )
            )

        content = _strip_thought_tags(msg.get("content", "") or "")
        return LLMResponse(
            content=content,
            model=resp.get("model", self.model),
            tool_calls=normalized,
            raw_message=dict(msg),
            raw=dict(resp),
        )


def _parse_openai_style_response(completion: Any, model: str) -> LLMResponse:
    """Shared parser for OpenAI-compatible APIs (Groq, Gemini via OpenAI)."""
    choice = completion.choices[0]
    msg = choice.message

    raw_calls = getattr(msg, "tool_calls", None) or []
    normalized: list[ToolCall] = []
    for call in raw_calls:
        args = call.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        normalized.append(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=args or {},
            )
        )

    raw_message: dict[str, Any] = {
        "role": "assistant",
        "content": msg.content or "",
    }
    if raw_calls:
        raw_message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments,
                },
            }
            for c in raw_calls
        ]

    # Strip any <thought> tags (Gemini/Gemma emit these)
    content = _strip_thought_tags(msg.content or "")

    return LLMResponse(
        content=content,
        model=getattr(completion, "model", model),
        tool_calls=normalized,
        raw_message=raw_message,
        raw=completion.model_dump() if hasattr(completion, "model_dump") else {},
    )


class GroqProvider(LLMProvider):
    """Groq cloud provider (OpenAI-compatible API)."""

    name = "groq"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.groq_model
        key = api_key or settings.groq_api_key
        if not key:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")
        self.client = GroqClient(api_key=key)

    def chat(self, messages, temperature=0.7):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return _parse_openai_style_response(completion, self.model)

    def chat_with_tools(self, messages, tools, temperature=0.7):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        return _parse_openai_style_response(completion, self.model)


class GeminiProvider(LLMProvider):
    """Google Gemini (or Gemma) via the OpenAI-compatible endpoint."""

    name = "gemini"

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.gemini_model
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.client = OpenAIClient(api_key=key, base_url=self.BASE_URL)

    def chat(self, messages, temperature=0.7):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return _parse_openai_style_response(completion, self.model)

    def chat_with_tools(self, messages, tools, temperature=0.7):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        return _parse_openai_style_response(completion, self.model)


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

def build_provider(provider_name: str | None = None) -> LLMProvider:
    """Create the provider chosen in settings (or the given name)."""
    name = (provider_name or settings.llm_provider).lower()
    if name == "groq":
        return GroqProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(
        f"Unknown LLM provider: {name!r}. Use 'ollama', 'groq', or 'gemini'."
    )


class LLM:
    """Facade over the active provider."""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider or build_provider()

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def chat(self, messages, temperature: float = 0.7) -> LLMResponse:
        return self.provider.chat(messages, temperature=temperature)

    def chat_with_tools(self, messages, tools, temperature: float = 0.7) -> LLMResponse:
        return self.provider.chat_with_tools(messages, tools, temperature=temperature)


# Singleton
llm = LLM()