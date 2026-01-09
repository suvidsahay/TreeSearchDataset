from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from config.config_loader import load_config, require


# Cache chats so every method gets the same instance per component (no surprises)
_CHAT_CACHE: Dict[str, Any] = {}
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _get_cfg() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


def _healthcheck_vllm(base_url: str, timeout_sec: int = 3) -> None:
    """
    Hard-fail if vLLM is not reachable.
    Uses OpenAI-compatible endpoints; /models is typically available.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        r = requests.get(url, timeout=timeout_sec)
        if r.status_code >= 400:
            raise RuntimeError(f"vLLM reachable but unhealthy: GET {url} -> {r.status_code}")
    except Exception as e:
        raise RuntimeError(f"vLLM not reachable at {base_url}. Start vLLM first. Root error: {e}") from e


def validate_config() -> None:
    """
    Call once at startup (tree_construct.py) to fail fast.
    """
    cfg = _get_cfg()

    # Require three knobs
    for comp in ("generation", "verification", "evaluation"):
        alias = require(cfg, f"llm.{comp}.alias")
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(f"llm.{comp}.alias must be a non-empty string")

        # Verify alias exists
        providers = require(cfg, "providers")
        if alias not in providers:
            raise ValueError(f"Unknown alias '{alias}' referenced in llm.{comp}.alias")

        prov = require(cfg, f"providers.{alias}.provider")
        if prov == "vllm":
            base_url = require(cfg, "vllm.base_url")
            if require(cfg, "vllm.require_reachable"):
                timeout_sec = int(require(cfg, "vllm.timeout_sec"))
                _healthcheck_vllm(base_url, timeout_sec=timeout_sec)

    # Key requirements (only for selected providers)
    used_providers = set(require(cfg, f"providers.{require(cfg, f'llm.{c}.alias')}.provider") for c in ("generation","verification","evaluation"))
    if "openai" in used_providers and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI cloud models.")
    if "anthropic" in used_providers and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic models.")
    # vLLM uses OpenAI *protocol*, so we allow a dummy key (no billing/validation).


def get_chat(component: str) -> Any:
    """
    component ∈ {"generation","verification","evaluation"}.
    Returns a configured LangChain chat client.
    Hard-fails if config is missing/invalid. No fallbacks.
    """
    if component not in ("generation", "verification", "evaluation"):
        raise ValueError(f"Unknown component: {component}")

    cache_key = component
    if cache_key in _CHAT_CACHE:
        return _CHAT_CACHE[cache_key]

    cfg = _get_cfg()
    alias = require(cfg, f"llm.{component}.alias")
    temp = float(require(cfg, f"llm.{component}.temperature"))

    provider = require(cfg, f"providers.{alias}.provider")
    model_id = require(cfg, f"providers.{alias}.model")

    if provider == "openai":
        chat = ChatOpenAI(
            model=model_id,
            temperature=temp,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=None,  # critical: OpenAI cloud
        )

    elif provider == "anthropic":
        chat = ChatAnthropic(
            model=model_id,
            temperature=temp,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    elif provider == "vllm":
        base_url = require(cfg, "vllm.base_url")
        # vLLM speaks OpenAI protocol; key can be dummy.
        chat = ChatOpenAI(
            model=model_id,
            temperature=temp,
            base_url=base_url.rstrip("/"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-vllm-key"),
        )

    else:
        raise ValueError(f"Unsupported provider '{provider}' for alias '{alias}'")

    _CHAT_CACHE[cache_key] = chat
    return chat
