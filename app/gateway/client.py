import json

from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings

GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant"}}
    ]
}

def _load_portkey_config(value):
    if not value:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    return text


PORTKEY_CONFIG = _load_portkey_config(settings.PORTKEY_CONFIG)

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG
)


def get_langchain_llm(feature: str = "kubernetes-ai") -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=PORTKEY_CONFIG,
            metadata={
                "feature": feature,
                "_user": "kubernetes-ai-system",
                "environment": "development"
            }
        )
    )

def extract_cache_status(response) -> str:
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
