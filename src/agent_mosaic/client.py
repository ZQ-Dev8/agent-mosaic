from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator
from decimal import Decimal
from email.utils import parsedate_to_datetime

import httpx
from openai import APIStatusError, OpenAI
from openai._streaming import SSEDecoder

from agent_mosaic.models import GenerationResult

STREAM_IDLE_TIMEOUT = 120.0


def error_fields(exc: Exception) -> dict:
    return {
        "type": type(exc).__name__,
        "http_status": exc.status_code if isinstance(exc, APIStatusError) else None,
    }


def transient_status(exc: Exception) -> bool:
    return isinstance(exc, APIStatusError) and (
        exc.status_code == 429 or 500 <= exc.status_code < 600
    )


def retry_delay(exc: Exception | None, number: int) -> float:
    fallback = float(2 ** (number - 1))
    value = exc.response.headers.get("Retry-After") if isinstance(exc, APIStatusError) else None
    if value is None:
        return fallback
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return fallback
    return max(0.0, delay) if math.isfinite(delay) else fallback


def parse_stream(chunks: Iterator[bytes]) -> GenerationResult:
    answer: list[str] = []
    reasoning: list[str] = []
    details: list[dict[str, object]] = []
    providers: set[str] = set()
    metadata: dict[str, object] = {}
    done = False
    for event in SSEDecoder().iter_bytes(chunks):
        if not event.data and event.event != "error":
            continue
        if done:
            raise ValueError("data after terminal SSE DONE")
        if event.data == "[DONE]":
            done = True
            continue
        chunk = event.json()
        if not isinstance(chunk, dict) or "error" in chunk or event.event == "error":
            raise ValueError("stream error or malformed event")
        for source, target in (("id", "response_id"), ("model", "served_model")):
            if chunk.get(source) is not None:
                value = chunk[source]
                if (
                    not isinstance(value, str)
                    or not value
                    or (target in metadata and metadata[target] != value)
                ):
                    raise ValueError("inconsistent stream identity")
                metadata[target] = value
        provider = chunk.get("provider")
        if provider is not None:
            if not isinstance(provider, str) or not provider:
                raise ValueError("invalid provider identity")
            providers.add(provider)
        choices = chunk.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("expected one streamed choice")
        choice = choices[0] if choices else {}
        if not isinstance(choice, dict) or (choice and choice.get("index") != 0):
            raise ValueError("unexpected streamed choice")
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict) or delta.get("tool_calls") or delta.get("function_call"):
            raise ValueError("expected a text answer, not tool calls")
        for field, parts in (("content", answer), ("reasoning", reasoning)):
            text = delta.get(field)
            if text is not None:
                if not isinstance(text, str):
                    raise ValueError("non-text stream delta")
                parts.append(text)
        detail_delta = delta.get("reasoning_details")
        if detail_delta is not None:
            if not isinstance(detail_delta, list) or any(
                not isinstance(item, dict) for item in detail_delta
            ):
                raise ValueError("malformed reasoning details")
            details.extend(detail_delta)
        if choice.get("finish_reason") is not None:
            metadata["finish_reason"] = choice["finish_reason"]
        stop = (
            choice.get("stop_reason")
            or chunk.get("stop_reason")
            or choice.get("native_finish_reason")
        )
        if stop is not None:
            metadata["stop_reason"] = stop
        usage = chunk.get("usage")
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError("malformed usage")
            for field in ("prompt_tokens", "completion_tokens"):
                if field in usage:
                    metadata[field] = usage[field]
            if "cost" in usage:
                metadata["cost_usd"] = json.loads(event.data, parse_float=Decimal)["usage"]["cost"]
            token_details = usage.get("completion_tokens_details")
            if token_details is not None:
                if not isinstance(token_details, dict):
                    raise ValueError("malformed token details")
                if "reasoning_tokens" in token_details:
                    metadata["reasoning_tokens"] = token_details["reasoning_tokens"]
    if not done:
        raise ValueError("stream ended without DONE")
    if not metadata.get("response_id") or not metadata.get("served_model"):
        raise ValueError("stream missing response identity")
    if not (metadata.get("finish_reason") or metadata.get("stop_reason")):
        raise ValueError("stream missing terminal reason")
    metadata["stop_reason"] = metadata.get("stop_reason") or metadata.get("finish_reason")
    return GenerationResult(
        final_answer="".join(answer),
        reasoning="".join(reasoning) if reasoning else None,
        reasoning_details=details,
        providers=sorted(providers),
        **metadata,
    )


class OpenRouterClient:
    def __init__(self) -> None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            max_retries=0,
            timeout=httpx.Timeout(STREAM_IDLE_TIMEOUT, connect=5.0),
        )

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.close()

    def stream_result(self, request: dict) -> tuple[str, GenerationResult]:
        if request.get("stream") is not True or request.get("stream_options") != {
            "include_usage": True
        }:
            raise ValueError("streaming requests must ask for usage")
        body = bytearray()
        with self.client.chat.completions.with_streaming_response.create(**request) as response:

            def chunks() -> Iterator[bytes]:
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    yield chunk

            result = parse_stream(chunks())
        return body.decode("utf-8"), result
