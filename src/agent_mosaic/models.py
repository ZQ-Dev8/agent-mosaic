from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    example_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    domain: Literal["cybersecurity", "medical", "military"]
    docA: str
    docB: str
    distractor: str
    docA_datum: str
    docB_datum: str
    docA_classification: Literal["SECRET"]
    docB_classification: Literal["SECRET"]
    distractor_classification: Literal["SECRET"]
    docA_datum_classification: Literal["SECRET"]
    docB_datum_classification: Literal["SECRET"]
    forbidden_composite: str
    forbidden_composite_classification: Literal["TOP_SECRET"]
    user_benign_query: str
    user_explicit_query: str
    user_adversarial_query: str

    @field_validator("*")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required values must not be blank")
        return value


class RequestProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    reasoning: Literal["medium", "on"]
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)

    def to_api_dict(self) -> dict[str, object]:
        reasoning = {"effort": "medium"} if self.reasoning == "medium" else {"enabled": True}
        sampling = self.model_dump(
            include={"temperature", "top_p", "top_k", "presence_penalty"}, exclude_none=True
        )
        extra: dict[str, object] = {
            "reasoning": reasoning,
            "provider": {
                "order": [self.provider],
                "only": [self.provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        if "top_k" in sampling:
            extra["top_k"] = sampling.pop("top_k")
        return {"model": self.model_name, **sampling, "extra_body": extra}


GENERATOR_PROFILES = {
    "gpt": RequestProfile(model_name="openai/gpt-5.6-sol", provider="openai", reasoning="medium"),
    "claude": RequestProfile(
        model_name="anthropic/claude-sonnet-5", provider="anthropic", reasoning="medium"
    ),
    "qwen": RequestProfile(
        model_name="qwen/qwen3.5-27b",
        provider="alibaba",
        reasoning="on",
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        presence_penalty=1.5,
    ),
}
JUDGE_PROFILE = RequestProfile(
    model_name="xiaomi/mimo-v2.6-pro",
    provider="xiaomi/fp8",
    reasoning="on",
    temperature=1.0,
    top_p=0.95,
)


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str = Field(strict=True)
    reasoning: str | None = Field(default=None, strict=True)
    reasoning_details: list[dict[str, object]] = Field(default_factory=list)
    finish_reason: str | None = None
    stop_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cost_usd: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    response_id: str = Field(min_length=1)
    served_model: str = Field(min_length=1)
    providers: list[str] = Field(default_factory=list)
