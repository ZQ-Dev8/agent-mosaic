from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from itertools import product
from pathlib import Path

from agent_mosaic.client import OpenRouterClient, error_fields, retry_delay, transient_status
from agent_mosaic.dataset import file_hash, load_dataset
from agent_mosaic.models import GENERATOR_PROFILES
from agent_mosaic.prompting import prompt_hashes, render_documents, render_messages, render_query
from agent_mosaic.storage import IdentityChanged, RunStore, SAFEGUARD_SENTINEL, json_hash, now

ORDER_SEED = 0
MAX_HTTP_ATTEMPTS = 3
MAX_EMPTY_ATTEMPTS = 4


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))}


def build_trial_plan(examples: list[dict[str, str]], repetitions: int) -> list[dict]:
    plan = []
    for example, intent, distractor, headers, rules in product(
        examples, ("benign", "explicit", "adversarial"), (False, True), (False, True), (False, True)
    ):
        for repetition in range(repetitions):
            plan.append(
                {
                    "example_id": example["example_id"],
                    "configuration": {
                        "intent": intent,
                        "distractor": distractor,
                        "headers": headers,
                        "rules": rules,
                        "repetition": repetition,
                    },
                }
            )
    random.Random(ORDER_SEED).shuffle(plan)
    return [{**trial, "order_position": i} for i, trial in enumerate(plan)]


def _response_record(manifest: dict, trial: dict, example: dict[str, str]) -> dict:
    stream = trial["stream"]
    profile = manifest["profiles"][stream]
    return {
        "run_id": manifest["run_id"],
        "trial_id": trial["trial_id"],
        "example_id": trial["example_id"],
        "domain": example["domain"],
        "configuration": trial["configuration"],
        "order_position": trial["order_position"],
        "collection_type": manifest["kind"],
        "generator": stream if manifest["kind"] == "simulate" else None,
        "harness": stream if manifest["kind"] == "live" else None,
        "requested_model": profile["model_name"],
        "profile": profile,
        "profile_sha256": json_hash(profile),
        "dataset_sha256": manifest["dataset_sha256"],
        "dataset_source": manifest["dataset_source"],
        "complete_benchmark": manifest["complete_benchmark"],
        "prompts_sha256": json_hash(manifest["prompts"]),
        "source_sha256": json_hash(manifest["sources"]),
        "prompt_sha256": trial["prompt_sha256"],
        "image_id": manifest["image"]["image_id"] if manifest["image"] else None,
        "status": "unscored",
        "final_answer": None,
        "empty_safeguard_refusal": False,
        "metadata": {},
        "attempt_count": 0,
        "unscored_reason": None,
    }


def _simulate_trial(
    store: RunStore, trial: dict, record: dict, request: dict, client: OpenRouterClient
) -> None:
    history = store.history(trial["trial_id"])
    if history:
        time.sleep(retry_delay(None, len(history)))
    for number in range(len(history) + 1, MAX_EMPTY_ATTEMPTS + 1):
        attempt = store.begin_attempt(trial["trial_id"], request)
        started = time.monotonic()
        raw = None
        result = None
        error = None
        retryable = False
        changed = None
        reason = None
        try:
            raw, result = client.stream_result(request)
        except Exception as exc:
            error = error_fields(exc)
            reason = "request_or_response_failure"
            retryable = transient_status(exc) and number < MAX_HTTP_ATTEMPTS
            delay = retry_delay(exc, number)
        else:
            record["metadata"] = result.model_dump(
                mode="json", exclude={"final_answer", "reasoning", "reasoning_details"}
            )
            try:
                store.observe_identity(trial["stream"] + ":model", result.served_model)
                if result.providers:
                    store.observe_identity(
                        trial["stream"] + ":provider", json.dumps(result.providers)
                    )
            except IdentityChanged as exc:
                changed = exc
                error = error_fields(exc)
                reason = "identity_changed"
            else:
                if {result.finish_reason, result.stop_reason} & {"length", "max_tokens"}:
                    reason = "truncated_response"
                elif (
                    result.finish_reason not in (None, "stop")
                    and not (
                        result.finish_reason == "content_filter" and result.stop_reason == "refusal"
                    )
                ) or result.stop_reason not in (
                    "stop",
                    "end_turn",
                    "stop_sequence",
                    "completed",
                    "refusal",
                ):
                    reason = "abnormal_response_termination"
                elif not result.final_answer.strip():
                    if result.stop_reason == "refusal":
                        record.update(
                            status="completed",
                            final_answer=SAFEGUARD_SENTINEL,
                            empty_safeguard_refusal=True,
                        )
                    else:
                        reason = "empty_answer"
                        retryable = result.cost_usd is not None and number < MAX_EMPTY_ATTEMPTS
                        if not retryable and number == MAX_EMPTY_ATTEMPTS:
                            reason = "empty_retries_exhausted"
                else:
                    record.update(status="completed", final_answer=result.final_answer)
        record.update(attempt_count=attempt, unscored_reason=reason)
        if result is not None and record["status"] == "unscored":
            record["final_answer"] = result.final_answer
        store.finish_attempt(
            trial["trial_id"],
            attempt,
            {
                "raw_response": raw,
                "result": result.model_dump(mode="json") if result else None,
                "error": error,
                "retryable": retryable,
                "duration_seconds": time.monotonic() - started,
            },
            None if retryable else {**record, "completed_at": now()},
        )
        if changed is not None:
            raise changed
        if not retryable:
            return
        time.sleep(delay if error is not None else retry_delay(None, number))
    raise RuntimeError("attempt budget exhausted without a terminal result")


def _live_trial(store: RunStore, trial: dict, record: dict, example: dict[str, str]) -> None:
    from agent_mosaic.live import collect_trial

    directory = store.root / "artifacts" / trial["trial_id"]
    if directory.exists() or directory.is_symlink():
        raise ValueError("live invocation directory is occupied; retained for review")
    request = {
        "example_id": example["example_id"],
        "configuration": trial["configuration"],
        "harness": trial["stream"],
        "image_id": record["image_id"],
        "prompt_sha256": trial["prompt_sha256"],
    }
    attempt = store.begin_attempt(trial["trial_id"], request)
    started = time.monotonic()
    outcome = None
    error = None
    try:
        outcome = collect_trial(
            example, trial["configuration"], trial["stream"], record["image_id"], directory
        )
        if (
            not isinstance(outcome, dict)
            or outcome.get("status") not in ("completed", "unscored")
            or type(outcome.get("empty_safeguard_refusal")) is not bool
            or not isinstance(outcome.get("metadata"), dict)
        ):
            raise ValueError("invalid live result")
        final = outcome.get("final_answer")
        if final is not None and not isinstance(final, str):
            raise ValueError("invalid live final answer")
        sentinel = outcome["empty_safeguard_refusal"]
        if sentinel:
            if final is not None and final != SAFEGUARD_SENTINEL and final.strip():
                raise ValueError("invalid live safeguard-refusal provenance")
            final = SAFEGUARD_SENTINEL
        status = outcome["status"]
        if status == "completed" and (not final or not final.strip()):
            status = "unscored"
        record.update(
            final_answer=final,
            status=status,
            empty_safeguard_refusal=sentinel,
            metadata=outcome["metadata"],
            unscored_reason=None if status == "completed" else "native_result_unscored",
        )
    except Exception as exc:
        error = error_fields(exc)
        record["unscored_reason"] = "native_dispatch_failure"
        store.pause("native_dispatch_failure")
        store.finish_attempt(
            trial["trial_id"],
            attempt,
            {"error": error, "retryable": False, "duration_seconds": time.monotonic() - started},
            {**record, "attempt_count": attempt, "completed_at": now()},
        )
        raise RuntimeError(
            "live dispatch failed; run paused; inspect retained artifacts and cleanup"
        ) from None
    store.finish_attempt(
        trial["trial_id"],
        attempt,
        {
            "result": outcome,
            "error": error,
            "retryable": False,
            "duration_seconds": time.monotonic() - started,
        },
        {**record, "attempt_count": attempt, "completed_at": now()},
    )


def collect(
    dataset: str | Path,
    out: Path,
    *,
    generator: str | None = None,
    harness: str | None = None,
    image: str | None = None,
    resume: bool = False,
    revision: str | None = None,
) -> dict:
    examples, dataset_source = load_dataset(dataset, revision)
    dataset_sha256 = dataset_source["sha256"]
    by_id = {example["example_id"]: example for example in examples}
    kind = "live" if harness else "simulate"
    image_metadata = None
    if harness:
        from agent_mosaic.live import prepare_image

        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is required")
        image_metadata = prepare_image(image, harness)
        if not isinstance(image_metadata.get("image_id"), str) or not image_metadata["image_id"]:
            raise ValueError("live image check did not return an immutable image ID")
    profiles = (
        {harness: GENERATOR_PROFILES["gpt"].model_dump()}
        if harness
        else {
            name: GENERATOR_PROFILES[name].model_dump()
            for name in ([generator] if generator else GENERATOR_PROFILES)
        }
    )
    settings = {
        "format": 1,
        "kind": kind,
        "dataset_sha256": dataset_sha256,
        "dataset_source": dataset_source,
        "complete_benchmark": len(examples) == 30,
        "prompts": prompt_hashes(),
        "sources": source_hashes(),
        "profiles": profiles,
        "image": image_metadata,
        "order_seed": ORDER_SEED,
        "retry_policy": {"http_attempts": MAX_HTTP_ATTEMPTS, "empty_attempts": MAX_EMPTY_ATTEMPTS},
    }
    base_plan = build_trial_plan(examples, 1 if harness else 5)
    plan = []
    for stream in profiles:
        for trial in base_plan:
            example = by_id[trial["example_id"]]
            config = trial["configuration"]
            prompt = (
                {
                    "query": render_query(example, config["intent"]),
                    "documents": render_documents(example, config["headers"], config["distractor"]),
                    "rules": config["rules"],
                }
                if harness
                else render_messages(example, config)
            )
            item = {**trial, "stream": stream, "prompt_sha256": json_hash(prompt)}
            plan.append({**item, "trial_id": json_hash({"settings": settings, "trial": item})})
    manifest = {**settings, "plan_sha256": json_hash(plan)}
    manifest["run_id"] = json_hash(manifest)
    with nullcontext(None) if harness else OpenRouterClient() as client:
        with RunStore(out, manifest, plan, resume) as store:
            for trial in plan:
                if store.result(trial["trial_id"]) is not None:
                    continue
                if prompt_hashes() != manifest["prompts"] or source_hashes() != manifest["sources"]:
                    raise ValueError("prompts or source changed during collection")
                example = by_id[trial["example_id"]]
                record = _response_record(manifest, trial, example)
                if harness:
                    _live_trial(store, trial, record, example)
                else:
                    request = {
                        **GENERATOR_PROFILES[trial["stream"]].to_api_dict(),
                        "messages": render_messages(example, trial["configuration"]),
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    }
                    _simulate_trial(store, trial, record, request, client)
            if dataset_source["kind"] == "jsonl" and file_hash(Path(dataset)) != dataset_sha256:
                raise ValueError(
                    "dataset changed during collection; results retained without export"
                )
            export = store.export("responses.jsonl")
            counts = {
                status: sum(store.result(trial["trial_id"])["status"] == status for trial in plan)
                for status in ("completed", "unscored")
            }
            return {
                "run_id": manifest["run_id"],
                "responses": str(export),
                "planned": len(plan),
                **counts,
            }
