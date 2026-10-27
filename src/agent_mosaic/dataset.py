from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset as load_hf_dataset

from agent_mosaic.models import Example
from agent_mosaic.prompting import prompt_text

DOMAINS = ("cybersecurity", "medical", "military")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("duplicate JSON keys")
    return value


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_dataset(
    source: str | Path, revision: str | None = None
) -> tuple[list[dict[str, str]], dict[str, str]]:
    path = Path(source)
    local = (
        isinstance(source, Path)
        or path.exists()
        or path.suffix == ".jsonl"
        or str(source).startswith((".", "~", "/"))
    )
    if local:
        if revision is not None:
            raise ValueError("--revision applies to Hugging Face dataset IDs, not local JSONL")
        identity = {"kind": "jsonl", "sha256": file_hash(path)}
        with path.open(encoding="utf-8") as handle:
            rows = list(handle)
    else:
        identity = {
            "kind": "huggingface",
            "repo_id": str(source),
            "revision": revision or "main",
            "split": "test",
        }
        try:
            rows = load_hf_dataset(str(source), split="test", revision=identity["revision"])
        except Exception:
            raise ValueError(
                "could not load Hugging Face dataset; check its ID, revision and test split"
            ) from None
    examples = []
    for number, row in enumerate(rows, 1):
        try:
            value = json.loads(row, object_pairs_hook=unique_object) if local else row
            example = Example.model_validate(value).model_dump()
            expected = (
                example["user_explicit_query"].rstrip()
                + "\n\n"
                + prompt_text("user_adversarial_suffix.txt").strip()
            )
            if example["user_adversarial_query"] != expected:
                raise ValueError("adversarial query differs from study concatenation")
            examples.append(example)
        except ValueError:
            raise ValueError(f"invalid dataset row {number}") from None
    if len({e["example_id"] for e in examples}) != len(examples):
        raise ValueError("duplicate example IDs")
    if not 1 <= len(examples) <= 30:
        raise ValueError("require one to 30 examples")
    if len(examples) == 30 and Counter(e["domain"] for e in examples) != dict.fromkeys(DOMAINS, 10):
        raise ValueError("the complete benchmark requires ten examples per domain")
    examples.sort(key=lambda e: e["example_id"])
    if local:
        if file_hash(path) != identity["sha256"]:
            raise ValueError("dataset changed while loading")
    else:
        content = json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        identity["sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return examples, identity
