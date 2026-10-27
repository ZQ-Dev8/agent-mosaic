from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import files


def prompt_text(name: str) -> str:
    return files("agent_mosaic").joinpath("prompts", *name.split("/")).read_text(encoding="utf-8")


def aggregation_policy() -> str:
    return prompt_text("agent_rules.txt")


def render_query(example: dict[str, str], intent: str) -> str:
    if intent not in ("benign", "explicit", "adversarial"):
        raise ValueError("intent must be benign, explicit, or adversarial")
    return example[f"user_{intent}_query"]


def render_documents(
    example: dict[str, str], headers: bool, distractor: bool
) -> list[tuple[str, str]]:
    names = ("docA", "docB", "distractor") if distractor else ("docA", "docB")
    return [
        (
            "docC.txt" if name == "distractor" else f"{name}.txt",
            (f"Classification: {example[name + '_classification']}\n\n" if headers else "")
            + example[name],
        )
        for name in names
    ]


def render_messages(example: dict[str, str], configuration: dict) -> list[dict[str, str]]:
    template = json.loads(prompt_text("templates/default.json"))
    system = prompt_text(template["system_prompt_file"])
    if configuration["rules"]:
        system = f"{system.rstrip()}\n\n{aggregation_policy()}"
    documents = render_documents(example, configuration["headers"], configuration["distractor"])
    values = {
        "task": render_query(example, configuration["intent"]),
        "document1": documents[0][1],
        "document2": documents[1][1],
        "distractor_document": documents[2][1] if configuration["distractor"] else "",
        "file_list": (
            '{"total_count": 3, "files": ["docs/docA.txt", "docs/docB.txt", "docs/docC.txt"]}'
            if configuration["distractor"]
            else '{"total_count": 2, "files": ["docs/docA.txt", "docs/docB.txt"]}'
        ),
    }
    roles = {"human": "user", "tool": "user", "assistant": "assistant"}
    messages = [{"role": "system", "content": system}]
    for message in template["messages"]:
        if message.get("role_filter") == "D" and not configuration["distractor"]:
            continue
        content = re.sub(
            r"\{(\w+)\}", lambda match: values.get(match[1], match[0]), message["content"]
        )
        messages.append({"role": roles[message["role"]], "content": content})
    return messages


def render_judge_messages(forbidden_composite: str, final_answer: str) -> list[dict[str, str]]:
    values = {"forbidden_statement": forbidden_composite, "model_output": final_answer}
    return [
        {"role": "system", "content": prompt_text("judge_system.txt").format(**values)},
        {"role": "user", "content": prompt_text("judge_user.txt").format(**values)},
    ]


def prompt_hashes() -> dict[str, str]:
    names = (
        "agentic_system.txt",
        "agent_rules.txt",
        "user_adversarial_suffix.txt",
        "judge_system.txt",
        "judge_user.txt",
        "templates/default.json",
    )
    return {name: hashlib.sha256(prompt_text(name).encode("utf-8")).hexdigest() for name in names}
