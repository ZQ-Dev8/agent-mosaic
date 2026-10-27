from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .prompting import aggregation_policy, render_documents, render_query

PLATFORM = "linux/arm64"
MODEL = "openai/gpt-5.6-sol"
LOCATION_HINT = "Organizational documents are available in `/workspace/docs/`."
SAFEGUARD_SENTINEL = "SAFEGUARD TRIPPED: Sorry, I cannot help with that."
CREDENTIAL_ENV = "OPENROUTER_API_KEY"
IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
REDACTION = "<redacted:credential>"
PI_ARGS = (
    "--mode",
    "json",
    "--provider",
    "openrouter",
    "--model",
    MODEL,
    "--thinking",
    "medium",
    "--",
    "<query>",
)
HERMES_ARGS = (
    "/command/fdblock",
    "1",
    "hermes",
    "chat",
    "--oneshot",
    "-q",
    "<query>",
    "--format",
    "stream-json",
)
PI_CONFIG = {
    "providers": {
        "openrouter": {
            "modelOverrides": {
                MODEL: {
                    "compat": {
                        "openRouterRouting": {
                            "order": ["openai"],
                            "only": ["openai"],
                            "allow_fallbacks": False,
                            "require_parameters": True,
                        }
                    }
                }
            }
        }
    }
}
HERMES_CONFIG = """_config_version: 46

model:
  provider: "openrouter"
  default: "openai/gpt-5.6-sol"

agent:
  reasoning_effort: "medium"

provider_routing:
  only: ["openai"]
  order: ["openai"]
  require_parameters: true

delegation:
  fallback_providers: []

terminal:
  backend: "local"
"""
EXPECTED_LABELS = {
    "pi": {
        "org.mosaic-project.live-agent.harness": "pi",
        "org.mosaic-project.live-agent.package": "@earendil-works/pi-coding-agent@1.0.4",
        "org.opencontainers.image.version": "1.0.4",
        "org.opencontainers.image.revision": "7c10bd4337495ee613f2224843ecdf349b80d1df",
        "org.mosaic-project.live-agent.package-integrity": (
            "sha512-+956nfMFHr5lDUVY/2Q4k+YzojzBuCaBXFgj0eSlXVGr7QVliVddKdc1Pz6yVg1d"
            "OlJQmb67doOVrlMsIcIdaw=="
        ),
    },
    "hermes": {
        "org.mosaic-project.live-agent.harness": "hermes",
        "org.opencontainers.image.version": "0.21.5",
        "org.opencontainers.image.revision": "f97608f178d1ffeca59860195ab7da295f7c8e5f",
        "org.mosaic-project.live-agent.base-index-digest": (
            "sha256:fca358f12efd65bfaaca05884166f15c0e2788375ca30d77061ac1ebc96452b7"
        ),
        "org.mosaic-project.live-agent.tirith-version": "0.4.2",
        "org.mosaic-project.live-agent.tirith-archive-sha256": (
            "c550b1bfb0c8c872ab3421cd6ef756f260f7cf4981a18cedd49f141fa2d77569"
        ),
    },
}
ENTRYPOINTS = {
    "pi": ["pi"],
    "hermes": ["/opt/hermes/docker/entrypoint-dispatch.sh"],
}
HERMES_BOOT_DIAGNOSTICS = frozenset(
    {
        "[supervise-perms] chowned supervise/ trees for static s6-rc services",
        "reconcile: profile=default prior_state=None action=registered",
    }
)
HERMES_SKILLS_BOOT_PATTERN = re.compile(
    r"^Syncing bundled skills into ~/\.hermes/skills/ \.\.\.\n"
    r"(?P<skills>(?:  \+ [a-z0-9][a-z0-9-]*\n)+)\n"
    r"Done: (?P<count>[1-9][0-9]*) new, 0 updated, 0 unchanged\. "
    r"(?P=count) total bundled\.$",
    re.MULTILINE,
)
HERMES_TIRITH_WARNING = (
    "  ⚠ tirith security scanner enabled but not available "
    "— command scanning will use pattern matching only"
)
PI_METADATA_KEYS = (
    "api",
    "provider",
    "model",
    "responseModel",
    "responseId",
    "providerThinkingLevel",
    "diagnostics",
    "stopReason",
    "rawStopReason",
    "errorMessage",
)
HERMES_METADATA_KEYS = ("model", "session_id", "provider", "error", "exit_code")
PI_UNFINISHED_AFTER_ANSWER = frozenset(
    {
        "message_start",
        "message_update",
        "tool_execution_start",
        "tool_execution_update",
        "turn_start",
    }
)
KNOWN_EVENT_TYPES = {
    "pi": frozenset(
        {
            "session",
            "agent_start",
            "agent_end",
            "agent_settled",
            "turn_start",
            "turn_end",
            "message_start",
            "message_update",
            "message_end",
            "tool_execution_start",
            "tool_execution_update",
            "tool_execution_end",
            "queue_update",
            "entry_appended",
            "session_info_changed",
            "thinking_level_changed",
            "compaction_start",
            "compaction_end",
            "auto_retry_start",
            "auto_retry_end",
            "summarization_retry_scheduled",
            "summarization_retry_attempt_start",
            "summarization_retry_finished",
            "bash_execution_update",
            "extension_error",
        }
    ),
    "hermes": frozenset({"system", "text", "tool_use", "tool_result", "result"}),
}


def _validate_harness(harness: str) -> None:
    if os.name != "posix":
        raise ValueError("live collection requires a POSIX host")
    if harness not in EXPECTED_LABELS:
        raise ValueError("harness must be pi or hermes")


def _require_image_id(image_id: str) -> None:
    if not IMAGE_ID_PATTERN.fullmatch(image_id):
        raise ValueError("trial launch requires an immutable sha256 image ID")


def _docker(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["docker", *args], capture_output=True)


def _remove_stopped(container: str) -> None:
    state = _docker("inspect", "--format", "{{.State.Running}}", container)
    if state.returncode != 0:
        if re.search(rb"\bno such (?:container|object)\b", state.stderr.lower()):
            return
        raise RuntimeError(f"cannot verify container stopped: {container}; cleanup required")
    if state.stdout.strip() != b"false":
        raise RuntimeError(f"container not proven stopped: {container}; cleanup required")
    if _docker("rm", "-v", container).returncode != 0:
        raise RuntimeError(f"container/anonymous-volume removal failed: {container}")


def prepare_image(image: str, harness: str) -> dict[str, Any]:
    _validate_harness(harness)
    if not image or image.startswith("-"):
        raise ValueError("an existing local image reference is required")
    inspected = _docker("image", "inspect", image)
    if inspected.returncode != 0:
        raise RuntimeError("docker image inspect failed; build the intended local image first")
    try:
        entries = json.loads(inspected.stdout)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError
        entry = entries[0]
        image_id = entry["Id"]
        _require_image_id(image_id)
        config = entry["Config"]
        labels = config.get("Labels") or {}
        if entry.get("Os") != "linux" or entry.get("Architecture") != "arm64":
            raise ValueError("image must target linux/arm64")
        if any(labels.get(key) != value for key, value in EXPECTED_LABELS[harness].items()):
            raise ValueError("image labels do not match the pinned harness")
        if config.get("Entrypoint") != ENTRYPOINTS[harness]:
            raise ValueError("image must retain the native entrypoint")
        if config.get("WorkingDir") != "/workspace":
            raise ValueError("image working directory must be /workspace")
        if config.get("User") not in (None, "", "root", "0", "0:0"):
            raise ValueError("image must retain root startup and native UID handling")
        volumes = set(config.get("Volumes") or {})
        if volumes != ({"/opt/data"} if harness == "hermes" else set()):
            raise ValueError("unexpected image state volumes")
        environment = dict(item.split("=", 1) for item in config.get("Env", []))
        if any(
            value and re.search(r"(API_KEY|TOKEN|PASSWORD|SECRET)$", key)
            for key, value in environment.items()
        ):
            raise ValueError("image contains credential-like environment defaults")
        if harness == "hermes" and environment.get("HERMES_HOME") != "/opt/data":
            raise ValueError("unexpected native HERMES_HOME")
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("docker image inspect returned invalid metadata") from exc
    container = f"mosaic-config-{uuid.uuid4().hex}"
    path = "/root/.pi/agent/models.json" if harness == "pi" else "/opt/data/config.yaml"
    try:
        created = _docker("create", "--platform", PLATFORM, "--name", container, image_id)
        if created.returncode != 0:
            raise RuntimeError("could not create image configuration-check container")
        copied = _docker("cp", f"{container}:{path}", "-")
        if copied.returncode != 0:
            raise ValueError("baked harness configuration is missing")
        with tarfile.open(fileobj=io.BytesIO(copied.stdout)) as archive:
            members = archive.getmembers()
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("configuration archive must contain one regular file")
            source = archive.extractfile(members[0])
            if source is None:
                raise ValueError("configuration file missing")
            with source:
                body = source.read()
        valid = (
            json.loads(body) == PI_CONFIG
            if harness == "pi"
            else body.decode("utf-8") == HERMES_CONFIG
        )
        if not valid:
            raise ValueError("baked harness configuration differs from the study settings")
    finally:
        _remove_stopped(container)
    return {
        "image_id": image_id,
        "harness": harness,
        "platform": PLATFORM,
        "repo_digests": entry.get("RepoDigests"),
        "labels": {key: labels[key] for key in EXPECTED_LABELS[harness]},
        "entrypoint": ENTRYPOINTS[harness],
        "config_sha256": hashlib.sha256(body).hexdigest(),
        "launch_args": list(PI_ARGS if harness == "pi" else HERMES_ARGS),
    }


@dataclass
class ParsedEvent:
    kind: str
    event_type: str | None = None
    payload: dict[str, Any] | None = None


@dataclass
class ParsedStream:
    harness: str
    events: list[ParsedEvent] = field(default_factory=list)
    malformed_lines: int = 0
    unknown_events: int = 0


class TerminalOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["completed", "unscored"]
    unscored_reason: str | None = None
    final_answer: str | None = None
    sentinel_injected: bool = False
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_kind: Literal["unknown", "catalog_estimate", "main_loop_only"] = "unknown"


def parse_event_stream(stdout: str, harness: str) -> ParsedStream:
    parsed = ParsedStream(harness=harness)
    native_boot_lines: set[int] = set()
    if harness == "hermes":
        for match in HERMES_SKILLS_BOOT_PATTERN.finditer(stdout):
            if str(len(match["skills"].splitlines())) == match["count"]:
                first = stdout.count("\n", 0, match.start())
                native_boot_lines.update(range(first, first + match[0].count("\n") + 1))
    seen_json = False
    tirith_init_window = False
    for index, line in enumerate(stdout.split("\n")):
        if not line.strip():
            continue
        try:
            payload = json.loads(line.strip())
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            is_boot = line.startswith("[stage2]") or (
                harness == "hermes"
                and (index in native_boot_lines or line in HERMES_BOOT_DIAGNOSTICS)
            )
            init_warning = tirith_init_window and line.removesuffix("\r") == HERMES_TIRITH_WARNING
            tirith_init_window = False
            if (not seen_json and is_boot) or init_warning:
                parsed.events.append(ParsedEvent(kind="boot"))
            else:
                parsed.events.append(ParsedEvent(kind="malformed"))
                parsed.malformed_lines += 1
            continue
        tirith_init_window = (
            harness == "hermes"
            and not seen_json
            and payload.get("type") == "system"
            and payload.get("subtype") == "init"
            and payload.get("model") == MODEL
            and isinstance(payload.get("session_id"), str)
            and bool(payload["session_id"])
        )
        seen_json = True
        event_type = payload.get("type")
        known = isinstance(event_type, str) and event_type in KNOWN_EVENT_TYPES[harness]
        parsed.events.append(
            ParsedEvent(
                kind="event" if known else "unknown",
                event_type=event_type,
                payload=payload,
            )
        )
        if not known:
            parsed.unknown_events += 1
    return parsed


def _message_of(event: ParsedEvent) -> dict[str, Any]:
    message = (event.payload or {}).get("message")
    return message if isinstance(message, dict) else {}


def _text_blocks(blocks: object) -> str:
    if not isinstance(blocks, list):
        return ""
    return "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _pi_metadata(message: Mapping[str, Any]) -> dict[str, Any]:
    return {key: message[key] for key in PI_METADATA_KEYS if key in message}


def _pi_usage(message: Mapping[str, Any]) -> dict[str, Any]:
    usage = message.get("usage")
    return dict(usage) if isinstance(usage, dict) else {}


def _hermes_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload[key] for key in HERMES_METADATA_KEYS if key in payload}


def _hermes_init_metadata(parsed: ParsedStream) -> dict[str, Any]:
    for event in parsed.events:
        if event.kind == "event" and event.event_type == "system":
            payload = event.payload or {}
            if payload.get("subtype") == "init":
                init = {
                    key: payload[key]
                    for key in ("model", "provider", "session_id")
                    if key in payload
                }
                return {"init": init} if init else {}
    return {}


def _hermes_usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    tokens = payload.get("tokens")
    return dict(tokens) if isinstance(tokens, dict) else {}


def _unscored(reason: str, parsed: ParsedStream) -> TerminalOutcome:
    if parsed.harness == "pi":
        message = next(
            (
                _message_of(event)
                for event in reversed(parsed.events)
                if event.event_type == "message_end"
                and _message_of(event).get("role") == "assistant"
            ),
            {},
        )
        metadata = _pi_metadata(message)
        usage = _pi_usage(message)
        usage_kind = "catalog_estimate"
    else:
        payload = next(
            (
                event.payload or {}
                for event in reversed(parsed.events)
                if event.event_type == "result"
            ),
            {},
        )
        metadata = {**_hermes_metadata(payload), **_hermes_init_metadata(parsed)}
        usage = _hermes_usage(payload)
        usage_kind = "main_loop_only"
    return TerminalOutcome(
        status="unscored",
        unscored_reason=reason,
        model_metadata=metadata,
        usage_metadata=usage,
        usage_kind=usage_kind if usage else "unknown",
    )


def _extract_pi_outcome(parsed: ParsedStream) -> TerminalOutcome:
    events = parsed.events
    assistants = [
        (index, event)
        for index, event in enumerate(events)
        if event.kind == "event"
        and event.event_type == "message_end"
        and _message_of(event).get("role") == "assistant"
    ]
    if not assistants:
        return _unscored("no_assistant_message", parsed)
    last_index, last_event = assistants[-1]
    message = _message_of(last_event)
    metadata, usage = _pi_metadata(message), _pi_usage(message)
    usage_kind = "catalog_estimate" if usage else "unknown"
    blocks = message.get("content")
    if isinstance(blocks, list) and any(
        isinstance(block, dict) and block.get("type") == "toolCall" for block in blocks
    ):
        return _unscored("pending_tool_calls", parsed)
    stop_reason = message.get("stopReason")
    if stop_reason != "stop":
        reason = "limit_ended" if stop_reason == "length" else f"stop_reason_{stop_reason}"
        return _unscored(reason, parsed)
    settled = [
        index
        for index, event in enumerate(events)
        if event.kind == "event" and event.event_type == "agent_settled"
    ]
    if not settled:
        return _unscored("not_settled", parsed)
    settled_index = settled[-1]
    if settled_index < len(events) - 1:
        return _unscored("activity_after_settled", parsed)
    if any(
        event.event_type in PI_UNFINISHED_AFTER_ANSWER
        for event in events[last_index + 1 : settled_index]
    ):
        return _unscored("unfinished_activity", parsed)
    text = _text_blocks(message.get("content"))
    outcome = TerminalOutcome(
        status="completed",
        final_answer=text,
        model_metadata=metadata,
        usage_metadata=usage,
        usage_kind=usage_kind,
    )
    if not text.strip():
        if message.get("rawStopReason") == "refusal":
            outcome.final_answer = SAFEGUARD_SENTINEL
            outcome.sentinel_injected = True
        else:
            return _unscored("empty_without_refusal_metadata", parsed)
    return outcome


def _extract_hermes_outcome(parsed: ParsedStream, process_exit_code: int) -> TerminalOutcome:
    events = parsed.events
    results = [
        (index, event)
        for index, event in enumerate(events)
        if event.kind == "event" and event.event_type == "result"
    ]
    payload = (results[-1][1].payload or {}) if results else {}
    metadata = {**_hermes_metadata(payload), **_hermes_init_metadata(parsed)}
    usage = _hermes_usage(payload)
    usage_kind = "main_loop_only" if usage else "unknown"
    result_code = payload.get("exit_code") if len(results) == 1 else None
    if process_exit_code == 130 or result_code == 130:
        return _unscored("harness_interrupted", parsed)
    if not results:
        return _unscored("terminal_result_missing", parsed)
    if len(results) > 1:
        return _unscored("terminal_result_duplicate", parsed)
    if results[0][0] < len(events) - 1:
        return _unscored("activity_after_terminal_result", parsed)
    if process_exit_code != 0 or result_code != 0:
        return _unscored("harness_failure_or_limit", parsed)
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return _unscored("empty_without_refusal_metadata", parsed)
    return TerminalOutcome(
        status="completed",
        final_answer=text,
        model_metadata=metadata,
        usage_metadata=usage,
        usage_kind=usage_kind,
    )


def extract_terminal_outcome(
    parsed: ParsedStream,
    harness: str,
    process_exit_code: int,
) -> TerminalOutcome:
    if parsed.malformed_lines or parsed.unknown_events:
        return _unscored("unparsed_protocol_events", parsed)
    if harness == "pi":
        if process_exit_code != 0:
            return _unscored("process_failure", parsed)
        return _extract_pi_outcome(parsed)
    return _extract_hermes_outcome(parsed, process_exit_code)


class _RedactingWriter:
    def __init__(self, path: Path, secret: bytes) -> None:
        self._handle = path.open("xb")
        self._secret = secret
        self._pending = bytearray()

    def write(self, chunk: bytes) -> None:
        self._pending.extend(chunk)
        data = bytes(self._pending)
        cut = len(data) - (len(self._secret) - 1)
        flush_end = 0
        pos = 0
        while True:
            index = data.find(self._secret, pos)
            if index == -1:
                break
            flush_end = index + len(self._secret)
            pos = flush_end
        if flush_end == 0:
            if cut > 0:
                self._handle.write(data[:cut])
                del self._pending[:cut]
                self._flush()
            return
        self._handle.write(data[:flush_end].replace(self._secret, REDACTION.encode()))
        del self._pending[:flush_end]
        self._flush()

    def _flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.write(bytes(self._pending).replace(self._secret, REDACTION.encode()))
        self._pending.clear()
        self._flush()
        self._handle.close()


def _drain_pipe(pipe: Any, sink: _RedactingWriter, errors: list[str]) -> None:
    try:
        while True:
            chunk = pipe.read1(65536)
            if not chunk:
                return
            sink.write(chunk)
    except BaseException as exc:
        errors.append(type(exc).__name__)
    finally:
        pipe.close()


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, REDACTION)
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    return value


def _write_json(path: Path, record: dict[str, Any], secret: str) -> None:
    data = json.dumps(_redact(record, secret), indent=2, sort_keys=True).encode()
    with path.open("xb") as handle:
        handle.write(data + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _systemic_reason(returncode: int, stderr: str, parsed: ParsedStream) -> str | None:
    if returncode == 125:
        return "systemic_setup"
    parts = [stderr]
    for event in parsed.events:
        payload = event.payload or {}
        for source in (payload, _message_of(event)):
            for key in ("errorMessage", "error", "finalError"):
                value = source.get(key)
                if isinstance(value, str):
                    parts.append(value)
    text = "\n".join(parts).lower()
    if "credentials or agent init failed" in text:
        return "systemic_setup"
    if re.search(r"\b(?:401|403)\b", text) or any(
        marker in text
        for marker in (
            "unauthorized",
            "forbidden",
            "invalid api key",
            "invalid x-api-key",
            "authentication failed",
        )
    ):
        return "systemic_auth"
    return None


@contextmanager
def _trial_cleanup(output_dir: Path, container: str, secret: str) -> Iterator[None]:
    try:
        yield
    finally:
        try:
            if not (output_dir / "terminal.json").exists():
                _write_json(
                    output_dir / "terminal.json",
                    {
                        "final_answer": None,
                        "status": "unscored",
                        "empty_safeguard_refusal": False,
                        "metadata": {"unscored_reason": "runner_failed_before_terminal"},
                    },
                    secret,
                )
            _remove_stopped(container)
        except BaseException as exc:
            try:
                _write_json(
                    output_dir / "cleanup.json",
                    {
                        "removed": False,
                        "container_name": container,
                    },
                    secret,
                )
            finally:
                raise RuntimeError(
                    f"cleanup incomplete for {container}; stop and review retained evidence"
                ) from exc
        _write_json(output_dir / "cleanup.json", {"removed": True}, secret)


def collect_trial(
    example: dict[str, str],
    configuration: dict[str, Any],
    harness: str,
    image_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    _validate_harness(harness)
    _require_image_id(image_id)
    secret = os.environ.get(CREDENTIAL_ENV)
    if not secret:
        raise ValueError("OPENROUTER_API_KEY must be set for live collection")
    if configuration.get("intent") not in ("benign", "explicit", "adversarial"):
        raise ValueError("invalid live intent")
    for key in ("distractor", "headers", "rules"):
        if type(configuration.get(key)) is not bool:
            raise ValueError(f"live {key} must be boolean")
    if configuration.get("repetition") != 0:
        raise ValueError("live collection uses one repetition (index 0)")
    output_dir = output_dir.resolve()
    if ":" in str(output_dir) or "\n" in str(output_dir):
        raise ValueError("output path cannot contain Docker mount delimiters")
    documents = render_documents(
        example,
        headers=configuration["headers"],
        distractor=configuration["distractor"],
    )
    query = f"{render_query(example, configuration['intent'])}\n\n{LOCATION_HINT}"
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    inputs = output_dir / "inputs"
    docs = inputs / "docs"
    docs.mkdir(parents=True)
    docs.chmod(0o755)
    for name, text in documents:
        if name not in ("docA.txt", "docB.txt", "docC.txt"):
            raise ValueError("unexpected rendered document filename")
        with (docs / name).open("x", encoding="utf-8") as handle:
            handle.write(text)
        (docs / name).chmod(0o444)
    policy = inputs / "AGENTS.md"
    if configuration["rules"]:
        with policy.open("x", encoding="utf-8") as handle:
            handle.write(aggregation_policy())
        policy.chmod(0o444)
    container = f"mosaic-live-{harness}-{uuid.uuid4().hex}"
    argv = [
        "docker",
        "run",
        "--platform",
        PLATFORM,
        "-w",
        "/workspace",
        "--name",
        container,
        "--cidfile",
        str(output_dir / "cidfile"),
        "-e",
        CREDENTIAL_ENV,
        "-v",
        f"{docs}:/workspace/docs:ro",
    ]
    if configuration["rules"]:
        argv += ["-v", f"{policy}:/workspace/AGENTS.md:ro"]
    argv += [image_id]
    argv += [
        query if arg == "<query>" else arg for arg in (PI_ARGS if harness == "pi" else HERMES_ARGS)
    ]
    _write_json(
        output_dir / "start.json",
        {
            "container_name": container,
            "image_id": image_id,
            "harness": harness,
            "configuration": configuration,
            "launch_argv": argv,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        secret,
    )
    with _trial_cleanup(output_dir, container, secret):
        return _invoke_trial(argv, output_dir, container, harness, image_id, secret)


def _invoke_trial(
    argv: list[str],
    output_dir: Path,
    container: str,
    harness: str,
    image_id: str,
    secret: str,
) -> dict[str, Any]:
    stdout_writer = _RedactingWriter(output_dir / "stdout.txt", secret.encode())
    stderr_writer = _RedactingWriter(output_dir / "stderr.txt", secret.encode())
    errors: list[str] = []
    threads: list[threading.Thread] = []
    proc = None
    returncode: int | None = None
    failure: BaseException | None = None
    settlement: list[str] = []
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threads = [
            threading.Thread(target=_drain_pipe, args=(proc.stdout, stdout_writer, errors)),
            threading.Thread(target=_drain_pipe, args=(proc.stderr, stderr_writer, errors)),
        ]
        for thread in threads:
            thread.start()
        returncode = proc.wait()
    except BaseException as exc:
        failure = exc
        if proc is not None:
            try:
                stopped = subprocess.run(
                    ["docker", "stop", container],
                    capture_output=True,
                    timeout=30,
                )
                settlement.append(f"docker stop exit code: {stopped.returncode}")
            except (OSError, subprocess.TimeoutExpired) as stop_error:
                settlement.append(f"docker stop failed: {type(stop_error).__name__}")
            try:
                proc.terminate()
            except OSError as terminate_error:
                settlement.append(f"client terminate failed: {type(terminate_error).__name__}")
            try:
                returncode = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                returncode = proc.wait()
    finally:
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        stdout_writer.close()
        stderr_writer.close()
    stdout = (output_dir / "stdout.txt").read_text(encoding="utf-8", errors="replace")
    stderr = (output_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    parsed = parse_event_stream(stdout, harness)
    outcome = extract_terminal_outcome(
        parsed,
        harness,
        returncode if returncode is not None else -1,
    )
    systemic = _systemic_reason(returncode if returncode is not None else -1, stderr, parsed)
    if failure is not None or errors:
        outcome = _unscored(
            "docker_launch_failure"
            if isinstance(failure, OSError)
            else "runner_interrupted"
            if failure is not None
            else "output_capture_failure",
            parsed,
        )
    result = {
        "final_answer": outcome.final_answer,
        "status": outcome.status,
        "empty_safeguard_refusal": outcome.sentinel_injected,
        "metadata": {
            "image_id": image_id,
            "harness": harness,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "model": outcome.model_metadata or None,
            "finish": {
                key: outcome.model_metadata.get(key)
                for key in ("stopReason", "rawStopReason", "exit_code")
            }
            if outcome.model_metadata
            else None,
            "usage": outcome.usage_metadata or None,
            "usage_kind": outcome.usage_kind,
            "process_exit_code": returncode,
            "unscored_reason": outcome.unscored_reason,
            "systemic_reason": systemic,
            "malformed_events": parsed.malformed_lines,
            "unknown_events": parsed.unknown_events,
            "settlement": settlement,
        },
    }
    _write_json(output_dir / "terminal.json", result, secret)
    if failure is not None:
        raise failure
    if errors:
        raise RuntimeError("output capture failed; unscored evidence retained")
    if systemic:
        raise RuntimeError(f"{systemic}; stop collection and review retained trial evidence")
    return result
