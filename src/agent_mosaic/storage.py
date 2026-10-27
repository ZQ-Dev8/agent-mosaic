from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_mosaic.dataset import unique_object

SAFEGUARD_SENTINEL = "SAFEGUARD TRIPPED: Sorry, I cannot help with that."


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_text(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def json_hash(value: object) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def check_path(path: Path) -> None:
    if path.resolve() != path or path.is_symlink():
        raise ValueError("output paths must not contain symlinks; use the canonical path")
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError("output files must be regular, unshared files")


def check_credentials(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "api_key",
                "authorization",
                "proxy_authorization",
                "extra_headers",
                "cookie",
                "set_cookie",
                "x_api_key",
                "password",
                "access_token",
            } or (normalized == "headers" and type(child) is not bool):
                raise ValueError("credential-bearing metadata is not allowed")
            check_credentials(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_credentials(child)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class IdentityChanged(ValueError):
    pass


class RunStore:
    def __init__(self, root: Path, manifest: dict, plan: list[dict], resume: bool) -> None:
        self.root = Path(os.path.abspath(root))
        if self.root.resolve() != self.root or self.root.is_symlink():
            raise ValueError("output directory must use a canonical, non-symlink path")
        self.root.mkdir(parents=True, exist_ok=True)
        if not resume and any(self.root.iterdir()):
            raise ValueError(
                "output directory must be new or empty; use --resume for an existing run"
            )
        if resume and not (self.root / "run.sqlite").is_file():
            raise ValueError("cannot resume: run.sqlite is missing")
        for name in (
            ".lock",
            "run.sqlite",
            "run.sqlite-journal",
            "run.sqlite-wal",
            "run.sqlite-shm",
        ):
            check_path(self.root / name)
        self.lock = (self.root / ".lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("output directory is already in use") from None
        self.conn: sqlite3.Connection | None = None
        try:
            if not resume and (self.root / "run.sqlite").exists():
                raise ValueError("run already exists")
            self.conn = sqlite3.connect(self.root / "run.sqlite")
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            check_credentials(manifest)
            if not resume:
                self.conn.executescript("""
                    CREATE TABLE run (manifest TEXT NOT NULL, digest TEXT NOT NULL,
                                      started_at TEXT NOT NULL);
                    CREATE TABLE trials (trial_id TEXT PRIMARY KEY, position INTEGER UNIQUE NOT NULL,
                                         plan TEXT NOT NULL, plan_hash TEXT NOT NULL,
                                         result TEXT, result_hash TEXT);
                    CREATE TABLE attempts (
                        trial_id TEXT NOT NULL REFERENCES trials(trial_id), number INTEGER NOT NULL,
                        request TEXT NOT NULL, request_hash TEXT NOT NULL, started_at TEXT NOT NULL,
                        outcome TEXT, outcome_hash TEXT, PRIMARY KEY (trial_id, number));
                    CREATE TABLE identities (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                """)
                with self.conn:
                    self.conn.execute(
                        "INSERT INTO run VALUES (?,?,?)",
                        (
                            json_text(manifest),
                            json_hash(manifest),
                            now(),
                        ),
                    )
                    self.conn.executemany(
                        "INSERT INTO trials VALUES (?,?,?,?,NULL,NULL)",
                        [
                            (trial["trial_id"], position, json_text(trial), json_hash(trial))
                            for position, trial in enumerate(plan)
                        ],
                    )
                sync_directory(self.root)
            else:
                saved = self.conn.execute("SELECT * FROM run").fetchall()
                if (
                    len(saved) != 1
                    or saved[0]["manifest"] != json_text(manifest)
                    or (saved[0]["digest"] != json_hash(manifest))
                ):
                    raise ValueError(
                        "resume mismatch: inputs, prompts, code, images, or settings changed"
                    )
                rows = self.conn.execute("SELECT * FROM trials ORDER BY position").fetchall()
                if len(rows) != len(plan) or any(
                    row["plan"] != json_text(trial) or row["plan_hash"] != json_hash(trial)
                    for row, trial in zip(rows, plan)
                ):
                    raise ValueError("resume mismatch: saved trial plan changed")
                for row in rows:
                    if (
                        row["result"] is not None
                        and json_hash(json.loads(row["result"])) != (row["result_hash"])
                    ):
                        raise ValueError("stored result changed")
                if self.conn.execute("SELECT 1 FROM identities WHERE name='__paused__'").fetchone():
                    raise ValueError("run paused for review; no automatic resume")
                for row in self.conn.execute("SELECT * FROM attempts"):
                    if json_hash(json.loads(row["request"])) != row["request_hash"]:
                        raise ValueError("stored request changed")
                    if row["outcome"] is None:
                        raise ValueError(
                            "incomplete dispatch requires review; no automatic resubmission"
                        )
                    if json_hash(json.loads(row["outcome"])) != row["outcome_hash"]:
                        raise ValueError("stored attempt changed")
                    if not json.loads(row["outcome"]).get("retryable", False):
                        trial_row = self.conn.execute(
                            "SELECT result FROM trials WHERE trial_id=?", (row["trial_id"],)
                        ).fetchone()
                        if trial_row["result"] is None:
                            raise ValueError("unresolved dispatch requires review")
            export_name = "scores.jsonl" if manifest["kind"] == "score" else "responses.jsonl"
            check_path(self.root / export_name)
            if (self.root / export_name).exists():
                self.export(export_name)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
        self.lock.close()

    def result(self, trial_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT result FROM trials WHERE trial_id=?", (trial_id,)
        ).fetchone()
        return json.loads(row[0]) if row[0] is not None else None

    def history(self, trial_id: str) -> list[dict]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM attempts WHERE trial_id=? ORDER BY number", (trial_id,)
            )
        ]

    def begin_attempt(self, trial_id: str, request: dict) -> int:
        check_credentials(request)
        if self.result(trial_id) is not None:
            raise ValueError("cannot redispatch a completed trial")
        history = self.history(trial_id)
        if history and (
            history[-1]["outcome"] is None
            or not json.loads(history[-1]["outcome"]).get("retryable", False)
        ):
            raise ValueError("previous attempt cannot be automatically retried")
        if any(row["request_hash"] != json_hash(request) for row in history):
            raise ValueError("resume mismatch: attempt request changed")
        number = len(history) + 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO attempts VALUES (?,?,?,?,?,NULL,NULL)",
                (
                    trial_id,
                    number,
                    json_text(request),
                    json_hash(request),
                    now(),
                ),
            )
        return number

    def finish_attempt(
        self, trial_id: str, number: int, outcome: dict, result: dict | None
    ) -> None:
        check_credentials(outcome)
        check_credentials(result)
        outcome = {**outcome, "ended_at": now()}
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE attempts SET outcome=?, outcome_hash=? "
                "WHERE trial_id=? AND number=? AND outcome IS NULL",
                (json_text(outcome), json_hash(outcome), trial_id, number),
            )
            if cursor.rowcount != 1:
                raise ValueError("attempt must be in flight")
            if result is not None:
                cursor = self.conn.execute(
                    "UPDATE trials SET result=?,result_hash=? WHERE trial_id=? AND result IS NULL",
                    (json_text(result), json_hash(result), trial_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("trial result is immutable")

    def pause(self, reason: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO identities VALUES ('__paused__',?)", (reason,))

    def observe_identity(self, name: str, value: str) -> None:
        previous = self.conn.execute(
            "SELECT value FROM identities WHERE name=?", (name,)
        ).fetchone()
        if previous is not None and previous[0] != value:
            self.pause(name)
            raise IdentityChanged("served model/provider changed; run paused for review")
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO identities VALUES (?,?)", (name, value))

    def export(self, filename: str) -> Path:
        rows = self.conn.execute("SELECT result FROM trials ORDER BY position").fetchall()
        if any(row[0] is None for row in rows):
            raise ValueError("cannot export an incomplete run")
        content = "".join(row[0] + "\n" for row in rows).encode("utf-8")
        path = self.root / filename
        check_path(path)
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("existing export differs or is incomplete; retained for review")
            return path
        with path.open("xb") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        sync_directory(self.root)
        return path


def load_responses(path: Path, dataset_sha256: str) -> tuple[list[dict], str]:
    if path.is_dir():
        path = path / "responses.jsonl"
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        value = json.loads(line, object_pairs_hook=unique_object)
        if not isinstance(value, dict) or value.get("dataset_sha256") != dataset_sha256:
            raise ValueError("responses do not match this dataset hash")
        if (
            any(
                not isinstance(value.get(field), str) or not value[field].strip()
                for field in ("run_id", "trial_id", "example_id")
            )
            or value.get("status") not in ("completed", "unscored")
            or type(value.get("empty_safeguard_refusal")) is not bool
            or "final_answer" not in value
            or (
                value.get("final_answer") is not None and not isinstance(value["final_answer"], str)
            )
        ):
            raise ValueError("invalid response record")
        final = value.get("final_answer")
        if value["status"] == "completed" and (not isinstance(final, str) or not final.strip()):
            raise ValueError("completed response requires a nonempty final answer")
        if value["empty_safeguard_refusal"] and final != SAFEGUARD_SENTINEL:
            raise ValueError("invalid safeguard-refusal sentinel")
        check_credentials(value)
        records.append(value)
    if not records or len({row["trial_id"] for row in records}) != len(records):
        raise ValueError("responses must contain unique trial IDs")
    return records, hashlib.sha256(raw).hexdigest()
