from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Agent Mosaic benchmark response collection and automated scoring."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    simulate = commands.add_parser(
        "simulate", help="Collect simulated responses via OpenRouter (paid)."
    )
    simulate.add_argument(
        "--generator", choices=("gpt", "claude", "qwen"), help="Default: all three."
    )
    live = commands.add_parser(
        "live", help="Collect native live-agent responses via Docker (paid)."
    )
    live.add_argument("--harness", choices=("pi", "hermes"), required=True)
    live.add_argument(
        "--image", required=True, help="Locally built study image tag; identity is checked."
    )
    scorer = commands.add_parser(
        "score", help="Score response finals with the adopted judge (paid)."
    )
    scorer.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Collection directory or its immutable responses.jsonl export.",
    )
    for command in (simulate, live, scorer):
        command.add_argument(
            "--dataset", required=True, help="Hugging Face dataset ID or local 17-field JSONL file."
        )
        command.add_argument("--revision", help="Hugging Face dataset commit, tag or branch.")
        command.add_argument(
            "--out", type=Path, required=True, help="New/empty researcher-owned directory."
        )
        command.add_argument(
            "--resume", action="store_true", help="Continue only with unchanged inputs/settings."
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            from agent_mosaic.scoring import score

            result = score(
                args.dataset, args.responses, args.out, resume=args.resume, revision=args.revision
            )
        else:
            from agent_mosaic.collection import collect

            result = collect(
                args.dataset,
                args.out,
                generator=getattr(args, "generator", None),
                harness=getattr(args, "harness", None),
                image=getattr(args, "image", None),
                resume=args.resume,
                revision=args.revision,
            )
    except KeyboardInterrupt:
        parser.exit(
            130, "Interrupted. Retained dispatches may require review; no automatic rerun.\n"
        )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2))
