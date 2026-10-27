<p align="center">
  <img src="assets/animation.gif" width="100%" alt="Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems" />
</p>

<h1 align="center">🔒 Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.30418"><img src="https://img.shields.io/badge/arXiv-2610.30418-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv: 2610.30418" /></a>
  <a href="https://huggingface.co/papers/2610.30418"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-FFD21E" alt="Hugging Face Paper" /></a>
  <a href="https://huggingface.co/datasets/ZQ-Dev/agent-mosaic-examples"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E" alt="Hugging Face Dataset" /></a>
  <a href="#install"><img src="https://img.shields.io/badge/Getting%20Started-Install-76B900" alt="Getting Started" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License" /></a>
</p>

## 📰 News

- **2026-10-30** — Research repository released.

## Introduction

Collect and automatically score responses for the **Agent Mosaic benchmark**, from
[Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems](https://arxiv.org/abs/2610.30418).
Run scripted simulated conversations or live Pi/Hermes agents, then assess their
final answers with the paper's automated judge.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.13 or newer, on a POSIX host.
- An OpenRouter account/key and access to the model/provider routes below.
- For live collection: Docker with Linux ARM64 support and sufficient local disk/memory.
  Build the images locally for `linux/arm64`.

The workflows use the paper's model profiles and harness configurations, listed
below. The paper used a 65,536-token serving ceiling.

> [!NOTE]
> **Collection and scoring cost money.** This repo assumes you manage spending through your
> OpenRouter account, keep API keys private, and inspect run outputs before sharing.
> Supply the key through your runtime environment. In Bash or Zsh:

```bash
read -r -s OPENROUTER_API_KEY
export OPENROUTER_API_KEY
```

## Install

Clone this repository, then:

```sh
uv sync --locked
```

The commands below load `ZQ-Dev/agent-mosaic-examples` directly with Hugging Face
`datasets.load_dataset`, which handles downloading and caching the `test` split.
The benchmark contains 30 examples (ten per domain), with 17 fields per example.

## Collect and score

Collection and scoring use paid model APIs. Choose a new output directory for each
run; use `--resume` to continue an existing run with matching inputs and settings.

### Simulated conversations

```sh
uv run --locked agent-mosaic simulate --dataset ZQ-Dev/agent-mosaic-examples \
  --generator gpt --out outputs/sim-gpt

uv run --locked agent-mosaic score --dataset ZQ-Dev/agent-mosaic-examples \
  --responses outputs/sim-gpt --out outputs/scores-sim-gpt
```

Repeat with `--generator claude` and `--generator qwen`, choosing a new output path.
Each generator has 30 examples × 24 configurations × five repetitions = 3,600 trials;
all three have 10,800. Configuration axes are intent, distractor, classification
headers and aggregation rules. Documents are delivered through a fixed scripted
conversation, with trials shuffled using seed 0.

| Generator | OpenRouter model | Provider | Controls |
| --- | --- | --- | --- |
| GPT | `openai/gpt-5.6-sol` | `openai` | medium reasoning |
| Claude | `anthropic/claude-sonnet-5` | `anthropic` | medium reasoning |
| Qwen | `qwen/qwen3.5-27b` | `alibaba` | reasoning on, temperature 1, top_p .95, top_k 20, presence penalty 1.5 |

Simulation and scoring use explicit provider routing with fallback disabled.

### Live Pi and Hermes

Build the Pi and Hermes images locally:

```sh
docker build --platform linux/arm64 -t agent-mosaic-pi:1.0.4 \
  -f containers/live_agents/pi.Dockerfile containers/live_agents

docker build --platform linux/arm64 -t agent-mosaic-hermes:0.21.5 \
  -f containers/live_agents/hermes.Dockerfile containers/live_agents
```

Run the collections:

```sh
uv run --locked agent-mosaic live --dataset ZQ-Dev/agent-mosaic-examples \
  --harness pi --image agent-mosaic-pi:1.0.4 --out outputs/live-pi

uv run --locked agent-mosaic live --dataset ZQ-Dev/agent-mosaic-examples \
  --harness hermes --image agent-mosaic-hermes:0.21.5 --out outputs/live-hermes

uv run --locked agent-mosaic score --dataset ZQ-Dev/agent-mosaic-examples \
  --responses outputs/live-pi --out outputs/scores-live-pi
  
uv run --locked agent-mosaic score --dataset ZQ-Dev/agent-mosaic-examples \
  --responses outputs/live-hermes --out outputs/scores-live-hermes
```

Each harness runs 720 trials: every configuration once per example, using GPT/OpenAI
with medium reasoning. Image identities are checked and recorded. Each trial starts
with fresh isolated state and A/B/optional distractor documents mounted read-only.
A neutral hint points the agent to `/workspace/docs/`, where it chooses which
documents to read. Rules-on trials deliver the study aggregation policy through
native `AGENTS.md` discovery.

Both harnesses retain their native prompts, tools, skills and auxiliary behavior.
Pi applies OpenAI routing with fallback disabled; Hermes applies OpenAI-only routing
through its native configuration.

### Shared scoring

Both output types use `xiaomi/mimo-v2.6-pro` via `xiaomi/fp8`, reasoning on,
temperature 1 and top_p .95, with the study prompts and strict two-boolean parser.
The judge compares each final answer with its example's forbidden composite and
returns separate mosaic-event and refusal labels; both can be true. Missing,
invalid or exhausted responses are exported as `unscored` with null labels.

## Outputs and interruption

Each run directory stores its authoritative state in `run.sqlite`: the frozen
manifest, plan, attempts and results. Completed passes export `responses.jsonl` or
`scores.jsonl`. Rows include stable trial/example IDs, configuration, repetition,
model and harness identity, final answer, status, input/prompt/profile/image identities
and available attempt, finish, and usage metadata. Hub inputs record their repository,
requested revision, split, and a SHA-256 of the validated records in `dataset_source`.
Score rows contain separate `mosaic_event` and `refusal` booleans.

Pi reports catalog-based usage estimates; Hermes reports main-loop usage.
Track charges through your provider's billing records.

Press Ctrl+C to stop collection and retain the run directory. Add `--resume` to the
same command using unchanged inputs, settings, code and image identities. Resumption
retains completed trials and existing attempt counts.


*Note:* Downloaded inputs, outputs, credentials, environments and caches are Git-ignored.

## Offline smoke check

```sh
env -u OPENROUTER_API_KEY uv run --locked python scripts/smoke.py
```

The offline smoke exercises the CLI with invented data, synthetic responses and
temporary outputs. Run this before any live experiments.

## Ethics & License

The benchmark dataset is entirely synthetic. Its documents, entities, and
classification labels are fictional experimental stimuli. Any resemblance to real
entities is purely coincidental. Medical, military, and cybersecurity content is research
material, not professional advice. `SECRET` and `TOP_SECRET` are experimental
labels. The dataset contains no actual confidential records. Examples were generated
with Claude Opus 4.6 and manually reviewed by the authors for correctness and realism.

First-party code: [MIT](LICENSE), copyright © 2026 Daniel C. Ruiz, Luca Luceri,
and Emilio Ferrara. The live images include components under their upstream licenses:
[Pi (MIT)](https://github.com/badlogic/pi-mono/blob/7c10bd4337495ee613f2224843ecdf349b80d1df/LICENSE),
[Hermes (MIT)](https://github.com/NousResearch/hermes-agent/blob/f97608f178d1ffeca59860195ab7da295f7c8e5f/LICENSE)
and Tirith 0.4.2 ([source](https://github.com/sheeki03/tirith/tree/v0.4.2),
[AGPL](https://github.com/sheeki03/tirith/blob/v0.4.2/LICENSE-AGPL),
[commercial license](https://github.com/sheeki03/tirith/blob/v0.4.2/LICENSE-COMMERCIAL),
[NOTICE](https://github.com/sheeki03/tirith/blob/v0.4.2/NOTICE)).
When redistributing built images, retain upstream licenses and notices and fulfill
their source and redistribution requirements.

## Citation

Please cite our paper!

```bibtex
@misc{ruiz2026agentmosaic,
  title={Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems},
  author={Daniel C. Ruiz and Luca Luceri and Emilio Ferrara},
  year={2026},
  eprint={2610.30418},
  archivePrefix={arXiv},
  primaryClass={cs.CR},
  url={https://arxiv.org/abs/2610.30418}
}
```
