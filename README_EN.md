# ark-solution-lab

> Turn one sentence from a customer into a **citation-backed solution draft** — with a reproducible
> experiment that answers a harder question: **how many agents does this problem actually need?**

**Status**: core code and experiments complete — 257 unit tests, **zero third-party dependencies** in the core pipeline.

---

## Headline result

Three pipelines, the same 20 customer requirements, one yardstick:

| Pipeline | Avg. calls | Avg. tokens | Unsupported-claim rate |
|---|---|---|---|
| `no-evidence` — answer from memory, no retrieval | 1.7 | 845 | **100%** (42/42) |
| `evidence-once` — retrieve once, write once | 1.7 | 1,499 | **0%** |
| `multi` — parallel read-only research + single-threaded writing + independent review | **4.5** | **4,585** | **0%** |

**Three findings:**

1. **Retrieval, not orchestration, is what stops fabrication.** Adding evidence drops the
   unsupported-claim rate from 100% to 0% — at *zero* extra calls.
2. **On the happy path, multi-agent costs 3.1× more tokens and buys nothing.**
   This matches Anthropic's and Cognition's published engineering post-mortems: orchestration
   does not produce correctness; it only pays off on the **failure path**.
3. **On the failure path it earns its keep — but review is not omnipotent.** Same injected fault,
   two pipelines:

| Injected fault | `evidence-once` leaked | `multi` leaked |
|---|---|---|
| Fabricated citation (transient) | **6/6** | **0/6** — caught and rewritten |
| Constraint ignored (persistent) | 6/6 | **6/6** — detected, but unfixable |
| Invalid JSON (transient) | 0/6 | 0/6 — absorbed by local retry, no orchestration needed |

> ⚠️ **Read the scope before the numbers.** The run above used `mock`, a deterministic
> rule script — **not a model**. Call counts, tokens, cost, contract compliance, retries and
> fault-interception rates are valid (they depend only on pipeline structure).
> **Unsupported-claim rate and constraint coverage reflect the script's settings, not model
> behaviour.** The 20 cases are hand-written, not a benchmark. Full scope and falsification
> conditions: [`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## What it is

Enterprise customers open with something like:

> "We want an internal knowledge-base Q&A for our support team, but the data cannot leave our
> network. Around 50 QPS, budget about 2,000 CNY/month."

Turning that into a defensible technical proposal has two failure modes:

1. **The model invents things.** "This model supports a 128k context" — if that number is wrong,
   the sales engineer just vouched for it with the company's reputation.
2. **Customer constraints get dropped.** They said "data cannot leave the network"; the proposal
   never mentions private deployment. You find out during the POC.

This project turns both into **checkable stages**, then asks how many agents that actually takes.

---

## Architecture

```
customer sentence
      │
      ▼
┌──────────────────┐
│ Intake           │  single-threaded: structured requirement;
│                  │  if under-specified → ask first, do not write a plan
└────────┬─────────┘
         ▼
┌──────────────────┴──────────────────┐
│ Evidence ×3   parallel · READ-ONLY  │  model selection / limits & compliance / pricing
└────────┬────────────────────────────┘
         ▼
┌──────────────────┐
│ Writer           │  single-threaded — the ONLY place allowed to produce output;
│                  │  every claim must carry an evidence reference
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Critic           │  rule layer (deterministic) + model layer; rejects and rewrites (max 2)
└────────┬─────────┘
         ▼
draft + evidence table + follow-up questions + cost/latency report
```

Design rationale, with sources, in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
What was rejected and at what cost, in [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md).

---

## Quick start

```bash
git clone <this-repo>
cd ark-solution-lab

# zero dependencies in the core pipeline
python -m arksol demo                        # run one case end to end
python -m arksol demo --pipeline no-evidence # compare against the baseline
python -m arksol eval --provider mock --out results/

python -m unittest discover -s tests -t .
```

With a real model (any OpenAI-compatible endpoint). **Keys go in `.env` or environment variables —
never in code:**

```bash
cp .env.example .env        # fill in your key; .env is git-ignored
python -m arksol eval --provider openai-compat --out results/ark/
```

Resolution order: explicit arguments → real environment variables → `.env`
(zero-dependency `.env` reader in `arksol/env.py`; no python-dotenv).

## Layout

```
arksol/            core package (no third-party deps)
  agents/          intake / evidence / writer / critic
  pipeline/        three comparison pipelines, retrieval, metrics
  evaluation/      case loading, experiment runner, report rendering
  llm/             providers: mock (fault-injectable) / openai-compat
  kb/              26 curated knowledge entries, each with source URL + read date + confidence
  contracts.py     hand-off contracts + minimal JSON-schema validator
  verify.py        independent auditor (one yardstick for all three pipelines)
  budget.py        triple budget ceiling + repeat-action breaker + metering
  state.py         externalised state + hand-off notes
evaluation/        20 cases
results/           report.md / results.csv / raw.json / traces/
tests/             257 stdlib unittest cases
```

## Docs

| Document | What's in it |
|---|---|
| [`results/report.md`](results/report.md) | experiment report + scope |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | design rationale with sources, 6 known limitations |
| [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) | what was rejected and why, with costs |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | metric definitions, anti-gaming notes, falsification conditions |
| [`docs/大白话讲解.md`](docs/大白话讲解.md) | plain-language walkthrough (Chinese) |
| [`docs/面试怎么讲.md`](docs/面试怎么讲.md) | interview script + 16 anticipated questions (Chinese) |

## References

- Anthropic · [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- Cognition · [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents) ｜ [one year later](https://cognition.com/blog/multi-agents-working)
- MAST · [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)

## Limitations

- The knowledge base is a **demo**, not a substitute for official documentation. Every entry
  carries a source URL and read date, but it may still be wrong or stale — verify before real use.
- All output is a **draft**. Nothing is sent to a customer automatically.
- MIT licensed.
