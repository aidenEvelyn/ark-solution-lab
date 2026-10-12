# ark-solution-lab

> Turn one sentence from a customer into a **citation-backed solution draft** — with a reproducible
> experiment that answers a harder question: **how many agents does this problem actually need?**

**Status**: core code and experiments complete — 257 unit tests, **zero third-party dependencies** in the core pipeline.

---

## Headline result

### Real model (`deepseek-v4.1-flash`, 8 cases)

| Pipeline | Success | Unsupported-claim rate | Constraint coverage | Avg. calls | Avg. tokens | Avg. latency |
|---|---|---|---|---|---|---|
| `no-evidence` — answer from memory | 100% | **100%** (50/50) | 90% | 2.0 | 3,280 | 13.1 s |
| `evidence-once` — retrieve once | 100% | **0%** (0/80) | 100% | 2.0 | 4,468 | 15.1 s |
| `multi` — parallel research + review | 88% | 4% (3/81) | 100% | **8.1** | **23,831** | **50.6 s** |

**Four findings** (full report: [`results/ark/report.md`](results/ark/report.md)):

1. **Retrieval, not orchestration, is what stops fabrication.** Adding evidence drops the
   unsupported-claim rate from 100% to 0% — at *zero* extra calls.
2. **On the happy path, multi-agent costs 5.3× the tokens and 3.3× the latency, and is
   strictly worse on quality** (0% → 4% unsupported). This matches Anthropic's and Cognition's
   engineering post-mortems: orchestration does not produce correctness.
3. **The independent reviewer rejected 6 of 7 runs.** The reasons were consistent: the issues
   came from *missing customer information* (no volume, no accuracy target, no compliance
   answer), not from poor writing. Rewriting cannot fix that — only asking can.
4. **Infrastructure failures are a real cost of real runs.** One case failed on a network flake
   (`Errno 2`). The report lists it instead of quietly dropping it.

Same injected faults, two pipelines:

| Injected fault | `evidence-once` leaked | `multi` leaked |
|---|---|---|
| Fabricated citation (transient) | **2/2** | **0/2** — caught and rewritten |
| Constraint ignored (persistent) | 2/2 | **2/2** — detected, but unfixable |
| Invalid JSON (transient) | 0/2 | 0/2 — absorbed by local retry |

> ⚠️ **Scope**: 8 hand-written cases, a single run, **not a benchmark**. Costs are estimates
> (0 when no price is configured) — **not a bill**. Constraint coverage uses keyword matching
> as a proxy. Full metric definitions and 5 falsification conditions:
> [`docs/EVALUATION.md`](docs/EVALUATION.md).

### Offline, reproducible (`mock` provider, 20 cases)

`mock` is not a model — it is a deterministic rule script for exercising pipeline mechanics at
zero cost:

| Pipeline | Avg. calls | Avg. tokens | Unsupported-claim rate | Coverage |
|---|---|---|---|---|
| `no-evidence` | 1.7 | 845 | **100%** (42/42) | 100% |
| `evidence-once` | 1.7 | 1,499 | **0%** | 100% |
| `multi` | **4.5** | **4,585** | **0%** | 100% |

Fault injection: fabricated citation **6/6 → 0/6** ｜ ignored constraint **6/6 → 6/6** ｜
invalid JSON **0/6 → 0/6**.

**How to read these**: call counts, tokens, cost, contract compliance, retries, breakers and
fault-interception rates are **valid** (they depend only on pipeline structure). The
unsupported-claim rate, constraint coverage and follow-up hit rate under `mock` reflect the
**script's settings, not model behaviour**. Report: [`results/report.md`](results/report.md).

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
