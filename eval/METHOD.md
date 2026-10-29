# Real-life memory environment: benchmark method

This document describes how to measure a memory layer on **your own agent sessions**.
Use it to compare your memory service against other systems, or against itself after a
change. The kit in this folder implements the pipeline. The probes and the memory are
yours and stay on your machine.

## Why not a QA benchmark

QA-style memory benchmarks build a synthetic conversation, then ask questions whose
answers are planted in it. That measures retrieval, and assumes every question needs
memory. In real use, most turns need no memory at all, and an injected memory that is
irrelevant or stale costs attention. The question here is different: **at the next real
user turn, did the system inject what helps, and stay quiet when nothing would?**

## Probes: real turns, judged in hindsight

- A **probe** is a real user turn from your past sessions, together with the previous
  turn and the session's client and project. Sample turns that probably need memory,
  routine turns, and turns that correct or update an earlier fact. Treat these strata as
  sampling hints, not as ground truth.
- **Hindsight**: up to six following turns of the same session. Only the judge sees them.
  They show what the user actually needed (e.g. "no, I meant the other dataset"). Systems
  never see them.
- Exclude scheduled jobs, sub-agent sessions, sessions about the benchmark itself, and
  anything marked local-only. Remove obvious secrets before any text reaches a model.

## Time and leakage

- Build **weekly snapshots** of the memory folder from version history, each taken before
  every probe in its week. A probe sees only its snapshot. Nothing from its own future
  leaks in. The cost is that a fact written mid-week shows up a week late.
- Split by time: early weeks are **dev**, later weeks are **test**. Each session belongs to
  one split only. Tune anything (thresholds, prompts, formats) on dev. Report on test.

## Systems: each used the way it is meant to be used

All systems read the **same snapshots of the same curated memory**. This track compares
recall and injection, not how well a system writes memory from raw chat. Each system must
use its native path:

| System | Write path | Read path |
|---|---|---|
| your memory service | its indexer | its own context endpoint and profile |
| OpenViking | native resource ingestion (layered) | native context assembly (`search_context`), scoped by URI path |
| mem0 | native `add(..., infer=True)`: its own LLM extracts facts, ingested week by week | native search, filtered to facts visible at that week |
| BM25 | heading-aware chunks of the snapshot | keyword top-10 |
| dense | the same chunks, one embedding model for everyone | cosine top-10 |

The author's first run used OpenViking's top-k file search and mem0 without extraction.
OpenViking's directory summaries and metadata, and mem0's raw evidence blocks, then filled
the byte budget. Their numbers looked three to four times worse than they are. Native usage
is not a courtesy; without it the comparison is wrong.

## Two regimes

1. **Same byte budget.** Every system gets the same per-client budget (the author used
   1,200 bytes for an assistant and 600 for a coding agent) and must fill it with whole
   items in rank order. Each item's visible text is capped at the same length. This shows
   what a system does with the space when it is forced to speak.
2. **Own threshold.** Each system gets one score threshold, tuned on dev only to maximize
   the composite score (below). Your own service may instead report its shipped policy.
   This shows when a system chooses to stay silent.

## Judging

- **Pool.** Pool the union of every system's top items per probe. Give items with
  different visible text different ids, so a label for a full card is never reused for a
  truncated view. Items that appear later, from a re-run, are judged with the same rubric
  and judge before scoring.
- **Blind judge.** An LLM judge sees the probe, the previous turn, the hindsight and the
  items, under shuffled neutral aliases (C001, C002…). It sees no system names, scores or
  ranks. It returns strict JSON:
  - whether this turn needed memory at all;
  - a label per item (`necessary`, `helpful`, `irrelevant`, `harmful`), each with a short
    rationale anchored by a verbatim quote from the item or the session.

  Invalid outputs stay unknown and are never scored as zero.
- **Audit.** A second judge from a different vendor re-judges a fixed 20% sample. Report
  agreement and Cohen's κ for both item labels and "needed memory". Expect the second to be
  much lower. Whether a turn needs memory is genuinely ambiguous.

## Metrics

All rates are on the test split. Errors stay in the denominators.

| Metric | Definition |
|---|---|
| Hit rate | share of turns with at least one injected item judged necessary or helpful |
| Useful precision | useful injected items / injected items |
| Harmful rate | harmful injected items / injected items (stale, wrong project, misleading) |
| Need recall | on turns judged to need memory: share where a *necessary* item was injected |
| Quiet rate | on turns judged not to need memory: share with nothing injected |
| Composite | per turn: recall a necessary item when needed, stay quiet when not, minus a penalty for harm |
| Volume | turns with any injection, items per turn, bytes per turn |

Report 95% confidence intervals from a **session-clustered bootstrap**: resample whole
sessions, 2,000 times. Compare systems with paired differences on the same probes.

## What the author's run found

500 probes from about five weeks of one person's assistant and coding-agent sessions. The
test split is 467 judged turns across 83 sessions. Under the same budget:

| | Ours | OpenViking | mem0 | Dense | BM25 |
|---|---|---|---|---|---|
| Hit rate | **44%** | 10% | 32% | 29% | 31% |
| Useful precision | 21% | 16% | 19% | 20% | 19% |
| Harmful rate | 2.0% | 1.7% | 4.6% | 4.3% | 4.8% |
| Items per turn | 3.2 | 0.6 | 2.7 | 2.0 | 2.0 |

- Precision is similar everywhere. The differences come from spending the budget better
  (no repeated text, so more items fit) and from scope and supersede filters (about half
  the harm).
- With their own thresholds, every system stays silent on most turns.
- Only about one turn in seven needed memory. On about two thirds of those, the needed
  fact was not in memory at all: mostly the state of a task in progress ("is it done?").
  The next gain is in what gets remembered, not in how it is searched.

**Limits.**
- One user, and the author's system was developed on that user's memory. Its injection
  format was simplified after an error analysis of an earlier run of this benchmark.
- LLM judges, with low agreement on "needed memory" (κ ≈ 0.17).
- The pool can miss memories that no system retrieved.
- All systems read the same curated memory, so writing memory from raw chat was not
  compared.

The data is private and is not released. Only the method and the code are.
