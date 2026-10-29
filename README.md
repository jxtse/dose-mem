# Titrate

**One Markdown memory, many agents, the right dose for each.**

Titrate is an open design for a memory layer shared by all your AI agents: a personal
assistant that chats across days, coding agents working inside one repository, and
whatever agent you add next. It is published as **build specs**. You hand them to your
own coding agent, and it builds the system on your machine, with your folders, your model
provider and your agents.

![Architecture](docs/figures/fig1-architecture.png)

## What it gives you

- **One source of truth you can read.** Memories are plain Markdown cards and project
  notes. The search index is derived and can be thrown away. You can open, diff and edit
  everything.
- **Different doses for different agents.** The assistant gets broad recall on most turns.
  A coding agent gets at most a couple of lines, only from the current project, and only
  when it starts a session, switches project or refers back ("as usual"). Each agent can
  still search memory on purpose.
- **Silence is an answer.** Every injection has to clear a score floor and fit a byte
  budget. When nothing qualifies, nothing is injected.
- **Memory that maintains itself.** Finished sessions are distilled into candidate facts,
  each quoting the conversation verbatim. An automatic judge promotes, merges, supersedes
  or rejects them. Corrections create a new card that supersedes the old one, which is kept
  on disk. Caps, canary probes and one-command rollback replace a human review queue.
- **A way to measure it on your own life.** The `eval/` kit replays your real past turns
  against weekly snapshots of your memory, and has a blind LLM judge decide, in
  hindsight, what each system injected that helped or hurt.

![Same prompt, different dose](docs/figures/fig2-dose.png)

## How it works

1. **You and your coding agent build the service from [`specs/01-memory-service.md`](specs/01-memory-service.md).**
   The agent first interviews you: where your notes live, which agents should read them,
   which embedding provider to use, and where state may live. Then it builds the indexer,
   the hybrid retriever, the recall profiles and one REST/MCP endpoint, plus a thin adapter
   for each of your agents.
2. **Then the write path from [`specs/02-write-path.md`](specs/02-write-path.md).**
   Session capture, distillation, an automatic judge and the guardrails.
3. **Optionally, measure it with [`eval/`](eval/).**
   Mine probes from your own sessions and compare your service with BM25, dense retrieval,
   OpenViking and mem0, each used through its native path. See
   [`eval/METHOD.md`](eval/METHOD.md).

`examples/` holds a tiny, fictional memory folder, a profile config, and what an injected
block looks like.

## Quick start

**Hand this to your coding agent** (Claude Code, Codex, or similar):

> Clone https://github.com/jxtse/titrate. Read `README.md`, then `specs/01-memory-service.md`.
> Interview me about my environment as the spec asks, then build the memory service for my
> agents. When it passes the spec's acceptance criteria, do the same for
> `specs/02-write-path.md`.

To measure it afterwards:

> Read `eval/METHOD.md` and `eval/README.md`. Write a probe miner for my agents' session
> logs, build weekly snapshots of my memory folder, and run the evaluation kit comparing my
> memory service with the baselines.

If you would rather read first, start with the two specs. They are short and contain only
design decisions; the implementation details are left to your agent.

## What we measured

On about five weeks of one person's real assistant and coding-agent sessions (467 judged
test turns), with the same memory and the same byte budget for every system:

![Results](docs/figures/fig4-results.png)

The reference implementation got a useful memory into 44% of turns, against 29–32% for mem0,
dense and BM25, with about half their harmful injections. Precision was similar for all
systems. The details, the caveats (one user, LLM judges) and what did *not* work are in
[`eval/METHOD.md`](eval/METHOD.md) and the
[blog post](https://jxtse.github.io).

## Repository map

| Path | What it is |
|---|---|
| `specs/01-memory-service.md` | Build spec: index, retrieval, recall profiles, service |
| `specs/02-write-path.md` | Build spec: capture, distill, automatic review, guardrails |
| `eval/` | Benchmark method and runnable evaluation kit (Python, `uv`) |
| `examples/` | Fictional memory folder, profile config, sample injected block |
| `docs/figures/` | Figures used above |

## Status

These specs describe a system the author runs daily, serving one personal assistant and
two coding agents. The reference implementation itself is not published: it is welded to
one machine, and a spec adapts better to yours. Issues and write-ups of your own builds
are welcome.

[中文说明](README.zh-CN.md) · MIT License
