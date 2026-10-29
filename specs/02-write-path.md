# Spec 02 — The write path: memory that updates itself, silently

> **Don't read this as documentation.** Hand it to your coding agent together with Spec 01
> and say: *"Build the write path for my memory service. Interview me first."*
> Build Spec 01 first. This spec writes into the folder that Spec 01 reads.

## 0. Agent: read this first

You are adding an automatic write path. Finished agent sessions are turned into candidate
facts. Each candidate is checked against evidence and against existing memory, and the
ones that pass are written as new Markdown cards. No human queue, no review reminders, no
daily digest.

The user's attention is the scarcest resource. Safety comes from guardrails, not from
asking the user.

## 1. The problem

Memory that only grows when the user remembers to say "remember this" stays thin and goes
stale. Memory that grows by dumping chat history fills up with noise: one-off progress
reports, test counts, environment snapshots. A review queue fixes quality but turns the
user into an unpaid moderator, and in practice the queue rots. In the author's first
attempt, 218 inbox candidates sat waiting. When they were finally processed automatically,
5 were promoted and 213 rejected.

## 2. Discover the environment

- **Session sources.** Where does each agent keep its transcripts (file layout, database),
  and how can you tell when a session is finished? Read them **read-only**.
- **Whose words count.** Which sessions involve only the user? Group chats with other
  people, scheduled jobs, sub-agents and this system's own maintenance sessions must be
  skipped.
- **Chat model and budget.** Which model distills and judges? What is the hard daily call
  cap the user accepts?
- **Card conventions.** The existing card template, types, and where project versus global
  cards go.

## 3. Non-negotiable design principles

### Capture

1. **Capture only finished, real sessions.** A session counts as finished when it has
   been idle long enough (the author uses 30 minutes) and has at least a few user turns.
   Use cursors so a growing session yields only its new part. Skip anything whose audience
   or identity is unclear.
2. **Digest before distilling.** Build a bounded digest: the user's turns, the final
   answers, tool names, and coarse error classes. Leave out raw tool output, system
   prompts and anything that looks like a credential. Keep provenance (source, session,
   working directory, time, digest hash) in private state, not in the card.

### Distill

3. **Atomic, self-contained facts, each with a verbatim quote.** Every candidate states one
   fact, names its project (or global), its type, and a confidence. It quotes the session
   word for word. Before going further, verify the quote locally against the digest
   (normalize only whitespace and quote marks).
4. **Durability is the main filter.** A fact must still be useful next month. Reject
   progress snapshots, counts from one run, environment versions, task status and
   anything unverifiable. In the author's audit, the distiller called every sampled
   candidate "durable", and only 30% actually were. Do not trust the distiller's own
   durability flag.

### Review: an automatic judge, not a human

5. **Four outcomes.**
   - **Promote** a new fact as a new card.
   - **Merge** a duplicate by attaching evidence to the candidate record. Add a new
     superseding card only if the new version is strictly more complete.
   - **Supersede** when the new evidence is newer and stronger, above all the user's own
     explicit correction. Write a new card with `supersedes:`. Never edit the old file.
   - **Reject** with a stored reason.

   Conflicts resolve to supersede or reject, never to "ask the user".
6. **Find the neighbours before judging.** Do a cheap deterministic pre-filter first. Then,
   only for candidates that pass, run a semantic neighbour search against existing cards,
   and judge in small batches with a model under strict structured output.
7. **Weaker evidence never overrides a stronger card.** A paraphrase by the assistant does
   not beat the user's own words. An old session does not beat a newer one.
8. **Archived projects stay archived.** A candidate about an archived project may be
   promoted to global only if it is a lasting, project-independent lesson.

### Write and guard

9. **Additive only.** New cards, new superseding cards, appended evidence. Old files are
   never rewritten or deleted by the write path. Tag auto-created cards (`created_by`,
   `reviewed_by`, `auto_promoted: true`) so they can be audited and rolled back as a set.
10. **Expiry by type.** Experiment results and decisions expire, for example after 180 and
    365 days. Preferences, pitfalls and conventions do not. Expired cards leave
    auto-recall but stay readable.
11. **Guardrails replace the reviewer.**
    - A **daily promotion cap**. Overflow waits for the next day.
    - **Canary probes before and after each batch.** Use a fixed set of prompts that should
      receive no memory. If false injection rises beyond a set margin, quarantine the whole
      batch automatically by appending archived superseding cards.
    - **One-command rollback** of all auto cards.
    - A **weekly freshness check**. A card whose sources have disappeared is re-judged and
      archived only if the judge confirms it is outdated.
12. **One shared, hard LLM budget.** Distillation, repair, retries, judging and canaries
    all reserve from the same daily counter before each physical call. The cap lives in
    config and cannot be bypassed by a parameter.
13. **Crash-safe and idempotent.** Persist each decision, publish plan and budget
    reservation before acting. A crash after writing a card must not write it twice or pay
    the judge twice. Run one worker under one lock.
14. **Silent by default.** Show status, counts, recent runs and failures in the existing
    admin view and CLI. Send no notifications.

## 4. Provider rule

Same as Spec 01. The chat model is configured by the user. Before relying on a model for
strict JSON, verbatim quoting and batch judging, test it on a real batch of the user's
sessions.

## 5. Components to build

1. **Session collectors**, one per agent from section 2: read-only, cursor-based, with idle
   detection.
2. **Digest builder** with redaction and size limits.
3. **Distiller** with local quote verification and one repair attempt for invalid JSON.
4. **Candidate store**: a small state machine (pending → triaged → promoted / merged /
   superseded / rejected), with provenance and idempotency keys.
5. **Auto-reviewer**: pre-filter, neighbour search and a batched judge.
6. **Publisher**: card writer, daily cap, canary run, quarantine and rollback.
7. **Freshness checker**: weekly, judge-confirmed archiving.
8. **Worker loop** under the host's service manager, with a heartbeat.

## 6. Acceptance criteria

- **End to end.** State a harmless test preference in a real agent session, then correct
  it in a later session.
  - After each session finishes, recall returns the current value within about an hour.
  - The old card stays on disk with `superseded_by` and is no longer injected.
- **Backlog.** Feed in an existing pile of untriaged notes and report what was promoted,
  merged and rejected, with reasons. Expect most to be rejected.
- **Spot check.** Read every auto-created card from the first week and report the share you
  would keep. Tighten the rejection rules if it is low.
- **Canary.** Force a bad batch in a test copy and show that it is quarantined
  automatically.
- **Budget.** The daily call counter never exceeds the cap, across restarts and day
  boundaries.

## 7. Don't

- Don't make the user review, approve, or receive reminders.
- Don't let an LLM rewrite or delete existing cards.
- Don't capture group chats, other people's words, scheduled jobs, or this system's own
  sessions.
- Don't store raw transcripts or tool output in the memory folder.
- Don't treat "the distiller said it is durable" as evidence.

## 8. Report back

Tell the user what is captured from which agents, the daily cap, where candidates and
decisions live, how to roll back auto cards, and the first-week numbers (candidates,
promoted, rejected, quarantined, LLM calls).
