# Spec 01 — The memory service: one Markdown truth, many agents

> **Don't read this as documentation.** Hand it to your coding agent and say:
> *"Build this for me, adapted to my environment. Interview me first."*
> Anything below that looks like a constant (a budget, a threshold, a timeout) is the
> author's value and is probably wrong for you. Treat it as a starting point to calibrate.

## 0. Agent: read this first

You are building a small, always-on **memory service** that several AI agents share. It
reads a folder of plain Markdown notes, the user's memory, and decides, on every prompt
an agent receives, **whether** to inject any of it and **how much**.

Do not start coding until you have finished section 2 with the user. Build in the order of
section 5, and check each component against section 6 before moving on.

## 1. The problem

People now run more than one agent: a personal assistant that chats across days, coding
agents inside repositories, and soon more. Each ships its own memory, or none. The result is
four half-memories that disagree, cannot be inspected, and inject too much or too little.

Two observations drive this design:

1. **The hard part is dose, not recall.** In the author's own sessions, only about one turn
   in seven actually needed a memory. A system that always injects something pays a
   distraction tax on the other six. A system that can stay silent, and that spends a fixed
   byte budget well when it does speak, beats one that simply finds more.
2. **Different agents need different doses.** A personal assistant benefits from broad recall
   across projects. A coding agent working on one task in one repo wants almost nothing,
   except the few preferences about *how* the user likes work done, and the current
   project's decisions.

## 2. Discover the environment (ask, never assume)

Ask the user, and record the answers in a config file the service reads:

- **Memory folder.** Where do your notes live? Is there an existing structure (cards,
  project folders, a wiki)? Which folders hold raw material that must *never* be indexed
  (raw logs, journals, inboxes, private scratch)?
- **Agents.** Which agents should read this memory? For each: how can it receive injected
  context (a prompt hook, a plugin, an extension, an MCP client), what identifies a session,
  and does it know its working directory?
- **Model provider.** Which embedding model, and which chat model for the write path
  (Spec 02)? Configure each as `base_url` / `api_key` / `model`, cloud or local. The service
  must not care which one.
- **Vector store.** Use one the user already runs or is happy to run. The index is derived
  and disposable, so this choice is easy to change later.
- **Host.** OS and service manager (launchd, systemd, a container, Windows service). Where
  may persistent state live? Is it a removable disk that might be unmounted?
- **Privacy.** Are there notes that must never leave the machine, not even to an embedding
  API? Agree on a front-matter flag (e.g. `external_processing: forbidden`) and honour it
  everywhere.
- **Devices.** Will agents on other machines read this memory? If yes, over what private
  network?

## 3. Non-negotiable design principles

### Truth and index

1. **Markdown is the single source of truth.** A human can open, diff, edit and version
   every memory. The search index is derived, rebuildable from scratch, and never written
   back into the notes. If the two disagree, the Markdown wins.
2. **Memory cards are atomic.** One card holds one durable fact: a preference, a pitfall, a
   decision, or a result. It has front matter: id, type, scope (global or a project), status,
   evidence, created/expires, and `supersedes` / `superseded_by`. Project documents (briefs,
   decision logs) are indexed alongside the cards.
3. **Every node has three layers, derived deterministically, without an LLM.**
   - L0 is a one-line claim. For a card, this is the card's claim sentence.
   - L1 is a short overview: the first section, a few hundred characters.
   - L2 is the full text, split into heading-aware chunks for retrieval.
4. **Every node has a stable URI and a revision** (e.g. `brain://projects/<slug>/cards/<id>`
   plus a content hash). Every injected item carries both, so the agent can ask for the
   full text (`read(uri, level=L2)`). A read of a stale revision is refused rather than
   silently answered with newer text.
5. **A single writer maintains the index.** It watches the folder and updates incrementally
   by content hash. An edit becomes searchable within seconds. Deleting, expiring or
   superseding a card removes it from recall immediately.

### Retrieval: decide whether to speak, then what to say

6. **Hybrid recall, queried independently.** Dense vectors and keyword search (BM25 with
   a tokenizer that suits the user's languages) each return their own top-k. Fuse them
   with reciprocal-rank fusion. If the embedding call misses its deadline, answer from
   keywords alone and mark the response degraded. Never make the agent wait.
7. **Scope first.** Resolve which project the prompt is about: from the agent's working
   directory (via a project registry of repo paths), from an alias in the prompt, or from
   neither. Search only that project plus global memory. When the reference is ambiguous,
   use global only.
8. **Filter before ranking matters.** Drop nodes that are expired, superseded, archived,
   out of scope, not yet indexed, or private to a different audience (e.g. a group chat).
   Re-check each surviving item against the current file and revision before injecting it.
9. **Abstain is a first-class outcome.** Each recall profile has a score floor. If nothing
   clears it, inject nothing. Calibrate the floor on a development split of real prompts
   only, never on the data you report.
10. **Budgets are bytes, and they count everything the model sees.** That includes the
    envelope, the URI and the revision. Pack whole items in rank order. Never truncate an
    item to make it fit, and never promote a lower-ranked item past one that did not fit.
11. **Never repeat yourself inside the budget.** If an item's L1 begins with its L0, inject
    only the part of L1 that adds something. In the author's benchmark, this one change let
    the same budget carry about 3 items instead of 2. It raised the share of turns that
    received a useful memory from 30% to 44%.
12. **Injected memory is data, not instructions.** Wrap it in a fixed envelope that says so,
    serialize it as data (e.g. JSON), and keep scores and diagnostics out of the model's
    context.
13. **Do not re-inject within a session.** Remember the URI and revision already injected per
    (client, session). Inject again only when the revision changes or the host compacts its
    context.

### Recall profiles: same memory, different dose

14. **Every client has a profile, and profiles are hot-reloaded per request.** A profile
    sets the budget, item limit, levels (L0 only, or L0+L1), score floor, scope mode, the
    types allowed from global scope, and the trigger.
    - **Full profile** (a personal assistant): recall on most turns, all projects plus global,
      L0+L1, around 1–2 KB.
    - **Working profile** (coding agents): recall only at session start, on a project switch,
      or when the prompt refers back ("as usual", "last time", "remember"). Strictly the
      current project, plus global cards whose type is preference, pitfall or convention.
      L0 only, at most two items, a few hundred bytes. The agent can always pull more
      on purpose through `search` / `read` tools.
15. **Archived projects stop auto-recall but stay readable.** When a project ends, its
    nodes leave automatic injection. Explicit search and read by project or URI still work.
    Lift its lasting lessons into global cards first.

### Service and access

16. **One service, one endpoint, thin clients.** Expose REST plus MCP from one process with
    one permission model. Endpoints: `search`, `context`, `read`, `ls`/`tree`, and the
    write endpoints of Spec 02. Each agent integration is a thin adapter: call `context`
    with the prompt, working directory, session id and client identity, and paste the
    result in.
17. **Every client has its own token and scope.** Tokens are stored hashed. Remote clients
    get read-only, project-limited tokens by default. Admin and metrics are reachable only
    from loopback with the owner identity.
18. **Fail closed.** If the storage volume is missing, the config is malformed or a
    permission check fails, inject nothing and say why in the trace. Never fall back to a
    stale cache or a different data directory.
19. **Observable, not chatty.** Keep a bounded trace per request (route, client, latency,
    counts, and filter reasons) without storing the query or memory text. A small admin view
    should be able to answer "why was this injected?" and "why was nothing injected?".

## 4. Provider rule

The user configures their own embedding and chat providers. The service only needs an
OpenAI-compatible (or adapter-wrapped) `base_url` / `api_key` / `model`. Before you trust a
provider's latency, rate limits or output format, test them on a real batch. These vary
between providers, and the author's numbers do not transfer.

## 5. Components to build, in order

1. **Config and registry.** The memory folder, the excluded paths, a project registry
   (slugs, aliases, repo paths, status), the providers, the storage location and the
   profiles.
2. **Node model.** Parse cards and project docs into layered nodes with URIs, revisions,
   scope and lifecycle fields.
3. **Indexer.** A single writer with incremental updates and tombstones, plus a periodic
   reconcile against the folder.
4. **Retriever.** Scope resolution, dense and keyword recall, fusion, filters and
   revision re-checks.
5. **Context assembler.** Profiles, abstain, byte packing, the de-duplicated L1, the
   envelope and session dedupe.
6. **Service.** REST plus MCP, identities and scopes, traces, and an admin view.
7. **Client adapters.** One per agent from section 2, each under a hundred lines where
   possible.
8. **Operations.** Run it under the host's service manager. Back it up. Deploy new versions
   as immutable releases with one-step rollback. Provide export and import of the index
   state for a new machine.

## 6. Acceptance criteria

- The index reconciles one-to-one with the eligible Markdown, and excluded folders contribute
  zero nodes.
- An edited card becomes searchable within seconds. Deleted, expired or superseded cards
  are recalled zero times.
- On a small labelled set of real prompts, including some that need no memory:
  - recall at the item limit is reported;
  - false injection on no-answer prompts is at most 10% for the full profile and at most 3%
    for the working profile.
- On about 40 ordinary coding prompts inside a repository, the working profile injects on at
  most 10% of them, and never from another project.
- Each agent's real request shows the expected memory block, or the expected absence of one.
- When the embedding provider is down, recall still answers from keywords, marked degraded,
  within the agent's hook deadline.
- Optionally, run the evaluation kit in `eval/` on your own sessions (see `eval/METHOD.md`).

## 7. Don't

- Don't let the index, a vector store or an LLM become a second source of truth.
- Don't use an LLM on the read path. Rewrites, summaries and rerankers are optional
  experiments, off by default, and kept only if your own evaluation shows a gain.
- Don't inject the same fact twice in one budget, or into the same session twice.
- Don't give coding agents the assistant's dose.
- Don't send content marked as local-only to any provider.

## 8. Report back

Tell the user what you built, where state lives, how to add a client, how to change a
profile, how to roll back, and the numbers you measured for section 6, failures included.
