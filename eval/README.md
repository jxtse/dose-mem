# dosemem eval kit

Measure a memory layer for AI agents on **your own sessions**. The kit replays real
user turns against time-ordered snapshots of your memory. Every system gets the same
byte budget. A blind LLM judge labels the pooled candidates, and the systems are scored
with session-clustered bootstrap confidence intervals.

The benchmark design (why real turns, hindsight judging, weekly snapshots, native usage
of third-party systems, the two regimes, the metrics) is described in
[METHOD.md](METHOD.md). This README covers the code.

Your probes, snapshots and judgments stay on your machine. The only network calls are
the ones you configure: a judge endpoint, optionally an audit judge, an embedding
endpoint for the dense baseline, and the third-party systems you choose to run.

## Quick Start

**1. Hand this to your coding agent:**

> Clone this repository. Read `eval/METHOD.md` and `eval/README.md`. Write a small
> adapter that converts my agents' session logs into the normalized sessions format of
> `dosemem_eval/mining.py`, and export weekly snapshots of my memory folder into the
> snapshot format below. Then run the kit on my memory with `bm25` and `dense`, plus my
> own memory service as a new adapter. Show me `summary.json` and let me check the
> provider config before you make any paid API call.

**2. Or by hand.** Try the offline example first (no network, fake judge):

```bash
cd eval
uv sync
uv run dosemem-eval all \
  --probes examples/tiny/probes.jsonl --snapshots examples/tiny/snapshots \
  --run runs/demo --system bm25 --system dense-fake --fake --audit
```

This prints a table and writes `runs/demo/summary.json`. The example data is fictional
and the fake judge uses a keyword rule, so the numbers mean nothing. They only show that
the pipeline works.

Then run it on your data:

```bash
cp dosemem.example.toml dosemem.toml          # set base_url / model / api_key_env
export JUDGE_API_KEY=... EMBED_API_KEY=...

uv run dosemem-eval mine      --sessions my-sessions.jsonl --snapshots my-snapshots \
                              --test-from 2026-02-01 --out runs/me/probes.jsonl
P="--probes runs/me/probes.jsonl --run runs/me --config dosemem.toml"
uv run dosemem-eval retrieve  $P --snapshots my-snapshots --system bm25 --system dense
uv run dosemem-eval pool      $P
uv run dosemem-eval judge     $P --stage canonical
uv run dosemem-eval judge     $P --stage audit          # optional second judge, fixed 20%
uv run dosemem-eval calibrate $P                        # one threshold per system, dev only
uv run dosemem-eval score     $P
uv run dosemem-eval report    $P
```

`runs/` is git-ignored. Keep it that way: it holds your conversations.

## Workflow

| Stage | What it does | Output in `<run>/` |
|---|---|---|
| `mine` | normalized sessions -> probes (previous turn, up to 6 hindsight turns, epoch, session-level dev/test split) | probes JSONL |
| `retrieve` | prepares each system on every snapshot in time order, then asks each probe against **its own** epoch; per-probe failures are recorded as outcomes | `retrieval/<system>.jsonl` |
| `pool` | union of every system's top-10 *visible* items, plus up to 5 unscoped BM25 items, deduplicated by visible-text hash; frozen once written | `pools.json` |
| `judge` | blind judge over the frozen pool (`canonical`), a fixed audit sample (`audit`), or items that appeared later (`supplementary`) | `judgments/<stage>/*.json` (raw responses kept) |
| `calibrate` | one score threshold per system that maximizes mean dev composite; ties go to the higher threshold; written once | `thresholds/<system>.json` |
| `score` | packs each ranked list under the byte budget (fixed regime and calibrated regime) and scores every probe | `scored.jsonl` |
| `report` | test-split metrics with 95% CIs, paired differences, audit kappa | `summary.json`, `per-probe.jsonl` |

`all` runs every stage in order.

## Data formats

**Probes** (`probes.jsonl`, one object per line):

```json
{"probe_id": "p04", "session_id": "s03", "client_group": "working", "split": "test",
 "epoch": "2026-01-12", "project_hint": "widget-detector",
 "user_turn": "Recall the current detection threshold ...",
 "previous_turn": "The shiny batch is still noisy.",
 "hindsight": [{"role": "assistant", "text": "The threshold is now 0.42 ..."}]}
```

- `client_group` is `assistant` or `working` (a coding agent). It selects the byte budget.
- `project_hint` is nullable. Every system gets the same hint and scopes to that project
  plus `global`.
- `hindsight` goes only to the judge. Systems get `Probe.system_view()`, which has no
  hindsight, split or session id.
- A session must sit entirely in dev or entirely in test. The loader enforces this.

**Snapshots** (a directory with one `<epoch>.jsonl` per snapshot, plus an optional
`epochs.json` that lists the epochs in time order):

```json
{"id": "wd-notes", "path": "projects/widget-detector/notes.md",
 "project": "widget-detector", "text": "# widget-detector: model notes\n...", "sha256": "<sha256 of text>"}
```

**Normalized sessions** (input to `mine`): see the docstring of `dosemem_eval/mining.py`.
Converting your agents' log formats into it is deliberately left to a small adapter on
your side.

## Systems

| `--system` | Implementation | Extra |
|---|---|---|
| `bm25` | stdlib BM25 (k1=1.5, b=0.75) over heading-aware 512-byte chunks | none |
| `dense` | cosine over the same chunks; any OpenAI-compatible `/embeddings` endpoint (`[embedding]`) | none |
| `dense-fake` | same, with a deterministic hash embedder (**tests and demo only**) | none |
| `openviking` | native `add_resource` (layered `semantic_and_vectors`) per epoch; native `search_context` recall, scoped with a `uri` path filter | `uv sync --extra openviking` |
| `mem0` | native `Memory.add(messages, infer=True)` over snapshots in time order, with `epoch_idx`/`removed_idx` metadata; search filtered to `epoch_idx <= k < removed_idx` and project in `[hint, global]` | `uv sync --extra mem0` |

Third-party SDKs are optional and imported lazily. No vendor code is copied into this
kit. You run and configure those systems yourself (servers, LLM and embedding providers).

**To add your own memory service**, implement two methods and register it in
`dosemem_eval/systems/__init__.py`:

```python
class MySystem:
    name = "mine"
    def prepare(self, corpus): ...          # index every snapshot (corpus.order is time-ordered)
    def retrieve(self, probe, epoch):       # probe = Probe.system_view()
        return [{"item_id": "...", "text": "what the agent would see", "score": 0.7, "source": "..."}]  # top-10
```

Return the text your service would actually inject. Labels attach to visible text, so
wrapping or metadata you return counts against your byte budget.

## Rules the code enforces

- **Equal budget.** `budget_items` packs whole items in rank order and stops at the first
  item that does not fit. It never truncates an item and never promotes a lower rank.
  Items are joined with `"\n\n"`, which counts toward the budget. Defaults: assistant
  1200 bytes, working 600 bytes. Override with `--budget assistant=1500`.
- **Visible view.** Each item is capped at 480 UTF-8 bytes on a character boundary, then
  identified by the hash of that text.
- **Blind judge** (protocol `dosemem-blind-span-v3`). The judge gets the rubric plus
  conversation evidence, and candidates under shuffled aliases `C001...`. It never sees
  system names, scores or ranks. Each source is sent as `{span_id, text}` segments.
  Rationales must cite a span from the item itself, and `memory_needed` must cite a span
  from the conversation. The output must be strict JSON with labels
  `necessary|helpful|irrelevant|harmful`. Requests use Chat Completions at temperature 0.
  Infrastructure failures (connection, 429, 5xx) get one retry. A completed but invalid
  response is final and leaves the probe **unknown**. It is never resampled and never
  scored as zero.
- **Audit.** `judge --stage audit` re-judges `ceil(20%)` of all probes with a fixed seed,
  using the `[audit_judge]` endpoint. `report` prints Cohen's kappa and a confusion
  matrix for `memory_needed` and for item labels (collapsed to useful / not useful /
  harmful). Audit labels never feed into the scores.
- **Unknowns and errors.** A system error stays in every denominator and gets zero
  effectiveness. If a judgment is missing, the label-based primary estimate is null, and
  the report gives the full-denominator lower and upper bounds instead.

## Metrics (`summary.json`)

For each system × regime × client group (`assistant`, `working`, `all`), on the test split:
`hit_rate`, `useful_precision`, `harmful_rate`, `need_recall`, `quiet_rate`,
`update_correctness`, `composite` (0/1/2 per turn), `items_per_turn`, `bytes_per_turn`,
latency p50/p95, `error_rate`, `judge_missing_rate`, `execution_coverage`. Each metric has
a 95% percentile CI from a session-clustered bootstrap (2000 resamples, `--seed`), and
`paired_differences` lists A−B for every pair of systems on the same resampled sessions.

Composite per turn: 0 if the system errored or injected anything harmful. Otherwise, if
memory was needed: 2 for a necessary item, 1 for helpful-only, 0 otherwise. If it was not
needed: 2 for injecting nothing, 1 for helpful-only, 0 otherwise.

## Development

```bash
uv sync --locked
uv run pytest -q
```

Tests run offline. The third-party adapters are tested against mocked SDK objects, and
the end-to-end test drives the CLI on `examples/tiny` with the fake judge.

## License

MIT
