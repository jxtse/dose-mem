---
id: pit-exit-zero-is-not-success
type: pitfall
scope: global
status: active
confidence: high
evidence: "session 2026-04-11: a run exited 0 but every tool call had silently failed"
created_by: auto
reviewed_by: auto-judge
auto_promoted: true
created_at: 2026-04-11
expires: null
---
Exit code 0 or passing unit tests do not prove an experiment worked; check real traces, tool calls and scores before reporting success.
