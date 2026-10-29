"""titrate_eval: blind, equal-budget evaluation of agent memory layers.

The kit replays real user turns (probes) against frozen time-ordered snapshots
of a memory corpus, packs every system's ranked output into the same byte
budget, labels the union of candidates with a blinded LLM judge, and scores
the systems with session-clustered bootstrap confidence intervals.
"""

__version__ = "0.1.0"
