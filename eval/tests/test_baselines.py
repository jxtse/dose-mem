from __future__ import annotations

import math
import subprocess
import sys

import pytest

from dosemem_eval.data import Corpus, Document
from dosemem_eval.systems.baselines import BM25Index, BM25System, DenseIndex, DenseSystem, chunks, tokens
from dosemem_eval.systems.embedders import HashEmbedder, OpenAICompatibleEmbedder


def item(identity: str, text: str, project: str = "global") -> dict:
    return {"item_id": identity, "text": text, "project": project}


def doc(identity: str, text: str, project: str = "global", path: str | None = None) -> Document:
    return Document.from_dict({"id": identity, "path": path or f"{identity}.md", "project": project, "text": text})


def test_tokenization_covers_cjk_without_neural_dependencies() -> None:
    assert tokens("HELLO_2 中文") == ["hello_2", "中", "文", "中文"]


def test_bm25_matches_closed_form_for_one_document() -> None:
    result = BM25Index([item("x", "apple apple banana")]).search("apple apple")
    expected = math.log(1 + 0.5 / 1.5) * 2 * 2.5 / (2 + 1.5)
    assert result[0]["score"] == pytest.approx(expected)


def test_bm25_scope_includes_global_but_excludes_other_project() -> None:
    engine = BM25Index([item("g", "apple"), item("p", "apple", "chosen"), item("q", "apple apple apple", "other")])
    assert {x["item_id"] for x in engine.search("apple", scope=["chosen", "global"])} == {"g", "p"}
    assert engine.search("apple", limit=1)[0]["item_id"] == "q"


def test_bm25_no_match_keeps_zero_scores_for_fixed_budget_regime() -> None:
    result = BM25Index([item("z", "banana"), item("a", "apple")]).search("unknown")
    assert [(x["item_id"], x["score"]) for x in result] == [("a", 0.0), ("z", 0.0)]
    assert BM25Index([]).search("anything") == []


def test_chunks_preserve_heading_provenance_unicode_and_identity() -> None:
    documents = [doc("s", "# Heading\n" + "中文段落" * 100 + "\n## Next\nshort", "p", "wiki/sample.md")]
    result = chunks(documents, max_bytes=256)
    assert len(result) > 2
    assert all(len(x["text"].encode()) <= 256 for x in result)
    assert all(x["source"] == "wiki/sample.md" and x["project"] == "p" for x in result)
    assert result[-1]["text"] == "## Next\nshort"
    assert chunks(documents, max_bytes=256) == result


def test_chunk_long_multibyte_heading_terminates_and_respects_budget() -> None:
    code = ("from dosemem_eval.systems.baselines import chunks; from dosemem_eval.data import Document; "
            "d=Document.from_dict({'id':'x','path':'x','text':'# '+chr(128512)*160+'\\n'+'a'*500}); "
            "rows=chunks([d],max_bytes=128); assert rows; assert all(len(r['text'].encode())<=128 for r in rows)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_dense_is_cosine_not_inner_product_and_filters_project() -> None:
    engine = DenseIndex([item("same", "x"), item("huge", "y"), item("wrong", "z", "other")], [[1, 1], [100, 0], [2, 2]])
    result = engine.search([1, 1], scope=["chosen", "global"])
    assert [x["item_id"] for x in result] == ["same", "huge"]
    assert result[0]["score"] == pytest.approx(1)
    assert result[1]["score"] == pytest.approx(1 / math.sqrt(2))


def test_dense_zero_norms_and_empty_index_are_finite() -> None:
    assert DenseIndex([item("z", "x")], [[0, 0]]).search([0, 0])[0]["score"] == 0
    assert DenseIndex([], []).search([1, 0]) == []


@pytest.mark.parametrize("vectors", [[[1, 2], [1]], [[float("nan"), 0], [1, 0]], [[float("inf"), 0], [1, 0]]])
def test_dense_rejects_mismatched_or_nonfinite_stored_vectors(vectors) -> None:
    with pytest.raises(ValueError):
        DenseIndex([item("a", "x"), item("b", "y")], vectors)


@pytest.mark.parametrize("query", [[1], [float("nan"), 0], [float("inf"), 0]])
def test_dense_rejects_invalid_query_vectors(query) -> None:
    with pytest.raises(ValueError):
        DenseIndex([item("a", "x")], [[1, 0]]).search(query)


def test_systems_only_see_their_probe_epoch() -> None:
    corpus = Corpus({"e1": [doc("a", "the old parser uses regex", "proj")],
                     "e2": [doc("a", "the new parser uses a grammar", "proj")]}, ["e1", "e2"])
    probe = {"user_turn": "which parser", "project_hint": "proj"}
    for system in (BM25System(), DenseSystem(HashEmbedder())):
        system.prepare(corpus)
        assert all("old" in row["text"] for row in system.retrieve(probe, "e1"))
        assert all("new" in row["text"] for row in system.retrieve(probe, "e2"))


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    first = HashEmbedder(16).embed(["alpha beta", "alpha beta", ""])
    assert first[0] == first[1]
    assert sum(x * x for x in first[0]) == pytest.approx(1)


def test_openai_compatible_embedder_uses_wire_format(monkeypatch) -> None:
    from dosemem_eval import providers

    sent = []

    def fake_post(target, path, payload):
        sent.append((path, payload))
        return 200, {"data": [{"index": i, "embedding": [float(i), 1.0]} for i in reversed(range(len(payload["input"])))]}

    monkeypatch.setattr("dosemem_eval.systems.embedders.post_json", fake_post)
    embedder = OpenAICompatibleEmbedder(providers.Endpoint(base_url="https://example.invalid/v1", model="m"),
                                        batch_size=2)
    vectors = embedder.embed(["a", "b", "c"])
    assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert [p["input"] for _, p in sent] == [["a", "b"], ["c"]]
    assert sent[0][0] == "/embeddings" and sent[0][1]["model"] == "m"
