from __future__ import annotations

import json

import pytest

from conftest import item, judgment, probe, response_for
from titrate_eval.fake_judge import fake_judge_answer
from titrate_eval.judging import (PROTOCOL_VERSION, RUBRIC, JudgeValidationError, audit_agreement,
                                  build_judge_request, cohen_kappa, extract_response_text,
                                  select_audit_probe_ids, validate_judgment)


def test_rubric_defines_every_label_and_the_strict_output_schema() -> None:
    for label in ("necessary:", "helpful:", "irrelevant:", "harmful:"):
        assert label in RUBRIC
    assert "BEGIN_MEMORY_EVIDENCE" in RUBRIC and "Return one JSON object only" in RUBRIC


def test_request_blinds_metadata_and_is_chat_completions_at_temperature_zero() -> None:
    p = dict(probe(), system="SECRET_SYSTEM_NAME", cwd="SECRET_CWD", project_hint="same-project-hint",
             canonical_labels="SECRET_LABEL")
    value = dict(item(), system="SECRET_ORIGIN", score=0.123456789, uri="SECRET_URI")
    canonical = build_judge_request(p, [value], model="primary-model")
    audit = build_judge_request(p, [value], stage="audit", model="audit-model")
    for request in [canonical, audit]:
        serialized = json.dumps(request["payload"])
        assert "SECRET_" not in serialized
        assert "0.123456789" not in serialized
        assert '"item_id": "a"' not in serialized
        assert "That setting is old" in serialized
        assert "same-project-hint" in serialized
        assert request["payload"]["temperature"] == 0
        assert request["endpoint"].endswith("chat/completions")
    assert canonical["payload"]["model"] == "primary-model"
    assert audit["payload"]["model"] == "audit-model"
    assert canonical["protocol"] == PROTOCOL_VERSION


def test_aliases_are_shuffled_reproducibly() -> None:
    items = [item(f"id{i}", f"text {i}") for i in range(12)]
    first = build_judge_request(probe(), items)
    assert first["alias_map"] == build_judge_request(probe(), items)["alias_map"]
    assert list(first["alias_map"]) == [f"C{i:03d}" for i in range(1, 13)]
    assert list(first["alias_map"].values()) != [f"id{i}" for i in range(12)]


def test_valid_judge_restores_ids_and_exact_anchor_sources() -> None:
    request = build_judge_request(probe(), [item("private-original-id")])
    parsed = validate_judgment(json.dumps(response_for(request)), request)
    assert parsed["items"][0]["item_id"] == "private-original-id"
    assert parsed["items"][0]["anchors"][0]["source"] == "private-original-id"
    assert parsed["memory_needed"] is True


@pytest.mark.parametrize("malformation", ["missing", "unknown", "duplicate", "stitched", "extra", "fence", "enum",
                                          "candidate_anchor", "nonstring_label", "nonstring_needed"])
def test_judge_invalid_evidence_never_becomes_a_score(malformation: str) -> None:
    request = build_judge_request(probe(), [item()])
    answer = response_for(request)
    if malformation == "missing":
        answer["items"] = []
    elif malformation == "unknown":
        answer["items"][0]["item_id"] = "C999"
    elif malformation == "duplicate":
        answer["items"].append(answer["items"][0])
    elif malformation == "stitched":
        answer["items"][0]["anchors"][0]["quote"] = "Use...setting."
    elif malformation == "extra":
        answer["score"] = 2
    elif malformation == "enum":
        answer["items"][0]["label"] = "necessary because it is useful"
    elif malformation == "candidate_anchor":
        answer["items"][0]["anchors"] = [{"source": "query", "span_id": next(iter(request["span_catalogue"]["query"]))}]
    elif malformation == "nonstring_label":
        answer["items"][0]["label"] = ["necessary"]
    elif malformation == "nonstring_needed":
        answer["memory_needed"] = {"value": "yes"}
    text = json.dumps(answer)
    if malformation == "fence":
        text = "```json\n" + text + "\n```"
    with pytest.raises(JudgeValidationError):
        validate_judgment(text, request)


def test_chat_and_responses_bodies_are_parsed_and_incomplete_rejected() -> None:
    chat = {"choices": [{"finish_reason": "stop", "message": {"content": '{"a":1}'}}]}
    assert extract_response_text(chat) == '{"a":1}'
    with pytest.raises(JudgeValidationError, match="finish_reason"):
        extract_response_text({"choices": [{"finish_reason": "length", "message": {"content": "{"}}]})
    with pytest.raises(JudgeValidationError, match="explicit_refusal"):
        extract_response_text({"choices": [{"finish_reason": "stop", "message": {"refusal": "no", "content": ""}}]})
    response = {"status": "completed", "output": [{"type": "reasoning"},
                {"type": "message", "content": [{"type": "output_text", "text": '{"a":1}'}]}]}
    assert extract_response_text(response, endpoint="/v1/responses") == '{"a":1}'
    with pytest.raises(JudgeValidationError, match="response_status"):
        extract_response_text(dict(response, status="incomplete"), endpoint="/v1/responses")


def test_audit_sampling_is_fixed_and_kappa_reports_confusion_matrix() -> None:
    ids = [f"p{i}" for i in range(21)]
    assert len(select_audit_probe_ids(ids)) == 5  # ceil(20% of 21)
    assert select_audit_probe_ids(ids) == select_audit_probe_ids(list(reversed(ids)))
    canonical = {"p1": judgment(), "p2": judgment("p2", False, {"a": "harmful"})}
    audit = {"p1": judgment(labels={"a": "helpful"}), "p2": judgment("p2", False, {"a": "harmful"})}
    result = audit_agreement(canonical, audit, ["p1", "p2", "missing"])
    assert result["memory_needed"]["kappa"] == 1
    assert result["item_labels"]["kappa"] == 1  # necessary/helpful collapse to "useful"
    assert result["item_labels"]["confusion_matrix"] == [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
    assert result["missing_probe_ids"] == ["missing"]
    assert canonical["p1"]["items"][0]["label"] == "necessary"
    assert cohen_kappa(["yes"], ["yes"], ["yes", "no"])["kappa"] is None  # degenerate, never 1
    partial = cohen_kappa(["yes", "no", "yes", "no"], ["yes", "no", "no", "no"], ["no", "yes"])
    assert partial["kappa"] == pytest.approx(0.5)


def test_segments_preserve_every_source_character() -> None:
    text = "  leading whitespace\n" + ("中文 sentence。\n" * 75) + ("x" * 900) + "\ntrailing  "
    request = build_judge_request(dict(probe(), user_turn=text, previous_turn=""), [item(text=text)])
    model, original = request["model_packet"], request["packet"]
    views = [model[key] for key in ("query", "previous", "project_context")] + model["hindsight"] + model["candidates"]
    sources = {key: original[key] for key in ("query", "previous", "project_context")}
    sources.update({f"hindsight_{i}": row["text"] for i, row in enumerate(original["hindsight"])})
    sources.update({row["item_id"]: row["text"] for row in original["candidates"]})
    seen = set()
    for view in views:
        assert "text" not in view
        assert "".join(segment["text"] for segment in view["segments"]) == sources[view["source"]]
        for segment in view["segments"]:
            assert segment["span_id"] not in seen
            seen.add(segment["span_id"])
            pos = request["span_catalogue"][view["source"]][segment["span_id"]]
            assert sources[view["source"]][pos["start"]:pos["end"]] == segment["text"]
            assert 0 < len(segment["text"]) <= 256
    parsed = validate_judgment(json.dumps(response_for(request)), request)
    assert parsed["anchors"][0]["quote"] == model["query"]["segments"][0]["text"]


@pytest.mark.parametrize("bad_anchor", ["unknown_id", "wrong_source", "legacy_quote", "extra_key", "catalogue_mismatch"])
def test_unknown_or_unfaithful_span_anchors_fail(bad_anchor: str) -> None:
    request = build_judge_request(probe(), [item()])
    answer = response_for(request)
    anchor = answer["anchors"][0]
    if bad_anchor == "unknown_id":
        anchor["span_id"] = "S999999"
    elif bad_anchor == "wrong_source":
        anchor["span_id"] = next(iter(request["span_catalogue"]["previous"]))
    elif bad_anchor == "legacy_quote":
        anchor.pop("span_id")
        anchor["quote"] = request["packet"]["query"]
    elif bad_anchor == "extra_key":
        anchor["text"] = "extra"
    else:
        request["span_catalogue"]["query"][anchor["span_id"]]["start"] += 1
    with pytest.raises(JudgeValidationError):
        validate_judgment(json.dumps(answer), request)


def test_packet_byte_bound_fails_without_truncating() -> None:
    with pytest.raises(ValueError, match="packet exceeds byte bound"):
        build_judge_request(dict(probe(), user_turn="x" * 500_001), [item()])


def test_fake_judge_speaks_the_real_protocol() -> None:
    request = build_judge_request(probe(), [item("a"), item("b", "unrelated words entirely")])
    parsed = validate_judgment(json.dumps(fake_judge_answer(request["payload"])), request)
    assert {row["item_id"] for row in parsed["items"]} == {"a", "b"}
