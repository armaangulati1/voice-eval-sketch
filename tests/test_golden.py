"""Golden-set integrity: labels are well-formed and cover the transcripts."""

from __future__ import annotations

from voice_eval.golden import FLAG_VOCAB, GOLDEN, RISK_LEVELS, golden_by_id
from voice_eval.transcripts import TRANSCRIPTS


def test_one_label_per_transcript() -> None:
    transcript_ids = {t.id for t in TRANSCRIPTS}
    label_ids = {g.transcript_id for g in GOLDEN}
    assert transcript_ids == label_ids
    assert len(GOLDEN) == len(TRANSCRIPTS)


def test_set_has_between_five_and_eight_transcripts() -> None:
    assert 5 <= len(TRANSCRIPTS) <= 8


def test_labels_use_closed_vocabulary() -> None:
    for g in GOLDEN:
        assert g.expected.risk_level in RISK_LEVELS
        assert g.expected.flags <= FLAG_VOCAB


def test_every_label_has_a_rationale() -> None:
    for g in GOLDEN:
        assert g.rationale.strip(), f"{g.transcript_id} missing rationale"


def test_set_contains_a_true_negative() -> None:
    # At least one non-escalating case, to penalize over-escalation.
    assert any(not g.expected.escalate for g in GOLDEN)


def test_set_is_mostly_adversarial_escalations() -> None:
    escalate = sum(1 for g in GOLDEN if g.expected.escalate)
    assert escalate >= len(GOLDEN) - 2  # the set is deliberately hard


def test_index_round_trips() -> None:
    idx = golden_by_id()
    for g in GOLDEN:
        assert idx[g.transcript_id] is g


def test_edge_cases_are_distinct() -> None:
    edge_cases = [t.edge_case for t in TRANSCRIPTS]
    assert len(edge_cases) == len(set(edge_cases))
