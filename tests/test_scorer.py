"""Scorer behavior and the demonstrated naive-vs-careful gap."""

from __future__ import annotations

from voice_eval.extractor import careful_rule_extractor, naive_keyword_extractor
from voice_eval.golden import GOLDEN, Extraction
from voice_eval.scorer import format_report, score_extractor


def test_perfect_extractor_scores_all_cases() -> None:
    # An oracle that returns each golden answer must pass every case.
    lookup = {g.transcript_id: g.expected for g in GOLDEN}

    def oracle(transcript):  # type: ignore[no-untyped-def]
        return lookup[transcript.id]

    summary = score_extractor(oracle)
    assert summary.passed == summary.total
    assert summary.case_accuracy == 1.0


def test_careful_extractor_beats_naive_on_adversarial_set() -> None:
    naive = score_extractor(naive_keyword_extractor)
    careful = score_extractor(careful_rule_extractor)
    # The whole point of the set: naive accuracy collapses, careful recovers.
    assert careful.passed > naive.passed


def test_naive_extractor_is_fooled_by_adversarial_framing() -> None:
    naive = score_extractor(naive_keyword_extractor)
    # It should fail the clear majority of the adversarial cases.
    assert naive.passed <= 2


def test_careful_extractor_handles_the_planted_edge_cases() -> None:
    careful = score_extractor(careful_rule_extractor)
    # The hand-authored reference is tuned to this synthetic set by design.
    assert careful.passed == careful.total


def test_case_requires_all_three_dimensions() -> None:
    # A case with a wrong risk level fails even if escalate + flags are right.
    lookup = {g.transcript_id: g.expected for g in GOLDEN}

    def wrong_risk(transcript):  # type: ignore[no-untyped-def]
        exp = lookup[transcript.id]
        bumped = "low" if exp.risk_level != "low" else "high"
        return Extraction(escalate=exp.escalate, risk_level=bumped, flags=exp.flags)

    summary = score_extractor(wrong_risk)
    for r in summary.results:
        assert not r.passed
        assert r.escalate_ok
        assert not r.risk_ok


def test_summary_dimension_accuracies_are_fractions() -> None:
    summary = score_extractor(careful_rule_extractor)
    for value in (
        summary.case_accuracy,
        summary.escalate_accuracy,
        summary.risk_accuracy,
        summary.flags_accuracy,
    ):
        assert 0.0 <= value <= 1.0


def test_report_lists_every_case() -> None:
    summary = score_extractor(naive_keyword_extractor)
    report = format_report("naive", summary)
    for g in GOLDEN:
        assert g.transcript_id in report
    assert "FAIL" in report  # naive must visibly fail something


def test_failures_helper_matches_failed_results() -> None:
    summary = score_extractor(naive_keyword_extractor)
    assert set(summary.failures()) == {r for r in summary.results if not r.passed}
