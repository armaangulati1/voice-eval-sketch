"""Excluded-judge discipline: an unreliable judge is dropped, not averaged in."""

from __future__ import annotations

import pytest

from voice_eval.judge import (
    JudgeExcludedError,
    faithful_judge_stub,
    judge_escalation_rate,
    literal_sentiment_judge_stub,
    validate_judge,
)


def test_faithful_judge_is_admitted() -> None:
    validation = validate_judge(faithful_judge_stub, "faithful")
    assert validation.admitted
    assert validation.agreement == 1.0
    assert validation.disagreements == ()


def test_literal_sentiment_judge_is_excluded() -> None:
    validation = validate_judge(literal_sentiment_judge_stub, "literal")
    assert not validation.admitted
    assert validation.agreement < validation.threshold
    assert validation.disagreements  # it disagreed with the golden set


def test_admitted_judge_may_contribute_a_number() -> None:
    rate = judge_escalation_rate(faithful_judge_stub, "faithful")
    assert 0.0 <= rate <= 1.0


def test_excluded_judge_refuses_to_contribute() -> None:
    # The discipline: a failing judge raises rather than moving the numbers.
    with pytest.raises(JudgeExcludedError):
        judge_escalation_rate(literal_sentiment_judge_stub, "literal")


def test_threshold_is_enforced() -> None:
    # Even a faithful judge is excluded if the bar is set above perfect.
    validation = validate_judge(faithful_judge_stub, "faithful", threshold=1.01)
    assert not validation.admitted
