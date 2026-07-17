"""Excluded-judge discipline.

An LLM-as-judge is convenient but must never be trusted blind. Before a judge is
allowed to contribute to any score, it is validated against the human golden
labels. If its agreement with ground truth falls below a threshold, it is
EXCLUDED from scoring entirely -- you fall back to the deterministic scorer
rather than let an unvalidated judge move the numbers.

This mirrors the discipline that a judge which fails agreement with the labeled
set is dropped from scoring, not silently averaged in. The judges here are
deterministic stubs so the methodology is testable offline; a real LLM judge
would be dropped in behind the same `Judge` signature and pass through the exact
same gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from voice_eval.golden import GOLDEN, golden_by_id
from voice_eval.transcripts import TRANSCRIPTS, Transcript, transcript_text

# A judge predicts a single boolean: should this conversation be escalated?
Judge = Callable[[Transcript], bool]

DEFAULT_AGREEMENT_THRESHOLD = 0.9


class JudgeExcludedError(RuntimeError):
    """Raised when an unvalidated/failing judge is asked to contribute to scoring."""


@dataclass(frozen=True)
class JudgeValidation:
    name: str
    agreement: float
    threshold: float
    admitted: bool
    disagreements: tuple[str, ...]


def validate_judge(
    judge: Judge,
    name: str,
    threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> JudgeValidation:
    """Check a judge against the golden escalate labels before trusting it."""
    by_id = golden_by_id()
    disagreements: list[str] = []
    for transcript in TRANSCRIPTS:
        expected = by_id[transcript.id].expected.escalate
        if judge(transcript) != expected:
            disagreements.append(transcript.id)
    total = len(TRANSCRIPTS)
    agreement = (total - len(disagreements)) / total if total else 0.0
    return JudgeValidation(
        name=name,
        agreement=agreement,
        threshold=threshold,
        admitted=agreement >= threshold,
        disagreements=tuple(disagreements),
    )


def judge_escalation_rate(
    judge: Judge,
    name: str,
    threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
) -> float:
    """Use a judge to estimate the escalation rate -- but ONLY if it validates.

    An excluded judge raises rather than returning a number. This is the whole
    point: a failing judge contributes nothing.
    """
    validation = validate_judge(judge, name, threshold)
    if not validation.admitted:
        raise JudgeExcludedError(
            f"judge {name!r} excluded: agreement {validation.agreement:.3f} "
            f"< threshold {validation.threshold:.3f} "
            f"(disagreed on {list(validation.disagreements)})"
        )
    escalated = sum(1 for t in TRANSCRIPTS if judge(t))
    return escalated / len(TRANSCRIPTS)


# ---------------------------------------------------------------------------
# Deterministic judge stubs, purely so the discipline is testable offline.
# ---------------------------------------------------------------------------


def faithful_judge_stub(transcript: Transcript) -> bool:
    """A judge that agrees with the golden labels -> should be ADMITTED."""
    return golden_by_id()[transcript.id].expected.escalate


def literal_sentiment_judge_stub(transcript: Transcript) -> bool:
    """A naive judge fooled by literal reassurance -> should be EXCLUDED.

    It escalates only on the most explicit danger words and is silenced by any
    reassuring phrase, so it disagrees with the golden set on the adversarial
    cases and fails validation.
    """
    text = transcript_text(transcript)
    reassured = any(cue in text for cue in ("i'm fine", "no big deal", "all clear", "nothing"))
    danger = any(tok in text for tok in ("chest pain", "suicide", "emergency"))
    return danger and not reassured


def summarize_validation(validation: JudgeValidation) -> str:
    verdict = "ADMITTED" if validation.admitted else "EXCLUDED"
    return (
        f"[{verdict}] judge={validation.name} "
        f"agreement={validation.agreement:.3f} "
        f"threshold={validation.threshold:.3f} "
        f"disagreements={list(validation.disagreements)}"
    )


# Convenience: the golden escalate count, for sanity in the README/CLI.
GOLDEN_ESCALATE_COUNT = sum(1 for g in GOLDEN if g.expected.escalate)
