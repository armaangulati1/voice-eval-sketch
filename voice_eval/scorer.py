"""Deterministic exact-match scorer over the golden set.

A case PASSES only if the extractor's output exactly matches the golden label on
all three dimensions (escalate, risk_level, flags). There is no partial credit
and no fuzzy matching: the labels are a small closed vocabulary, so exact match
is the honest bar. Dimension-level agreement is reported separately so a run can
show *where* a model fails, not just that it did.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from voice_eval.golden import GOLDEN, Extraction, GoldenLabel
from voice_eval.transcripts import TRANSCRIPTS, Transcript

Extractor = Callable[[Transcript], Extraction]


@dataclass(frozen=True)
class CaseResult:
    transcript_id: str
    edge_case: str
    expected: Extraction
    actual: Extraction
    escalate_ok: bool
    risk_ok: bool
    flags_ok: bool

    @property
    def passed(self) -> bool:
        return self.escalate_ok and self.risk_ok and self.flags_ok


@dataclass(frozen=True)
class EvalSummary:
    results: tuple[CaseResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def case_accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def escalate_accuracy(self) -> float:
        return self._dim_accuracy(lambda r: r.escalate_ok)

    @property
    def risk_accuracy(self) -> float:
        return self._dim_accuracy(lambda r: r.risk_ok)

    @property
    def flags_accuracy(self) -> float:
        return self._dim_accuracy(lambda r: r.flags_ok)

    def _dim_accuracy(self, pred: Callable[[CaseResult], bool]) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if pred(r)) / self.total

    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.passed)


def _index_transcripts() -> dict[str, Transcript]:
    return {t.id: t for t in TRANSCRIPTS}


def _score_case(label: GoldenLabel, transcript: Transcript, extractor: Extractor) -> CaseResult:
    actual = extractor(transcript)
    return CaseResult(
        transcript_id=label.transcript_id,
        edge_case=transcript.edge_case,
        expected=label.expected,
        actual=actual,
        escalate_ok=actual.escalate == label.expected.escalate,
        risk_ok=actual.risk_level == label.expected.risk_level,
        flags_ok=actual.flags == label.expected.flags,
    )


def score_extractor(extractor: Extractor) -> EvalSummary:
    """Run an extractor across the whole golden set and return a summary."""
    by_id = _index_transcripts()
    results = tuple(
        _score_case(label, by_id[label.transcript_id], extractor) for label in GOLDEN
    )
    return EvalSummary(results=results)


def format_report(name: str, summary: EvalSummary) -> str:
    """Human-readable per-case report for the CLI/README demo."""
    lines = [f"Eval report: {name}", "=" * (13 + len(name))]
    for r in summary.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.transcript_id}  ({r.edge_case})")
        if not r.passed:
            lines.append(f"        expected: {r.expected}")
            lines.append(f"        actual:   {r.actual}")
    lines.append("")
    lines.append(
        f"cases {summary.passed}/{summary.total} "
        f"(case-accuracy {summary.case_accuracy:.3f})  "
        f"escalate {summary.escalate_accuracy:.3f}  "
        f"risk {summary.risk_accuracy:.3f}  "
        f"flags {summary.flags_accuracy:.3f}"
    )
    return "\n".join(lines)
