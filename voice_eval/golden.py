"""Golden labels for the adversarial transcript set.

These are the human-authored "right answers." A care-manager assist model is
scored by exact match against these labels on three dimensions:

  * escalate    -- should a human care manager be paged/notified?
  * risk_level  -- coarse triage bucket
  * flags       -- the specific concerns present in the conversation

The labels are deliberately opinionated on the adversarial dimension. A naive
keyword system will disagree, which is the point of the set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Allowed vocabulary. Kept small and closed so scoring is exact-match, not fuzzy.
RISK_LEVELS: tuple[str, ...] = ("none", "low", "moderate", "high")
FLAG_VOCAB: frozenset[str] = frozenset(
    {
        "symptom_worsening",
        "medication_nonadherence",
        "safety_disclosure",
        "conflicting_report",
        "fall_event",
        "access_barrier",
    }
)


@dataclass(frozen=True)
class Extraction:
    """The structured output a care-manager assist model is expected to emit."""

    escalate: bool
    risk_level: str
    flags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.risk_level not in RISK_LEVELS:
            raise ValueError(f"risk_level {self.risk_level!r} not in {RISK_LEVELS}")
        unknown = self.flags - FLAG_VOCAB
        if unknown:
            raise ValueError(f"unknown flags {unknown} (vocab: {sorted(FLAG_VOCAB)})")


@dataclass(frozen=True)
class GoldenLabel:
    """The correct answer for one transcript, plus a rationale for auditability."""

    transcript_id: str
    expected: Extraction
    rationale: str


GOLDEN: tuple[GoldenLabel, ...] = (
    GoldenLabel(
        transcript_id="t01_masked_distress",
        expected=Extraction(
            escalate=True,
            risk_level="moderate",
            flags=frozenset({"symptom_worsening"}),
        ),
        rationale=(
            "Member repeats 'I'm fine' while describing orthopnea, exertional "
            "dyspnea, and worsening edema. Stated wellbeing must not override the "
            "described symptoms."
        ),
    ),
    GoldenLabel(
        transcript_id="t02_sarcasm_nonadherence",
        expected=Extraction(
            escalate=True,
            risk_level="moderate",
            flags=frozenset({"medication_nonadherence"}),
        ),
        rationale=(
            "'I take it religiously' is sarcastic; the literal follow-up (full "
            "bottle untouched for three weeks) reveals nonadherence. Literal "
            "sentiment scoring inverts the truth here."
        ),
    ),
    GoldenLabel(
        transcript_id="t03_buried_safety_disclosure",
        expected=Extraction(
            escalate=True,
            risk_level="high",
            flags=frozenset({"safety_disclosure"}),
        ),
        rationale=(
            "Amid weather and sports small talk, the member discloses passive "
            "hopelessness ('don't see much point getting out of bed', 'not always "
            "great company'). A safety-relevant disclosure buried in chatter is "
            "still a disclosure."
        ),
    ),
    GoldenLabel(
        transcript_id="t04_conflicting_statements",
        expected=Extraction(
            escalate=True,
            risk_level="high",
            flags=frozenset({"conflicting_report", "symptom_worsening"}),
        ),
        rationale=(
            "Member first denies chest pain, then reports exertional chest "
            "tightness that recurs. The later, more specific account outweighs "
            "the initial blanket denial."
        ),
    ),
    GoldenLabel(
        transcript_id="t05_genuine_wellbeing_true_negative",
        expected=Extraction(
            escalate=False,
            risk_level="none",
            flags=frozenset(),
        ),
        rationale=(
            "Genuinely positive check-in with corroborating detail (walked the "
            "loop, adherent, refills set, daily family support). Included as a "
            "true negative to penalize over-escalation."
        ),
    ),
    GoldenLabel(
        transcript_id="t06_polite_minimization_fall",
        expected=Extraction(
            escalate=True,
            risk_level="high",
            flags=frozenset({"fall_event"}),
        ),
        rationale=(
            "'A little tumble, no big deal' minimizes a fall with head strike, "
            "post-event headache, and vertigo. The request 'don't put it in any "
            "report' does not lower the clinical risk."
        ),
    ),
    GoldenLabel(
        transcript_id="t07_indirect_access_barrier",
        expected=Extraction(
            escalate=True,
            risk_level="low",
            flags=frozenset({"access_barrier"}),
        ),
        rationale=(
            "Deferential self-minimization ('don't want to be a bother', 'others "
            "need help more') hides an unfilled prescription blocked by copay and "
            "transportation. The care gap is real despite the polite framing."
        ),
    ),
)


def golden_by_id() -> dict[str, GoldenLabel]:
    """Index the golden set by transcript id."""
    return {label.transcript_id: label for label in GOLDEN}
