"""Two reference 'models' under evaluation, plus an optional LLM hook.

The whole file is deterministic and offline by default. Nothing here calls a
network unless you explicitly opt into the LLM extractor with an API key, and
even then the eval never depends on it (see judge.py for why).

The two built-in extractors exist to make the eval set *do something visible*:

  naive_keyword_extractor  -- surface keyword + literal-sentiment matching.
                              Passes the genuine-positive control and little
                              else; it is fooled by every adversarial framing.
  careful_rule_extractor   -- a hand-authored, context-aware reference tuned to
                              this synthetic set. A high score here means the
                              rules encode the labels, NOT that the approach
                              generalizes. It is a demonstration target, not a
                              production classifier.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from voice_eval.golden import Extraction
from voice_eval.transcripts import Transcript, transcript_text

Extractor = Callable[[Transcript], Extraction]


# ---------------------------------------------------------------------------
# 1. Naive keyword extractor: how a surface-level accuracy eval passes.
# ---------------------------------------------------------------------------

# Literal reassurance cues that a naive system reads as "member is okay".
_REASSURANCE_CUES = (
    "i'm fine",
    "no big deal",
    "all clear",
    "nothing",
    "not important",
    "don't want to be a bother",
    "religiously",
    "model patient",
)

# Only the most explicit danger tokens trip a naive escalation.
_EXPLICIT_DANGER = (
    "chest pain",
    "can't breathe",
    "suicide",
    "emergency",
    "call 911",
)


def naive_keyword_extractor(transcript: Transcript) -> Extraction:
    text = transcript_text(transcript)
    reassured = any(cue in text for cue in _REASSURANCE_CUES)
    danger = any(tok in text for tok in _EXPLICIT_DANGER)

    if danger and not reassured:
        return Extraction(escalate=True, risk_level="high", flags=frozenset())
    # Literal reassurance suppresses escalation -- the core naive failure mode.
    return Extraction(escalate=False, risk_level="none", flags=frozenset())


# ---------------------------------------------------------------------------
# 2. Careful rule extractor: reads intent/context, not just keywords.
# ---------------------------------------------------------------------------

_WORSENING_CUES = (
    "winded",
    "drowning",
    "swelling",
    "slippers don't fit",
    "sitting up",
)
_NONADHERENCE_CUES = (
    "still sitting there full",
    "bottle's still",
    "full on the counter",
    "pharmacy left it",
)
_SARCASM_ADHERENCE_CUES = ("religiously", "model patient", "never miss")
_SAFETY_CUES = (
    "don't see much point",
    "what's the use",
    "not always great company",
    "just me and my thoughts",
)
_CONFLICT_DENIAL_CUES = ("no, none at all", "all clear")
_CONFLICT_SYMPTOM_CUES = ("tightness in my chest", "tight feeling")
_FALL_CUES = ("tumble", "slipped on the tile", "knocked my head")
_FALL_SEVERITY_CUES = ("headache", "room spun", "knocked my head")
_ACCESS_CUES = (
    "copay",
    "ride fell through",
    "still at the pharmacy",
    "more than i expected",
)


def _any(text: str, cues: tuple[str, ...]) -> bool:
    return any(c in text for c in cues)


def careful_rule_extractor(transcript: Transcript) -> Extraction:
    text = transcript_text(transcript)
    flags: set[str] = set()
    risk = "none"

    def raise_risk(level: str) -> None:
        nonlocal risk
        order = ("none", "low", "moderate", "high")
        if order.index(level) > order.index(risk):
            risk = level

    # Safety disclosure buried in small talk -> highest priority.
    if _any(text, _SAFETY_CUES):
        flags.add("safety_disclosure")
        raise_risk("high")

    # Conflicting report: a denial plus a later specific symptom.
    if _any(text, _CONFLICT_DENIAL_CUES) and _any(text, _CONFLICT_SYMPTOM_CUES):
        flags.add("conflicting_report")
        flags.add("symptom_worsening")
        raise_risk("high")

    # Fall with head strike / vertigo, even when minimized.
    if _any(text, _FALL_CUES) and _any(text, _FALL_SEVERITY_CUES):
        flags.add("fall_event")
        raise_risk("high")

    # Symptom worsening described while stating "I'm fine".
    if _any(text, _WORSENING_CUES):
        flags.add("symptom_worsening")
        raise_risk("moderate")

    # Nonadherence hidden behind sarcasm ("I take it religiously").
    if _any(text, _NONADHERENCE_CUES) and _any(text, _SARCASM_ADHERENCE_CUES):
        flags.add("medication_nonadherence")
        raise_risk("moderate")

    # Access barrier hidden behind deference.
    if _any(text, _ACCESS_CUES):
        flags.add("access_barrier")
        raise_risk("low")

    escalate = len(flags) > 0
    return Extraction(escalate=escalate, risk_level=risk, flags=frozenset(flags))


# ---------------------------------------------------------------------------
# 3. Optional LLM extractor: opt-in, never required by the offline suite.
# ---------------------------------------------------------------------------


def llm_extractor_available() -> bool:
    """True only if an API key is present. The offline suite never needs this."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_extractor(transcript: Transcript) -> Extraction:  # pragma: no cover
    """Thin, opt-in LLM extractor stub.

    Intentionally NOT wired to a live client in this demo. If you wanted to
    evaluate a real model, you would implement the call here and then run it
    through the SAME scorer and the SAME excluded-judge discipline as everything
    else. The eval harness treats an LLM output as just another candidate to be
    checked against the golden labels -- never as an authority.
    """
    raise NotImplementedError(
        "LLM extractor is a documented stub in this demo. Implement a client "
        "call here to evaluate a real model; it would then be scored by the "
        "identical golden-set logic used for the deterministic extractors."
    )
