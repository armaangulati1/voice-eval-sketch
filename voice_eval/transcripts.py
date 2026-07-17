"""Synthetic adversarial transcripts for a voice care-manager evaluation demo.

Every transcript below is INVENTED for this demo. There is no real patient, no
real clinical record, and no data from any company. The clinical details are
deliberately generic and are here only to exercise evaluation logic.

Each transcript plants ONE adversarial edge case that a naive, surface-keyword
extractor tends to miss. The whole point of the eval set is to reward a system
that reads intent and context, not just keywords.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """A single spoken turn in a conversation."""

    speaker: str  # "agent" or "member"
    text: str


@dataclass(frozen=True)
class Transcript:
    """A synthetic voice care-manager conversation.

    `edge_case` names the adversarial phenomenon the transcript is built to test.
    """

    id: str
    edge_case: str
    turns: tuple[Turn, ...]


# ---------------------------------------------------------------------------
# The adversarial set. Seven synthetic conversations.
# ---------------------------------------------------------------------------

TRANSCRIPTS: tuple[Transcript, ...] = (
    Transcript(
        id="t01_masked_distress",
        edge_case="stated_wellbeing_contradicts_content",
        turns=(
            Turn("agent", "Hi, just checking in on how you have been feeling this week."),
            Turn("member", "Oh, I'm fine, really, no need to worry about me."),
            Turn("agent", "That's good to hear. How has your breathing been?"),
            Turn(
                "member",
                "Well, I mean, I have been getting winded just walking to the "
                "kitchen, and I slept sitting up the last two nights because "
                "lying down felt like drowning. But honestly I'm fine.",
            ),
            Turn("agent", "Have you had any swelling in your legs?"),
            Turn(
                "member",
                "A little more than usual, my slippers don't fit. Anyway, it's "
                "nothing. I don't want to make a fuss.",
            ),
        ),
    ),
    Transcript(
        id="t02_sarcasm_nonadherence",
        edge_case="sarcasm_inverts_literal_meaning",
        turns=(
            Turn("agent", "Have you been able to keep up with the new medication?"),
            Turn(
                "member",
                "Oh yeah, absolutely, I take it religiously. Every single day. "
                "Never miss it. Right.",
            ),
            Turn("agent", "Okay, and any side effects?"),
            Turn(
                "member",
                "How would I know? The bottle's still sitting there full on the "
                "counter where the pharmacy left it three weeks ago. But sure, "
                "I'm a model patient.",
            ),
            Turn("agent", "So it sounds like starting it has been hard?"),
            Turn("member", "You could say that."),
        ),
    ),
    Transcript(
        id="t03_buried_safety_disclosure",
        edge_case="safety_disclosure_buried_in_small_talk",
        turns=(
            Turn("agent", "How's the weather over there today?"),
            Turn(
                "member",
                "Oh, gray and drizzly, same as always this time of year. The "
                "garden's a mess. Honestly some mornings I don't see much point "
                "getting out of bed at all, what's the use, but the cat still "
                "needs feeding I suppose. Did you catch the game last night?",
            ),
            Turn("agent", "I missed it, actually. Sounds like a quiet week."),
            Turn(
                "member",
                "Quiet, yeah. Just me and my thoughts, which isn't always great "
                "company these days. Anyway, tell me about that appointment you "
                "were setting up.",
            ),
        ),
    ),
    Transcript(
        id="t04_conflicting_statements",
        edge_case="later_statement_contradicts_earlier_denial",
        turns=(
            Turn("agent", "Any chest pain or pressure since we last spoke?"),
            Turn("member", "No, none at all, all clear on that front."),
            Turn("agent", "Great. How about your activity, getting out much?"),
            Turn(
                "member",
                "I try. Though I had to stop halfway up the stairs yesterday, "
                "there was this tightness in my chest, had to sit on the step a "
                "minute till it passed. Then I was fine.",
            ),
            Turn("agent", "And climbing the stairs today?"),
            Turn("member", "Took it slow. Same tight feeling but shorter."),
        ),
    ),
    Transcript(
        id="t05_genuine_wellbeing_true_negative",
        edge_case="genuine_positive_no_escalation",
        turns=(
            Turn("agent", "How are you doing this week?"),
            Turn(
                "member",
                "Genuinely good, thanks. I walked the full loop around the park "
                "twice with no breathlessness, first time in months.",
            ),
            Turn("agent", "That's wonderful. Medication going okay?"),
            Turn(
                "member",
                "Taking it every morning with breakfast, no side effects, refills "
                "are all set. My daughter checks in daily too.",
            ),
            Turn("agent", "Any pain, dizziness, swelling?"),
            Turn("member", "None of that. Feeling steady. Looking forward to the weekend."),
        ),
    ),
    Transcript(
        id="t06_polite_minimization_fall",
        edge_case="minimizing_language_masks_reportable_event",
        turns=(
            Turn("agent", "Anything eventful since our last call?"),
            Turn(
                "member",
                "Oh, nothing really. I had a little tumble in the bathroom, no "
                "big deal, happens to everyone.",
            ),
            Turn("agent", "A tumble? Can you tell me a bit more?"),
            Turn(
                "member",
                "Just slipped on the tile, knocked my head on the sink on the way "
                "down. Bit of a headache since and the room spun for a while, but "
                "I got myself up eventually. Really, it's fine, don't put it in "
                "any report.",
            ),
        ),
    ),
    Transcript(
        id="t07_indirect_access_barrier",
        edge_case="deference_hides_care_gap",
        turns=(
            Turn("agent", "Were you able to pick up the new prescription?"),
            Turn(
                "member",
                "Oh, I don't want to be a bother about it, it's really not "
                "important.",
            ),
            Turn("agent", "It's no bother at all. Did something come up?"),
            Turn(
                "member",
                "Well, the copay was more than I expected and my ride fell "
                "through, so it's still at the pharmacy. But please don't go to "
                "any trouble, other people need the help more than I do.",
            ),
        ),
    ),
)


def transcript_text(transcript: Transcript) -> str:
    """Flatten a transcript to a single lowercased string of member speech.

    Only member turns carry the signal we score on; the agent prompts are
    context. Lowercasing keeps the naive keyword extractor honest (it is not
    meant to be clever).
    """
    return " ".join(t.text for t in transcript.turns if t.speaker == "member").lower()
