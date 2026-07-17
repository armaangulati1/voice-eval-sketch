"""voice-eval-sketch: a small, self-authored adversarial evaluation demo.

All transcripts are synthetic and invented for this demo. Not affiliated with,
and not built on data from, any company. Demo scope, self-authored synthetic
data, offline-reproducible.
"""

from voice_eval.extractor import (
    careful_rule_extractor,
    naive_keyword_extractor,
)
from voice_eval.golden import GOLDEN, GoldenLabel
from voice_eval.scorer import CaseResult, EvalSummary, score_extractor
from voice_eval.transcripts import TRANSCRIPTS, Transcript, Turn

__all__ = [
    "TRANSCRIPTS",
    "Transcript",
    "Turn",
    "GOLDEN",
    "GoldenLabel",
    "naive_keyword_extractor",
    "careful_rule_extractor",
    "score_extractor",
    "CaseResult",
    "EvalSummary",
]
