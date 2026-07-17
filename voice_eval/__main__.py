"""Run the demo offline:  python -m voice_eval

Prints the eval report for both built-in extractors and the excluded-judge
validation. No network, no API key required.
"""

from __future__ import annotations

from voice_eval.extractor import careful_rule_extractor, naive_keyword_extractor
from voice_eval.judge import (
    JudgeExcludedError,
    faithful_judge_stub,
    judge_escalation_rate,
    literal_sentiment_judge_stub,
    summarize_validation,
    validate_judge,
)
from voice_eval.scorer import format_report, score_extractor


def main() -> None:
    print(format_report("naive_keyword_extractor", score_extractor(naive_keyword_extractor)))
    print()
    print(format_report("careful_rule_extractor", score_extractor(careful_rule_extractor)))
    print()
    print("Excluded-judge discipline")
    print("=========================")
    for judge, name in (
        (faithful_judge_stub, "faithful_judge_stub"),
        (literal_sentiment_judge_stub, "literal_sentiment_judge_stub"),
    ):
        print(summarize_validation(validate_judge(judge, name)))

    # A validated judge may contribute a number; an excluded one may not.
    print()
    for judge, name in (
        (faithful_judge_stub, "faithful_judge_stub"),
        (literal_sentiment_judge_stub, "literal_sentiment_judge_stub"),
    ):
        try:
            rate = judge_escalation_rate(judge, name)
            print(f"{name}: escalation-rate estimate = {rate:.3f}")
        except JudgeExcludedError as exc:
            print(f"{name}: refused -> {exc}")


if __name__ == "__main__":
    main()
