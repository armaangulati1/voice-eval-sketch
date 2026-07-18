# voice-eval-sketch

A small, self-authored demonstration of adversarial evaluation for voice-care
conversation AI. All transcripts are synthetic and invented for this demo. Not
affiliated with, and not built on data from, any company. Demo scope,
self-authored synthetic data, offline-reproducible.

---

## What this is

Naive accuracy evals reward a model for getting the easy cases right. The hard,
consequential cases in a care-navigation conversation are the ones where the
literal words and the real meaning diverge: a member who says "I'm fine" while
describing worsening symptoms, sarcasm that inverts a medication answer, a
safety-relevant disclosure buried in small talk, a fall softened into "a little
tumble." A model can score well on average and still miss exactly these.

This repo is a compact sketch of how I would evaluate for that. It contains:

- **A hand-authored adversarial golden set** of 7 synthetic conversation
  transcripts, each planting one ambiguous or emotionally loaded edge case that
  surface-level scoring tends to miss (`voice_eval/transcripts.py`,
  `voice_eval/golden.py`).
- **A deterministic, exact-match scorer** that grades an extraction/classification
  output per case on three dimensions (escalate / risk level / concern flags),
  with no partial credit and no fuzzy matching (`voice_eval/scorer.py`).
- **Explicit excluded-judge discipline**: if an LLM-as-judge is used, it is first
  validated against the human golden labels and **excluded from scoring** if its
  agreement falls below a threshold, rather than being silently averaged in
  (`voice_eval/judge.py`).
- **Offline tests, ruff-clean lint, and a CI workflow.**

Everything runs offline with the standard library. No API key is required.

## The demonstrated gap

Two reference "models" ship with the repo so the eval set visibly does work.
Run `python -m voice_eval`:

| extractor | cases passed | what it does |
| --- | --- | --- |
| `naive_keyword_extractor` | **1 / 7** | surface keywords + literal sentiment; fooled by every adversarial framing, passes only the genuine-positive control |
| `careful_rule_extractor` | **7 / 7** | hand-authored, context-aware reference tuned to this set |

The naive extractor is the illustration of the problem: it "reads" reassurance
literally and escalates only on the most explicit danger words, so it collapses
to 1/7 on the adversarial set while looking fine on the easy negative. The gap
between the two columns is the entire value of an adversarial eval set.

**Honest caveat on the 7/7.** `careful_rule_extractor` is a rule-based reference
tuned by hand to this specific synthetic set. A perfect score here means the
rules encode the labels, **not** that the approach generalizes. It is a
demonstration target for the scorer, not a production classifier. The
interesting artifact is the eval set and the discipline around it, not the
extractor.

## Excluded-judge discipline

An LLM-as-judge is convenient but must never move the numbers unvalidated.
Before any judge contributes to a score, `validate_judge` checks its agreement
against the human golden labels. Two deterministic judge stubs make the rule
testable offline:

```
[ADMITTED] judge=faithful_judge_stub          agreement=1.000  threshold=0.900
[EXCLUDED] judge=literal_sentiment_judge_stub  agreement=0.143  threshold=0.900
```

An excluded judge does not get down-weighted; it raises `JudgeExcludedError` and
contributes nothing. You fall back to the deterministic scorer. A real LLM judge
would drop in behind the same `Judge` signature and pass through the identical
gate.

## Run it

```bash
pip install -e ".[dev]"
python -m voice_eval     # prints both eval reports + judge validation
ruff check .
pytest -q                # offline test suite
```

Optional live LLM extractor: `voice_eval/extractor.py` has a documented
`llm_extractor` stub. It is intentionally not wired to a client here. If you
implemented it, its output would be scored by the exact same golden-set logic
as everything else, and any LLM judge would face the same exclusion gate.

## Layout

```
voice_eval/
  transcripts.py   # 7 synthetic adversarial transcripts (invented for this demo)
  golden.py        # human-authored golden labels + rationales, closed vocabulary
  extractor.py     # naive vs careful reference extractors + optional LLM stub
  scorer.py        # deterministic exact-match scorer, per-case + per-dimension
  judge.py         # excluded-judge discipline (validate -> admit or exclude)
  __main__.py      # `python -m voice_eval` demo runner
tests/             # offline pytest: golden integrity, scorer gap, judge exclusion, no-names guard
```

## Scope and honesty

- **Synthetic only.** Every transcript is invented for this demo. There is no
  real patient, no real clinical record, and no data from any company. The
  clinical detail is deliberately generic and exists only to exercise eval logic.
- **Not affiliated** with any company or product.
- **Not production.** This is a ~1-2 hour scoped sketch of a methodology, not a
  system. It demonstrates how I think about adversarial evaluation and judge
  discipline; it is not a shipped classifier.
- A test (`tests/test_no_company_names.py`) enforces that the artifact stays
  generic: no company or product name from its blocklist appears anywhere in it.
  The blocklist is loaded from `tests/vendor_blocklist.local.txt`, which is
  gitignored and ships with nobody's name in it;
  `tests/vendor_blocklist.example.txt` is the committed template and contains
  invented placeholder tokens only. An earlier version of this repo hard coded
  a list of real names in the test file itself, which was both a neutrality
  failure and a targeting signal, and it forced the guard to exempt itself from
  its own scan. The guard now scans every text file including itself. Note the
  guard's scope: it checks the working tree, not git history, and it only
  catches names that are on its list.
