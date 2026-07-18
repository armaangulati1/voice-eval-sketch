"""Guard: the artifact stays generic and vendor neutral.

This repo is a reusable eval sketch for voice transcript extraction. It is
deliberately not about any particular product, so no company or product name
may appear anywhere in it. A specific company belongs in the message that links
to this repo, never in the repo.

WHY THE BLOCKLIST IS NOT COMMITTED
----------------------------------
An earlier version of this guard hard coded a list of real company names. That
was a mistake, and not a small one. In an artifact that is otherwise
deliberately generic, a named list of companies is the single thing that
reveals which companies the author had in mind. It turns neutral tooling into a
targeting artifact, and it also made the README's generic claim false, since
the guard had to exempt itself from its own scan to pass.

So the real blocklist now lives in ``vendor_blocklist.local.txt``, which is
gitignored. ``vendor_blocklist.example.txt`` is committed and holds only
invented placeholder tokens, so the guard and its meta tests still work on a
fresh clone. The guard therefore no longer needs a self exemption: it scans
every text file including itself.

Word boundaries matter, so a token must not trip on ordinary words that happen
to contain it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", ".ruff_cache", ".pytest_cache", ".venv", "__pycache__"}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".cfg", ".ini"}

# The blocklist files are the one legitimate place a blocked token may appear,
# because listing the tokens is their entire function. This is a targeted
# exemption for two data files, not the old blanket self exemption of the test
# module that also carried the claim.
BLOCKLIST_FILENAMES = {"vendor_blocklist.local.txt", "vendor_blocklist.example.txt"}

LOCAL_BLOCKLIST = Path(__file__).resolve().parent / "vendor_blocklist.local.txt"
EXAMPLE_BLOCKLIST = Path(__file__).resolve().parent / "vendor_blocklist.example.txt"


def _load_tokens() -> tuple[str, ...]:
    """Tokens from the local blocklist if present, otherwise the example.

    The example holds invented placeholders only, so a fresh clone gets a guard
    that demonstrably works without shipping anyone's name.
    """
    source = LOCAL_BLOCKLIST if LOCAL_BLOCKLIST.exists() else EXAMPLE_BLOCKLIST
    tokens = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            tokens.append(stripped.lower())
    return tuple(tokens)


FORBIDDEN = _load_tokens()
FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(token) for token in FORBIDDEN) + r")\b",
    re.IGNORECASE,
)


def _iter_text_files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def test_no_forbidden_company_names_in_repo() -> None:
    offenders: list[str] = []
    for path in _iter_text_files():
        if path.name in BLOCKLIST_FILENAMES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_RE.search(text):
            hits = sorted({match.group(0).lower() for match in FORBIDDEN_RE.finditer(text)})
            offenders.append(f"{path.relative_to(ROOT)} contains {hits}")
    assert not offenders, "company names leaked into the artifact: " + "; ".join(offenders)


def test_the_guard_actually_scans_files() -> None:
    """A guard that silently scans nothing would always pass."""
    scanned = list(_iter_text_files())
    assert len(scanned) >= 10
    assert any(path.name == "README.md" for path in scanned)
    assert any(path.suffix == ".py" for path in scanned)


def test_the_guard_scans_its_own_module() -> None:
    """The old guard exempted itself because it carried a list of real names.

    It carries none now, so it must be inside its own scan. If this file ever
    reacquires a name, the guard has to be the thing that catches it.
    """
    scanned = {path.resolve() for path in _iter_text_files()}
    assert Path(__file__).resolve() in scanned


def test_the_blocklist_is_non_empty() -> None:
    """A guard compiled from an empty list would match nothing and always pass."""
    assert FORBIDDEN, "no blocklist tokens loaded"


def test_the_guard_catches_a_planted_name() -> None:
    planted = FORBIDDEN[0]
    assert FORBIDDEN_RE.search(f"we integrate with {planted} today")
    assert FORBIDDEN_RE.search(f"We Integrate With {planted.upper()} Today")


def test_the_guard_respects_word_boundaries() -> None:
    for benign in ("message", "usage", "passage", "transcript", "utterance", "call"):
        assert not FORBIDDEN_RE.search(benign), benign
    # A blocked token embedded in a longer word must not trip either.
    embedded = FORBIDDEN[0] + "ish"
    assert not FORBIDDEN_RE.search(embedded), embedded


def test_the_real_blocklist_is_never_committed() -> None:
    """The local blocklist must stay untracked, which is the whole point.

    The example file is the committed one and holds placeholders only.
    """
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "vendor_blocklist.local.txt" in gitignore
