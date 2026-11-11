"""Citation verifier: every quote must appear verbatim in the cited clause."""
import re

_DIACRITICS = re.compile("[ً-ْـ]")
_SPACES = re.compile(r"\s+")
MIN_QUOTE_LEN = 10


def _norm(text: str) -> str:
    # Only strip diacritics/tatweel and collapse whitespace: stay strict on the letters.
    return _SPACES.sub(" ", _DIACRITICS.sub("", text)).strip()


def verify_citation(cit: dict, articles: dict[int, dict]) -> tuple[bool, str]:
    """Return (ok, reason). `articles` maps article number -> article dict."""
    try:
        art = articles.get(int(cit["article"]))
        clause_no = int(cit["clause"])
        quote = _norm(str(cit["quote"]))
    except (KeyError, TypeError, ValueError):
        return False, "malformed citation"
    if art is None:
        return False, "article not found"
    clause = next((c for c in art["clauses"] if c["n"] == clause_no), None)
    if clause is None:
        return False, "clause not found"
    if len(quote) < MIN_QUOTE_LEN:
        return False, "quote too short"
    if quote not in _norm(clause["ar"]):
        return False, "quote not found in clause text"
    return True, "ok"
