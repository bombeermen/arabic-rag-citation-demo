"""BM25 retrieval over regulation articles (Arabic text), no external dependencies."""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"

_DIACRITICS = re.compile("[ً-ْـ]")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("ات", "ون", "ين", "ان", "ها", "ه", "ا", "ي")


def normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    for a in "أإآ":
        text = text.replace(a, "ا")
    return text.replace("ى", "ي").replace("ة", "ه")


_STOP = {normalize(w) for w in (
    "ما هي هو هل كم كيف في من الى على عن ان او و التي الذي هذا هذه ذلك بعد قبل "
    "عند خلال يجوز ان لا ثم كل اي").split()}


def stem(tok: str) -> str:
    for p in _PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            tok = tok[len(p):]
            break
    for s in _SUFFIXES:
        if tok.endswith(s) and len(tok) - len(s) >= 3:
            tok = tok[: -len(s)]
            break
    return tok


def tokenize(text: str) -> list[str]:
    toks = _NON_WORD.split(normalize(text).lower())
    return [stem(t) for t in toks if t and t not in _STOP]


def load_articles() -> dict[int, dict]:
    reg = json.loads((DATA / "regulation.json").read_text(encoding="utf-8"))
    return {a["article"]: a for a in reg["articles"]}


def article_text_ar(a: dict) -> str:
    return a["title_ar"] + " " + " ".join(c["ar"] for c in a["clauses"])


class BM25:
    def __init__(self, articles: dict[int, dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids = list(articles)
        self.docs = [Counter(tokenize(article_text_ar(articles[i]))) for i in self.ids]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = tokenize(query)
        scored = []
        for aid, d, ln in zip(self.ids, self.docs, self.lens):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            scored.append((aid, s))
        scored.sort(key=lambda x: -x[1])
        return [x for x in scored[:k] if x[1] > 0]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    arts = load_articles()
    bm = BM25(arts)
    qs = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    hits = total = 0
    for q in qs:
        res = bm.search(q["q_ar"], 5)
        ids = [a for a, _ in res]
        if q["gold"]:
            total += 1
            ok = any(g in ids for g in q["gold"])
            hits += ok
            print(f"Q{q['id']:>2} gold={q['gold']} top5={ids} {'OK' if ok else 'MISS'}")
        else:
            print(f"Q{q['id']:>2} (no answer) top5={ids}")
    print(f"Recall@5 = {hits}/{total}")
