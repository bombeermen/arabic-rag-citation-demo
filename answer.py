"""Answer a question: BM25 -> Claude Haiku (JSON with verbatim quotes) -> verify -> refuse if nothing verified."""
import json
import re
import sys

import anthropic

from retrieve import BM25, load_articles
from verify import verify_citation

MODEL = "claude-haiku-5-5"
TOP_K = 5
REFUSAL_AR = "لا تتوفر إجابة على هذا السؤال في اللوائح الرسمية."

SYSTEM = (
    "You answer questions from civil servants strictly using the regulation excerpts provided. "
    "Never use outside knowledge. Respond with ONLY a JSON object, no other text, in this exact format:\n"
    '{"answer": "<short answer in Arabic>", "citations": '
    '[{"article": <int>, "clause": <int>, "quote": "<verbatim Arabic text>"}]}\n'
    "Rules: each quote must be copied character for character from the Arabic text of the cited clause "
    "(a contiguous fragment of that single clause). Cite only clauses that directly support the answer. "
    'If the excerpts do not answer the question, return {"answer": "", "citations": []}.'
)


def build_context(ids: list[int], articles: dict[int, dict]) -> str:
    parts = []
    for i in ids:
        a = articles[i]
        lines = [f"[Article {i}] {a['title_ar']}"]
        lines += [f"({i}.{c['n']}) {c['ar']}" for c in a["clauses"]]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def ask(question: str, bm: BM25, articles: dict[int, dict], client: anthropic.Anthropic) -> dict:
    hits = bm.search(question, TOP_K)
    ids = [a for a, _ in hits]
    result = {"retrieved": ids, "answer": "", "citations": [], "refused": True, "usage": None}
    if not ids:
        result["answer"] = REFUSAL_AR
        return result

    user = f"Regulation excerpts:\n\n{build_context(ids, articles)}\n\nQuestion: {question}"
    resp = client.messages.create(
        model=MODEL, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    result["usage"] = (resp.usage.input_tokens, resp.usage.output_tokens)
    text = "".join(b.text for b in resp.content if b.type == "text")
    result["raw"] = text
    try:
        data = parse_json(text)
    except json.JSONDecodeError:
        data = {}

    cits = data.get("citations") or []
    for c in cits:
        ok, reason = verify_citation(c, articles)
        c["verified"], c["reason"] = ok, reason
    result["citations"] = cits
    if any(c["verified"] for c in cits):
        result["refused"] = False
        result["answer"] = data.get("answer", "")
    else:
        result["answer"] = REFUSAL_AR
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    arts = load_articles()
    bm = BM25(arts)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    qs = {q["id"]: q for q in json.loads(open("data/questions.json", encoding="utf-8").read())}
    for qid in [int(a) for a in sys.argv[1:] if a.isdigit()]:
        q = qs[qid]
        r = ask(q["q_ar"], bm, arts, client)
        print(f"=== Q{qid}: {q['q_en']}")
        print("retrieved:", r["retrieved"], "| gold:", q["gold"], "| tokens(in,out):", r["usage"])
        for c in r["citations"]:
            print(f"  cite art {c.get('article')}.{c.get('clause')} verified={c['verified']} ({c['reason']})")
            print(f"    quote: {c.get('quote')}")
        print("REFUSED" if r["refused"] else "ANSWERED", "->", r["answer"])
        if "--debug" in sys.argv or r["refused"]:
            print("RAW:", r.get("raw"))
