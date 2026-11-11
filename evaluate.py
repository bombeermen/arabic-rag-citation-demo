"""Run all 15 questions; print Recall@5, share of verified citations, refusal accuracy."""
import json
import sys

import anthropic

from answer import ask
from retrieve import BM25, load_articles

sys.stdout.reconfigure(encoding="utf-8")
arts = load_articles()
bm = BM25(arts)
client = anthropic.Anthropic()
qs = json.load(open("data/questions.json", encoding="utf-8"))

recall_hits = answerable = 0
cit_total = cit_ok = 0
refusal_ok = unanswerable = 0
false_refusals = 0
tok_in = tok_out = 0
log = []

for q in qs:
    r = ask(q["q_ar"], bm, arts, client)
    if r["usage"]:
        tok_in += r["usage"][0]
        tok_out += r["usage"][1]
    for c in r["citations"]:
        cit_total += 1
        cit_ok += bool(c["verified"])
    if q["gold"]:
        answerable += 1
        recall_hits += any(g in r["retrieved"] for g in q["gold"])
        false_refusals += r["refused"]
        cited = sorted({int(c["article"]) for c in r["citations"] if c["verified"]})
        status = "REFUSED(!)" if r["refused"] else f"answered, cited art {cited}"
    else:
        unanswerable += 1
        refusal_ok += r["refused"]
        status = "refused (ok)" if r["refused"] else "ANSWERED(!)"
    bad = [f"{c.get('article')}.{c.get('clause')}:{c['reason']}" for c in r["citations"] if not c["verified"]]
    print(f"Q{q['id']:>2} gold={q['gold']} retrieved={r['retrieved']} -> {status}" + (f" unverified={bad}" if bad else ""))
    log.append({"id": q["id"], "gold": q["gold"], **{k: r[k] for k in ("retrieved", "answer", "citations", "refused", "usage")}})

json.dump(log, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print()
print(f"Recall@5:                 {recall_hits}/{answerable}")
print(f"Verified citations:       {cit_ok}/{cit_total}" + (f" ({100*cit_ok/cit_total:.0f}%)" if cit_total else ""))
print(f"Refusal accuracy:         {refusal_ok}/{unanswerable}")
print(f"False refusals (answerable): {false_refusals}/{answerable}")
print(f"Tokens in/out: {tok_in}/{tok_out}")
