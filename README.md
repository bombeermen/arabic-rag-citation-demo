# Arabic RAG chatbot with verified citations (demo slice)

A small, working slice of a RAG chatbot that answers civil servants' questions **only from official regulations**, cites the exact article and clause, and **refuses** when the regulations do not answer the question.

The regulation in `data/` is **synthetic** (27 articles, 55 clauses, Arabic and English in parallel), generated for this demo. It is not a real legal text.

## How it works

1. **Retrieval** (`retrieve.py`): BM25 over articles, implemented without dependencies. Arabic text is normalized (diacritics, alef variants) and lightly stemmed.
2. **Answer** (`answer.py`): top-5 articles are sent to Claude Haiku, which must reply in a fixed JSON format: a short answer plus citations (`article`, `clause`, verbatim `quote`).
3. **Citation check** (`verify.py`): every quote must appear verbatim in the cited clause. A citation that does not match is marked unverified.
4. **Refusal**: if no citation is verified, the bot returns a refusal instead of an answer.

Not included on purpose: embeddings, reranker, local model, classifier, UI.

## Results (15 questions: 10 answerable, 5 not covered by the regulation)

| Metric | Result |
|---|---|
| Recall@5 (retrieval) | 10/10 |
| Verified citations | 12/12 (100%) |
| Refusal accuracy | 5/5 |
| False refusals on answerable questions | 0/10 |

Total cost of the full run: about 14.6k tokens with Claude Haiku. Per-question output is in `results.json`.

**Caveat:** the regulation and the questions were written together, so wording overlaps and these numbers are optimistic. This is a small demo, not a benchmark on real documents.

## Run it

You need: Python 3.12, Git, and your own Anthropic API key (create one at https://console.anthropic.com/settings/keys; a few dollars of credit is plenty, the full evaluation costs a few cents). Commands below are for Windows PowerShell; on macOS/Linux use `python3`, `.venv/bin/python` and `export ANTHROPIC_API_KEY=...`.

```powershell
git clone https://github.com/bombeermen/arabic-rag-citation-demo.git
cd arabic-rag-citation-demo

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# set the key for this terminal session only (not stored in any file)
$secure = Read-Host "Anthropic API key" -AsSecureString
$env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new("", $secure).Password

.\.venv\Scripts\python.exe retrieve.py       # retrieval only, no API calls
.\.venv\Scripts\python.exe answer.py 1 11    # try questions 1 and 11
.\.venv\Scripts\python.exe evaluate.py       # all 15 questions
```

The API key is read from the `ANTHROPIC_API_KEY` environment variable and is never committed.

## Files

- `data/regulation.json`: synthetic regulation, article → numbered clauses, Arabic and English
- `data/questions.json`: 15 evaluation questions with gold article numbers
- `retrieve.py`, `answer.py`, `verify.py`, `evaluate.py`: the pipeline
- `results.json`: output of the last evaluation run
