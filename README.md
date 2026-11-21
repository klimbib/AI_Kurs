# Datasheet RAG bot for transistors

A question-answering bot over transistor datasheets (PDF), overview tables (Markdown) and purchasing prices.
It finds the relevant chunks with hybrid search (vector + BM25 + reranker), answers with `openai/gpt-4o-mini`
only from those sources, cites them, and says "I could not find this" when the documents do not contain the answer.
Access groups (`data/access.json`) decide which documents a user may see. Results are measured against a test set
(`evals/testset.json`) and a baseline (`baseline.json`). Results and problems: see [REPORT.md](REPORT.md).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY=sk-or-...`.
The first search run downloads the embedding and reranker models (~300 MB) into `.models/`.

## Start as a bot

```powershell
python main.py "Which maximum current has transistor type IRF540N?"
python main.py --groups purchasing "Which P-channel MOSFET is the cheapest?"
```

## Start the full chain 

```powershell
python main.py            # all 5 steps
python main.py --full     # same, but re-ingest every file
```

Chunk size (120 tokens) -> uses a original value, because small (read only first page with main inforamation),  well structured documents. The test with other chunk size targets (80,150) have not influenced the results, because the the size of generated chunks are mostly defined by document structure. 

