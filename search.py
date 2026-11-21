"""Lesson 7 practice (starter): vector, BM25, hybrid and reranked search over the lesson 6 chunks.

Usage:
    python starter/search.py                  # evaluate all configurations on eval/testset.json
    python starter/search.py "vpn backup"    # show the top results of every configuration for one query
    python starter/search.py --groups finance "operating costs"
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
CHUNKS = ROOT / "output" / "ingest" / "chunks.jsonl"
QUESTIONS = ROOT / "evals" / "testset.json"
OUT = ROOT / "output" / "search"
CACHE = OUT / "embeddings_cache.json"
MODELS = ROOT / ".models"   # downloaded once, ~300 MB

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # 384 dims, ~220 MB, uk + en
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"                                # ~80 MB cross-encoder
COLLECTION = "northwind"
K = 5                # how many results we evaluate (recall@K)
CANDIDATES = 20      # how many hybrid candidates go to the reranker
RRF_K = 60

_embedder = None
_reranker = None


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]


def text_for_embedding(chunk: dict) -> str:
    """Contextual header: the section path in front of the text (lesson 6, chunking)."""
    return f"{chunk['section']}\n{chunk['text']}" if chunk["section"] else chunk["text"]


# ---------------------------------------------------------------- embeddings

# TODO 1-2 · Goal: turn every chunk into a vector, and do it only once.
#   Why: search compares vectors, not words; the model is slow, so unchanged chunks must come from the cache.
#   Done when: similar texts get closer vectors (check 1) and a second run prints "embedded now 0" (check 2).
def embed(texts: list[str]) -> list[list[float]]:
    """TODO 1: embed texts with EMBED_MODEL.
    TODO 2: cache vectors in output/embeddings_cache.json by sha256(model + text): a re-run embeds nothing."""
    global _embedder
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    keys = [hashlib.sha256((EMBED_MODEL + text).encode("utf-8")).hexdigest() for text in texts]
    missing = [i for i, key in enumerate(keys) if key not in cache]
    if missing:
        if _embedder is None:
            _embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(MODELS))
        vectors = _embedder.embed([texts[i] for i in missing])
        for i, vector in zip(missing, vectors):
            cache[keys[i]] = [float(x) for x in vector]
        OUT.mkdir(parents=True, exist_ok=True)
        lines = [f"{json.dumps(key)}: {json.dumps(vector)}" for key, vector in cache.items()]
        CACHE.write_text("{\n" + ",\n".join(lines) + "\n}", encoding="utf-8")   # one hash + vector per line
    embed.last_missing = len(missing)
    return [cache[key] for key in keys]


embed.last_missing = 0


# ---------------------------------------------------------------- index

# TODO 3 · Goal: put all chunk vectors into Qdrant together with their metadata.
#   Why: the vector database finds the nearest vectors fast and returns the chunk (source, section, access) with them.
#   Done when: the collection has 69 points and each point carries its chunk as payload (check 3).
def build_index(chunks: list[dict]) -> QdrantClient:
    """TODO 3: create a Qdrant collection (cosine) and upsert one point per chunk with the chunk as payload."""
    vectors = embed([text_for_embedding(c) for c in chunks])
    client = QdrantClient(":memory:")
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    client.upsert(
        COLLECTION,
        points=[models.PointStruct(id=i, vector=vector, payload=chunk)
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))],
    )
    return client


# TODO 8 · Goal: a user only gets chunks of the groups they belong to.
#   Why: access rules from lesson 6 must work inside the search; a prompt "do not show finance data" is not protection.
#   Done when: the Q3 budget is hidden from all_employees but found for finance (check 8, "access filter: OK").
def access_filter(groups: list[str]) -> models.Filter:
    """TODO 8: only chunks whose `access` shares at least one group with the user."""
    return models.Filter(must=[models.FieldCondition(key="access", match=models.MatchAny(any=groups))])


def allowed(chunk: dict, groups: list[str]) -> bool:
    return bool(set(chunk["access"]) & set(groups))


# ---------------------------------------------------------------- search

# TODO 4 · Goal: semantic search: the query becomes a vector, Qdrant returns the k nearest chunks.
#   Why: it finds the answer even when the user's words differ from the document (another language included).
#   Done when: "How many days a week can I work from home?" finds the remote work rule (check 4).
def vector_search(client: QdrantClient, query: str, groups: list[str], k: int = K) -> list[dict]:
    """TODO 4: embed the query with the same model and return the top-k chunks (with "score")."""
    vector = embed([query])[0]
    points = client.query_points(COLLECTION, query=vector, limit=k, query_filter=access_filter(groups)).points
    return [{**p.payload, "score": p.score} for p in points]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# TODO 6 · Goal: classic keyword search (BM25) over the same chunks.
#   Why: embeddings miss exact codes and names like "BV-2013-02"; BM25 finds them at once.
#   Done when: "agreement BV-2013-02" returns the manual chunk 6.3 first (check 6).
class BM25Index:
    def __init__(self, chunks: list[dict]):
        """TODO 6: build BM25 over the tokenized texts (section path included, like for embeddings)."""
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(text_for_embedding(c)) for c in chunks])

    def search(self, query: str, groups: list[str], k: int = K) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [{**chunk, "score": float(score)} for chunk, score in ranked
                if score > 0 and allowed(chunk, groups)][:k]


# TODO 7 · Goal: merge the vector and BM25 rankings into one list.
#   Why: their scores are on different scales (0.36 vs 13.0), so we combine ranks, not scores.
#   Done when: a chunk found by both searches beats one found by a single search (check 7),
#   and hybrid recall@5 is higher than vector-only (check "TODO 6-7").
def rrf(rankings: list[list[dict]], k: int = K) -> list[dict]:
    """TODO 7: Reciprocal Rank Fusion: score = sum of 1 / (RRF_K + rank), rank starts at 1."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1 / (RRF_K + rank)
            by_id.setdefault(cid, chunk)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**by_id[cid], "score": scores[cid]} for cid in best]


# TODO 9 · Goal: re-order the top-20 hybrid candidates with a cross-encoder.
#   Why: it reads the query and the chunk together, so the right chunk moves to the top (higher MRR).
#   Done when: the right chunk goes from last to first place (check 9). Then compare MRR and ms/query in the table.
def rerank(query: str, candidates: list[dict], k: int = K) -> list[dict]:
    """TODO 9: score every (query, chunk) pair with the cross-encoder and keep the top-k."""
    # Hint: TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODELS)).rerank(query, texts) yields one score per text
    global _reranker
    if not candidates:
        return []
    if _reranker is None:
        _reranker = TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODELS))
    scores = list(_reranker.rerank(query, [text_for_embedding(c) for c in candidates]))
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [{**chunk, "score": float(score)} for chunk, score in ranked[:k]]


def make_configs(client: QdrantClient, bm25: BM25Index) -> dict:
    return {
        "vector": lambda q, g: vector_search(client, q, g),
        "bm25": lambda q, g: bm25.search(q, g),
        "hybrid": lambda q, g: rrf([vector_search(client, q, g, CANDIDATES), bm25.search(q, g, CANDIDATES)]),
        "hybrid+rerank": lambda q, g: rerank(q, rrf([vector_search(client, q, g, CANDIDATES),
                                                     bm25.search(q, g, CANDIDATES)], CANDIDATES)),
    }


# ---------------------------------------------------------------- evaluation

# TODO 5 · Goal: measure the search with numbers instead of "looks fine".
#   Why: without recall@k and MRR we cannot tell whether hybrid or the reranker actually helps.
#   Done when: the fake test gives recall 0.67 and MRR 0.50 (check 5) and the table shows real numbers per config.
def evaluate(search, questions: list[dict], k: int = K) -> dict:
    """TODO 5: recall@k (a relevant chunk is in the top-k) and MRR (1 / rank of the first relevant chunk)."""
    questions = [q for q in questions if q["relevant"]]   # skip questions without a relevant chunk
    hits, rr, misses = 0, 0.0, []
    found, total = 0, 0    # relevant chunks in the top-k / all relevant chunks listed in testset.json
    start = time.perf_counter()
    for q in questions:
        ids = [h["chunk_id"] for h in search(q["question"], q["groups"])[:k]]
        rank = next((i for i, cid in enumerate(ids, 1) if cid in q["relevant"]), None)
        found += len(set(ids) & set(q["relevant"]))
        total += len(set(q["relevant"]))
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append(q["id"])
    ms = (time.perf_counter() - start) * 1000 / len(questions)
    return {"recall": hits / len(questions), "mrr": rr / len(questions), "misses": misses, "ms": ms,
            "relevant_found": found, "relevant_total": total, "chunk_recall": found / total}


def check_access(client: QdrantClient, bm25: BM25Index) -> bool:
    """A user without the purchasing group must never get the price list, whatever they ask;
    a user with the purchasing group must find it (so the test cannot pass on an empty result)."""
    q = "What is the price per 1000 pieces of the IRF540N?"
    hidden = vector_search(client, q, ["all"]) + bm25.search(q, ["all"])
    visible = vector_search(client, q, ["purchasing"]) + bm25.search(q, ["purchasing"])
    return (all("purchasing" not in h["access"] for h in hidden)
            and any(h["source"].startswith("purchasing/") for h in visible))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))




# ---------------------------------------------------------------- main

def show(configs: dict, query: str, groups: list[str]):
    for name, search in configs.items():
        print(f"\n== {name}")
        for h in search(query, groups):
            print(f"  {h['score']:8.4f}  {h['chunk_id']:<50} {h['section'][:45]}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?")
    parser.add_argument("--groups", default="all")
    args = parser.parse_args(argv)

    chunks = load_chunks()
    client = build_index(chunks)
    bm25 = BM25Index(chunks)
    configs = make_configs(client, bm25)
    print(f"chunks {len(chunks)} | embedded now {embed.last_missing} (rest from cache) | model {EMBED_MODEL}")

    if args.query:
        show(configs, args.query, args.groups.split(","))
        return

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    print(f"\n{'config':<15}{'recall@' + str(K):>10}{'MRR':>8}{'ms/query':>10}{'rel found':>11}   misses")
    results = {}
    for name, search in configs.items():
        r = evaluate(search, questions)
        results[name] = r
        print(f"{name:<15}{r['recall']:>10.2f}{r['mrr']:>8.2f}{r['ms']:>10.0f}{str(r['relevant_found']) + '/' + str(r['relevant_total']):>11}   {' '.join(r['misses']) or '-'}")
    access_ok = check_access(client, bm25)
    print("\naccess filter: " + ("OK, purchasing chunk hidden from group all, found for purchasing" if access_ok
                                   else "FAIL, purchasing chunk visible to group all or not found for purchasing (TODO 8)"))
    line = " | ".join(f"{n} {r['recall']:.2f}/{r['mrr']:.2f}" for n, r in results.items())
    print(f"\nFOR CHAT: recall@{K}/MRR on {len(questions)} questions | {line}")
    (OUT / "eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotImplementedError as e:
        sys.exit(f"Not implemented yet: {e}")
