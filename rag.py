
"""

Usage:
    python rag.py                         # answer all questions in eval/testset.json and score them
    python rag.py "what component are available?"     # one question, answer with sources
    python rag.py --groups pursaching "what is the cost of the component?"  # one question, answer with sources


"""





import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import search  # noqa: E402

load_dotenv(ROOT / ".env")

QUESTIONS = ROOT / "evals" / "testset.json"

PROMPTS = ROOT / "prompts"
OUT = ROOT / "output" / "rag"

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.15, 0.60          # USD per 1M tokens, check openrouter.ai/models before you rely on it
# Local model instead (lesson 3): ollama pull llama3.2:3b
# BASE_URL, API_KEY, MODEL, PRICE_IN, PRICE_OUT = "http://localhost:11434/v1", "ollama", "llama3.2:3b", 0, 0

K = 5                      # how many chunks retrieval returns
HISTORY_TURNS = 3          # last turns of a dialogue that condense() sees
CONTEXT_BUDGET = 1500      # max tokens of sources we put into the prompt


METHOD = "hybrid+rerank"  # search method used for rag. option "vector", "bm25", "hybrid", "hybrid+rerank"
NOT_FOUND = "I could not find this in the documents available to you."

ENC = tiktoken.get_encoding("o200k_base")   # the tokenizer of gpt-4o / gpt-4o-mini


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    found: bool             # false when the sources do not contain the answer
    answer: str
    citations: list[str]    # source labels used in the answer, e.g. ["S1", "S3"]


ANSWER_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "answer", "strict": True, "schema": Answer.model_json_schema()}}


class LLM:
    """A thin wrapper over the chat API: one call, its tokens, cost and time. Ready to use."""

    def __init__(self):
        if not API_KEY:
            raise SystemExit("OPENROUTER_API_KEY is missing: copy .env.example to .env and paste your key")
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60, max_retries=2)
        self.calls, self.tokens_in, self.tokens_out = 0, 0, 0

    def chat(self, messages: list[dict], schema: dict | None = None, max_tokens: int = 500) -> str:
        r = self.client.chat.completions.create(model=MODEL, messages=messages, temperature=0,
                                                max_tokens=max_tokens, **({"response_format": schema} if schema else {}))
        self.calls += 1
        if r.usage:
            self.tokens_in += r.usage.prompt_tokens
            self.tokens_out += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    def cost(self) -> float:
        return (self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT) / 1_000_000


PROMPT_VERSION = "v1" # prompt folder prompts/<version>/ used by default (evals.py --prompt <version>)


def load_prompt(name: str = "system", version: str = PROMPT_VERSION) -> str:
    path = PROMPTS / version / f"{name}.txt"
    if not path.exists():
        raise SystemExit(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------- context

# TODO 1 · Goal: turn the retrieved chunks into numbered sources that fit a token budget.
#   Why: the model can only cite what has a label, and every token of context costs money and latency.
#   Done when: sources are labelled S1, S2, ... with document, section and date, and the total stays
#   within the budget: a chunk that does not fit is skipped, a smaller one after it may still go in (check 1).
def build_context(hits: list[dict], budget: int = CONTEXT_BUDGET) -> tuple[str, dict]:
    """Return the context text and {label: chunk} for the chunks that fit into `budget` tokens."""
    blocks, sources, used = [], {}, 0
    for hit in hits:
        label = f"S{len(sources) + 1}"
        date = hit.get("effective") or (hit.get("ingested_at") or "")[:10] or "n/a"
        block = (f"[{label}] document: {hit.get('title') or hit['source']} ({hit['source']}) | "
                 f"section: {hit.get('section') or '-'} | date: {date}\n{hit['text']}")
        cost = count_tokens(block)
        if used + cost > budget:
            continue            # does not fit: skip it, a smaller chunk later may still go in
        blocks.append(block)
        sources[label] = hit
        used += cost
    return "\n\n".join(blocks), sources


# ---------------------------------------------------------------- answer

# TODO 2 (ready: read it, no code to write) · Goal: ask the model to answer only from the sources and return JSON {found, answer, citations}.
#   Why: structured output (lesson 4) lets the code check citations and "not found" instead of parsing prose.
#   Done when: the user message holds the sources and the question, the call uses ANSWER_FORMAT,
#   and the reply is parsed into Answer (check 2).
def generate(llm: LLM, question: str, context: str, system: str) -> Answer:
    """TODO 2: build messages (system prompt + sources + question), call llm.chat with ANSWER_FORMAT, parse."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"<sources>\n{context}\n</sources>\n\nQuestion: {question}"}]
    return Answer.model_validate_json(llm.chat(messages, schema=ANSWER_FORMAT))


# TODO 3 · Goal: keep only citations that point to sources we actually gave the model.
#   Why: a model can invent "S7" when there were five sources, write "[S3]" instead of "S3",
#   or answer without citing anything. An answer that cites nothing real is not grounded, so it becomes "not found".
#   Done when: "[S1]" counts as S1, unknown labels are dropped,
#   and found=True with no valid citation becomes found=False (check 3).
def validate_citations(ans: Answer, sources: dict) -> tuple[Answer, list[str]]:
    """Normalize labels ("[S1]" -> "S1"), return the cleaned answer and the dropped (invalid) labels."""
    valid, dropped = [], []
    for raw in ans.citations:
        m = re.search(r"S\s*(\d+)", raw, re.IGNORECASE)
        label = f"S{m.group(1)}" if m else raw
        if label in sources:
            if label not in valid:
                valid.append(label)
        else:
            dropped.append(raw)
    found = ans.found and bool(valid)
    return Answer(found=found, answer=ans.answer if found else "", citations=valid if found else []), dropped


# TODO 9 · Goal: rewrite a follow-up ("what is their e-mail?") into a question that stands on its own.
#   Why: retrieval sees only the current message; "their" finds nothing without the previous turn.
#   Done when: with history the model rewrites the question before the search, without history
#   there is no extra call (check 9).
def condense(llm: LLM, history: list[list[str]], question: str) -> str:
    """One LLM call that returns the standalone question; no history -> no call."""
    if not history:
        return question
    turns = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])
    messages = [{"role": "system", "content": load_prompt("condense", PROMPT_VERSION)},
                {"role": "user", "content": f"History:\n{turns}\n\nLast question: {question}"}]
    return llm.chat(messages, max_tokens=100).strip() or question


# TODO 4 · Goal: put the steps together and say "not found" instead of guessing.
#   Why: when the documents do not contain the answer, a made-up answer is worse than no answer.
#   Done when: no hits -> NOT_FOUND without calling the model; found=False -> NOT_FOUND and no sources;
#   otherwise the answer with the cited chunks (check 4).
def rag(llm: LLM, retriever, question: str, groups: list[str], history: list | None = None,
        k: int = K, budget: int = CONTEXT_BUDGET, prompt: str = PROMPT_VERSION) -> dict:
    """condense -> retrieve -> build_context -> generate -> validate_citations -> NOT_FOUND if needed."""
    standalone = condense(llm, history or [], question)
    hits = retriever.search(standalone, groups, k)
    result = {"question": question, "standalone": standalone, "found": False, "answer": NOT_FOUND,
              "citations": [], "cited": [], "sources": [], "dropped": [], "retrieved": [h["chunk_id"] for h in hits]}
    if not hits:
        return result
    context, sources = build_context(hits, budget)
    if not sources:
        return result
    ans, dropped = validate_citations(generate(llm, standalone, context, load_prompt("system", prompt)), sources)
    result["dropped"] = dropped
    if ans.found:
        result.update(found=True, answer=ans.answer, citations=ans.citations,
                      cited=[sources[c]["chunk_id"] for c in ans.citations],
                      sources=[{"label": c, "chunk_id": sources[c]["chunk_id"], "source": sources[c]["source"],
                                "section": sources[c]["section"]} for c in ans.citations])
    return result


class Retriever:
    """Search of search.py (with the access filter on groups), using the method set in METHOD."""

    def __init__(self):
        chunks = search.load_chunks()
        self.client = search.build_index(chunks)
        self.bm25 = search.BM25Index(chunks)
        self.configs = search.make_configs(self.client, self.bm25)   # vector | bm25 | hybrid | hybrid+rerank

    def search(self, query: str, groups: list[str], k: int = K) -> list[dict]:
        return self.configs[METHOD](query, groups)[:k]


def print_result(r: dict) -> None:
    print(f"\nQ: {r['question']}")
    if r["standalone"] != r["question"]:
        print(f"   (searched as: {r['standalone']})")
    print(f"A: {r['answer']}")
    for s in r["sources"]:
        print(f"   [{s['label']}] {s['source']} | {s['section']}")


def run_questions(retriever) -> list[dict]:
    """Answer every question of eval/testset.json with the search method METHOD, score it
    and write output/rag/answers.json."""
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    llm, results, answered = LLM(), [], 0
    start = time.perf_counter()
    for q in questions:
        r = rag(llm, retriever, q["question"], q["groups"])
        r["id"], r["relevant"] = q["id"], q["relevant"]
        r["hit"] = bool(set(r["retrieved"]) & set(q["relevant"]))     # retrieval found a relevant chunk
        answered += r["found"]
        results.append(r)
        mark = "  [relevant=[]]" if not q["relevant"] else ""     # unanswerable question: NOT_FOUND is correct
        print(f"{q['id']}  found={str(r['found']):<5} retrieval_hit={str(r['hit']):<5} {r['answer'][:80]!r}{mark}")
    summary = {"method": METHOD, "model": MODEL, "questions": len(results),
               "answered": answered, "retrieval_hit": sum(r["hit"] for r in results), "llm_calls": llm.calls,
               "tokens_in": llm.tokens_in, "tokens_out": llm.tokens_out, "cost_usd": round(llm.cost(), 5),
               "seconds": round(time.perf_counter() - start, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answers.json").write_text(json.dumps({"summary": summary, "results": results},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmethod {METHOD} | answered {answered}/{len(results)} | retrieval hit "
          f"{summary['retrieval_hit']}/{len(results)} | cost ${summary['cost_usd']} | written to {OUT / 'answers.json'}")
    return results


def chat(llm: LLM, retriever, first: str, groups: list[str]) -> None:
    """Dialogue: answer the first question, then keep asking until an exit word, an empty line or Ctrl+C / Ctrl+D.
    The last 3 turns are the history, so a follow-up is rewritten into a standalone question by condense()."""
    history: list[list[str]] = []
    question = first
    while True:
        r = rag(llm, retriever, question, groups, history)
        print_result(r)
        history = (history + [[question, r["answer"]]])[-HISTORY_TURNS:]
        try:
            question = input("\n> next question (empty line or 'exit' to finish): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in ("", "exit", "quit", "q"):
            return


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--groups", default="all", help="comma separated access groups for a question")
    args = ap.parse_args(argv)
    retriever = Retriever()
    if args.query:
        chat(LLM(), retriever, args.query, args.groups.split(","))      # one question starts a dialogue
    else:
        run_questions(retriever)                                         # testset.json: no dialogue


if __name__ == "__main__":
    main()
