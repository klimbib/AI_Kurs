"""Evals for the RAG bot.

Usage:
    python evals.py                          # run the bot on the testset with judge + groundedness, write
                                             # eval/evals.json, print the comparison with baseline.json, gate (TODO 7):
                                             # green -> baseline is overwritten, red -> exit code 1, baseline stays
    python evals.py --prompt v1              # the same for another prompt folder prompts/<version>/
    python evals.py --judge-prompt judge-v1  # use another judge prompt folder prompts/<name>/
    python evals.py --calibrate              # judge against human labels (eval/labels.json), write
                                             # eval/calibration_evals.json, print the agreement table
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import rag  # noqa: E402

TESTSET = ROOT / "evals" / "testset.json"
LABELS = ROOT / "evals" / "labels.json"
BASELINE = ROOT / "baseline.json"
OUT = ROOT / "output"
KINDS_MIN = 2          # every kind of question needs at least this many cases
TOLERANCE = 0.05       # the gate allows a metric to drop this much (one case out of 37 is 0.03)
LOWER_IS_BETTER = {"false_abstain"}   # for these metrics a rise is the regression
JUDGE = "judge-v2"     # judge prompt folder prompts/<JUDGE>/ (only judge-v1 exists at the moment)


def load_testset() -> list[dict]:
    return json.loads(TESTSET.read_text(encoding="utf-8"))


def chunk_texts() -> dict:
    return {c["chunk_id"]: c["text"] for c in rag.search.load_chunks()}


def norm(text: str) -> str:
    """Lower case, one kind of minus and space: '−15 °C' and '-15  °c' compare equal. Ready to use."""
    text = text.lower().replace("−", "-").replace("–", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text)





# ---------------------------------------------------------------- deterministic scores

def is_unanswerable(case: dict) -> bool:
    """No relevant chunk (no_answer, or access case the user may not read): the correct result is NOT_FOUND."""
    return case["kind"] == "no_answer" or not case.get("relevant")


def score(result: dict, case: dict) -> dict:
    """TODO 2: deterministic scores for one answer."""
    if is_unanswerable(case):
        abstain = not result["found"]
        return {"facts": None, "cited": None, "abstain": abstain, "pass": abstain}
    answer = norm(result["answer"])
    facts = bool(result["found"]) and all(any(norm(opt) in answer for opt in group)
                                          for group in case.get("must", []))
    cited = bool(set(result.get("cited", [])) & set(case.get("relevant", [])))
    return {"facts": facts, "cited": cited, "abstain": None, "pass": facts and cited}


def rate(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(scores: list[dict], cases: list[dict], results: list[dict]) -> dict:
    """Metrics over one run. Ready to use."""
    answerable = [r for r, c in zip(results, cases) if not is_unanswerable(c)]
    m = {"pass": rate([s["pass"] for s in scores]), "facts": rate([s["facts"] for s in scores]),
         "cited": rate([s["cited"] for s in scores]), "abstain": rate([s["abstain"] for s in scores]),
         "false_abstain": rate([not r["found"] for r in answerable])}
    for key in ("judge", "grounded"):
        if any(key in s for s in scores):
            m[key] = rate([s.get(key) for s in scores])
    kinds = sorted({c["kind"] for c in cases})
    m["by_kind"] = {k: rate([s["pass"] for s, c in zip(scores, cases) if c["kind"] == k]) for k in kinds}
    m["failed"] = [c["id"] for s, c in zip(scores, cases) if not s["pass"]]
    return m




# ---------------------------------------------------------------- LLM as a judge

class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str                                   # reason first: the model explains before it decides
    verdict: Literal["correct", "incorrect"]


VERDICT_FORMAT = {"type": "json_schema",
                  "json_schema": {"name": "verdict", "strict": True, "schema": Verdict.model_json_schema()}}


# TODO 4 · Goal: let a model grade what string matching cannot: "No, upgrades are not reimbursed" vs "No".
#   Why: `must` lists break on yes/no questions, paraphrases and other languages. A judge compares meaning.
#   Done when: judge sends prompts/<JUDGE>/correct.txt as the system prompt, the question, the reference and the
#   answer in the user message, calls llm.chat with schema=VERDICT_FORMAT and returns a Verdict (check 4).
def judge(llm, case: dict, answer: str) -> Verdict:
    """TODO 4: one judge call for one answer."""
    user = (f"Question: {case['question']}" + chr(10)*2 + f"Reference answer: {case.get('reference', '')}" + chr(10)*2
            + f"Answer to grade: {answer}")
    messages = [{"role": "system", "content": (rag.PROMPTS / JUDGE / "correct.txt").read_text(encoding="utf-8").strip()},
                {"role": "user", "content": user}]
    return Verdict.model_validate_json(llm.chat(messages, schema=VERDICT_FORMAT, max_tokens=300))


# TODO 5 · Goal: measure the judge before you trust it.
#   Why: the judge is a model too. If it calls wrong answers correct, every number built on it is inflated.
#   Compare it with labels people set by hand and read every disagreement.
#   Done when: calibrate returns {"agreement", "false_pass", "false_fail", "disagreements"}: agreement = share of
#   items where judge == person, false_pass = share of items the judge passed but a person failed,
#   false_fail = the opposite, disagreements = ids of the items where they differ (check 5).
def calibrate(judged: list[bool], human: list[bool], ids: list[str]) -> dict:
    """TODO 5: judged[i] and human[i] are True when answer i was marked correct."""
    n = len(ids) or 1
    return {"agreement": sum(j == h for j, h in zip(judged, human)) / n,
            "false_pass": sum(j and not h for j, h in zip(judged, human)) / n,
            "false_fail": sum(h and not j for j, h in zip(judged, human)) / n,
            "disagreements": [i for i, j, h in zip(ids, judged, human) if j != h]}


# ---------------------------------------------------------------- groundedness

class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    supported: bool


class Claims(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[Claim]


CLAIMS_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "claims", "strict": True, "schema": Claims.model_json_schema()}}


# TODO 6 · Goal: catch answers that say more than their sources.
#   Why: an answer with a valid citation can still add a number or a rule the source never had.
#   Correctness compares with the reference; groundedness compares with the sources, so it also works
#   where nobody wrote a reference, for example on production logs.
#   Done when: grounded returns (None, []) for a not-found answer without a model call; otherwise it sends
#   prompts/<JUDGE>/grounded.txt as the system prompt, the sources and the answer, parses Claims and returns
#   (all claims supported, [texts of the unsupported claims]) (check 6).
def grounded(llm, answer: str, found: bool, sources: list[str], question: str = "") -> tuple[bool | None, list[str]]:
    """TODO 6: sources = texts of the chunks the answer cites; question (optional) helps to read short answers."""
    if not found:
        return None, []
    nl = chr(10) * 2
    user = "Sources:" + nl + nl.join(f"[{i + 1}] {t}" for i, t in enumerate(sources)) + nl
    if question:
        user += f"Question: {question}" + nl
    user += f"Answer: {answer}"
    messages = [{"role": "system", "content": (rag.PROMPTS / JUDGE / "grounded.txt").read_text(encoding="utf-8").strip()},
                {"role": "user", "content": user}]
    claims = Claims.model_validate_json(llm.chat(messages, schema=CLAIMS_FORMAT, max_tokens=2000)).claims
    bad = [c.claim for c in claims if not c.supported]
    return not bad, bad


# ---------------------------------------------------------------- the gate

# TODO 7 · Goal: turn the eval into a check that can block a release.
#   Why: evals nobody runs catch nothing. In CI the job runs on every change of a prompt or a model
#   and fails the build when quality drops, before users see it.
#   Done when: gate returns a list of failures: every number in the baseline (pass, facts, cited, abstain,
#   grounded ...) that is now lower by more than TOLERANCE (for LOWER_IS_BETTER metrics: higher), plus every case that passed in the baseline and fails
#   now, written as "regression: q16". An empty list means the gate is green (check 7).
def gate(metrics: dict, baseline: dict) -> list[str]:
    """TODO 7: compare the current metrics with the baseline."""
    now, old = flat_metrics(metrics), flat_metrics(baseline)
    failures = []
    for k, b in old.items():
        v = now.get(k)
        if v is None or b is None:
            continue
        if (v - b > TOLERANCE) if k in LOWER_IS_BETTER else (b - v > TOLERANCE):
            failures.append(f"{k}: {b:.2f} -> {v:.2f}")
    failed_before = set(baseline.get("failed", []))
    failures += [f"regression: {i}" for i in metrics.get("failed", []) if i not in failed_before]
    return failures


def flat_metrics(d: dict) -> dict:
    return {**{k: v for k, v in d.items() if k in ("pass", "facts", "cited", "abstain", "false_abstain",
                                                    "judge", "grounded")},
            **{f"kind:{k}": v for k, v in d.get("by_kind", {}).items()}}


# ---------------------------------------------------------------- running everything (ready to use)

def compare(m: dict, base: dict) -> None:
    """Print baseline vs now for every metric (and every kind); 'worse' = beyond TOLERANCE."""
    now, old = flat_metrics(m), flat_metrics(base)
    print(f"{'metric':<22}{'baseline':>10}{'now':>8}{'diff':>8}")
    for k, v in now.items():
        b = old.get(k)
        if v is None or b is None:
            print(f"{k:<22}{'-' if b is None else f'{b:.2f}':>10}{'-' if v is None else f'{v:.2f}':>8}{'':>8}")
            continue
        d = v - b
        bad = d > TOLERANCE if k in LOWER_IS_BETTER else d < -TOLERANCE
        print(f"{k:<22}{b:>10.2f}{v:>8.2f}{d:>+8.2f}{'  worse' if bad else ''}")


def run_bot(llm, retriever, cases: list[dict], prompt: str) -> list[dict]:
    return [{"id": c["id"], **rag.rag(llm, retriever, c["question"], c["groups"], c.get("history"), prompt=prompt)}
            for c in cases]


def grounded_or_skip(llm, answer, found, sources, question=""):
    """grounded() once TODO 6 is done; until then (None, []) so that TODO 4 and 5 can run first."""
    try:
        return grounded(llm, answer, found, sources, question)
    except NotImplementedError:
        return None, []


def add_judges(judge_llm, results: list[dict], scores: list[dict], cases: list[dict], chunks: dict):
    for r, s, c in zip(results, scores, cases):
        v = judge(judge_llm, c, r["answer"])
        s["judge"], s["judge_reason"] = v.verdict == "correct", v.reason
        s["grounded"], s["unsupported"] = grounded_or_skip(judge_llm, r["answer"], r["found"],
                                                           [chunks[i] for i in r["cited"]], c["question"])


def write_report(prompt: str, m: dict, results, scores, cases, bot_usd, eval_usd) -> Path:
    """A markdown report for people: the numbers, then every case worth reading."""
    f = lambda v: "-" if v is None else f"{v:.2f}"
    lines = [f"# Eval report: prompt {prompt}, model {rag.MODEL}", "",
             f"{len(cases)} cases. Bot ${bot_usd:.4f}, judge ${eval_usd:.4f}.", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {f(m[k])} |" for k in ("pass", "facts", "cited", "abstain", "false_abstain", "judge",
                                                 "grounded", "pass_at_1", "pass_all") if m.get(k) is not None]
    if m.get("flaky") is not None:
        lines += ["", f"Flaky cases: {', '.join(m['flaky']) or 'none'}"]
    lines += ["", "## By kind", "", "| kind | pass |", "|---|---|"] + [f"| {k} | {f(v)} |" for k, v in m["by_kind"].items()]
    lines += ["", "## Cases to read", ""]
    for r, s, c in zip(results, scores, cases):
        if s["pass"] and s.get("judge") is not False and s.get("grounded") is not False:
            continue
        lines += [f"### {c['id']} ({c['kind']}): {c['question']}", "", f"- answer: {r['answer']}",
                  f"- reference: {c['reference']}", f"- deterministic: facts {s['facts']}, cited {s['cited']}, "
                  f"abstain {s['abstain']}"]
        if "judge" in s:
            lines.append(f"- judge: {'correct' if s['judge'] else 'incorrect'}. {s['judge_reason']}")
        if s.get("unsupported"):
            lines.append(f"- not in the cited sources: {'; '.join(s['unsupported'])}")
        lines.append("")
    path = OUT / f"report-{prompt}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_calibration(judge_llm, chunks: dict):
    items = json.loads(LABELS.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in load_testset()}
    judged, ground = [], []
    for it in items:
        v = judge(judge_llm, cases[it["case"]], it["answer"])
        judged.append(v.verdict == "correct")
        g, bad = grounded_or_skip(judge_llm, it["answer"], it["found"], [chunks[i] for i in it["cited"]],
                               cases[it["case"]]["question"])
        ground.append(g)
        it.update(judge_reason=v.reason, judge_ok=judged[-1], grounded_model=g, unsupported=bad)
    ids, human = [it["id"] for it in items], [it["correct"] for it in items]
    det = [score(it, cases[it["case"]])["pass"] for it in items]
    res = {"deterministic": calibrate(det, human, ids), "judge": calibrate(judged, human, ids)}
    g_items = [(g, it) for g, it in zip(ground, items) if g is not None]
    if g_items:   # empty until TODO 6 is done
        res["grounded"] = calibrate([g for g, _ in g_items], [it["grounded"] for _, it in g_items],
                                    [it["id"] for _, it in g_items])
    print(f"{'':<24}{'agreement':>10}{'false pass':>12}{'false fail':>12}")
    for name, c in res.items():
        print(f"{name:<24}{c['agreement']:>10.2f}{c['false_pass']:>12.2f}{c['false_fail']:>12.2f}")
    LABELS.with_name("calibration_evals.json").write_text(
        json.dumps({"model": rag.MODEL, "prompts": JUDGE, **res, "items": items}, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None):
    global JUDGE
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=rag.PROMPT_VERSION)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--save-baseline", action="store_true")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--judge-prompt", default=JUDGE, help="judge prompt folder under prompts/, e.g. judge-v1")
    args = p.parse_args(argv)

    JUDGE = args.judge_prompt
    cases, chunks = load_testset(), chunk_texts()
    if args.calibrate:
        return run_calibration(rag.LLM(), chunks)

    bot, judge_llm, retriever = rag.LLM(), rag.LLM(), rag.Retriever()
    results = run_bot(bot, retriever, cases, args.prompt)
    scores = [score(r, c) for r, c in zip(results, cases)]
    add_judges(judge_llm, results, scores, cases, chunks)   # judge/grounded are part of the baseline
    m = aggregate(scores, cases, results)

    # small result file: metrics only, no answers, no report
    (ROOT / "evals" / "evals.json").write_text(json.dumps(m, indent=2))

    # terminal: only the comparison with the baseline
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    compare(m, base)
    failures = gate(m, base)
    if not failures:
        BASELINE.write_text(json.dumps(m, indent=2))
        print("\nGATE PASSED -> baseline.json updated")
    else:
        print("\nGATE FAILED -> baseline.json unchanged")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        raise SystemExit(f"{e} is not done yet: read the comment above the function")
