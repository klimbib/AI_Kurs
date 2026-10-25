import os, time, json, glob
from enum import Enum
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()  # бере OPENROUTER_API_KEY з файлу .env

PROMPT_FILE = "prompts/extract_V2.txt"
PROMPT = open(PROMPT_FILE, encoding="utf-8").read()
OUT_DIR = "results/V4"
CONFIG = json.load(open("config.json", encoding="utf-8"))["sampling"]


def sampling(model):
    """Default temperature/top_p from config.json, overridden per model."""
    return {**CONFIG["default"], **CONFIG["models"].get(model, {})}

PAGES = 2  # граничні параметри зазвичай на перших сторінках
PDF_READER = "pymupdf"  # "pymupdf" or "unstructured"
UNSTRUCTURED_STRATEGY = "fast"  # "fast" (text only), "hi_res" (layout/tables, needs poppler+tesseract)

TARGETS = [
    # назва, base_url, ключ, модель
    ("OpenRouter", "https://openrouter.ai/api/v1",
     os.getenv("OPENROUTER_API_KEY"), "openai/gpt-4o-mini",60),
    ("OpenRouter", "https://openrouter.ai/api/v1",
     os.getenv("OPENROUTER_API_KEY"), "minimax/minimax-m3.1-flash", 60),
   # ("Ollama", "http://localhost:11434/v1",
   # "ollama", "llama3.2:3b",180),
]

# USD per 1M tokens (input, output); local models are free
PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "minimax/minimax-m3.1-flash": (0.09, 0.50),
    "llama3.2:3b": (0.0, 0.0),
}
SHEETS = 1000  # normalisation target for the cost estimate


class Voltage(BaseModel):
    value_v: float = Field(ge=5, le=200)


class Current(BaseModel):
    value_a: float = Field(ge=0.01, le=100)


class Component(str, Enum):
    transistor = "transistor"
    diode = "diode"


class Part(BaseModel):
    component: Component
    max_temperature_c: float = Field(ge=125, le=200)
    max_voltage:  Voltage
    max_current: Current


def errors_text(e):
    return "; ".join(f"{'.'.join(map(str, x['loc'])) or 'JSON'}: {x['msg']}"
                     for x in e.errors())


def pdf_text_pymupdf(path):
    doc = pymupdf.open(path)
    return "\n".join(doc[i].get_text() for i in range(min(PAGES, len(doc))))


def pdf_text_unstructured(path):
    from unstructured.partition.pdf import partition_pdf  # imported lazily: heavy, optional
    elements = partition_pdf(filename=path, strategy=UNSTRUCTURED_STRATEGY,
                             infer_table_structure=UNSTRUCTURED_STRATEGY == "hi_res")
    return "\n".join(el.text for el in elements
                     if el.text and (el.metadata.page_number or 1) <= PAGES)


def pdf_text(path):
    if PDF_READER == "unstructured":
        return pdf_text_unstructured(path)
    return pdf_text_pymupdf(path)


def norm_str(v):
    return None if v is None else "".join(str(v).lower().split())


def norm_part(v):
    if v is None:
        return None
    v = "".join(c for c in str(v).upper() if c.isalnum())
    return v[:-3] if v.endswith("PBF") else v


def same_num(a, b):
    if a is None or b is None:
        return a is b
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def get(d, path):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


# категорія -> функція порівняння
CATEGORIES = {
    "part_number": lambda a, b: norm_part(a) == norm_part(b),
    "component": lambda a, b: norm_str(a) == norm_str(b),
    "type": lambda a, b: norm_str(a) == norm_str(b),
    "max_temperature_c": same_num,
    "max_voltage.value_v": same_num,
    "max_voltage.symbol": lambda a, b: norm_str(a) == norm_str(b),
    "max_voltage.temperature_c": same_num,
    "max_current.value_a": same_num,
    "max_current.symbol": lambda a, b: norm_str(a) == norm_str(b),
    "max_current.temperature_c": same_num,
}


def evaluate(entries, golden):
    """Порівнює результати з golden.json по категоріях (лише файли з golden)."""
    gold = {g["file"]: g for g in golden}
    stats = {c: [0, 0] for c in CATEGORIES}  # [правильно, всього]
    diffs, skipped, statuses = [], 0, {}
    for e in entries:
        g = gold.get(e["file"])
        if g is None:
            skipped += 1
            continue
        statuses[e.get("status", "error")] = statuses.get(e.get("status", "error"), 0) + 1
        for cat, cmp in CATEGORIES.items():
            got, exp = get(e, cat), get(g, cat)
            stats[cat][1] += 1
            if cmp(got, exp):
                stats[cat][0] += 1
            else:
                diffs.append(f"  {e['file']:<24} {cat:<28} expected={exp!r}  got={got!r}")
    return stats, diffs, skipped, statuses


def cost_lines(model, entries):
    """Token usage and cost per sheet, normalised to SHEETS sheets."""
    used = [e["usage"] for e in entries if "usage" in e]
    if not used:
        return ["Cost: no usage data", ""]
    n = len(used)
    tin = sum(u["prompt_tokens"] for u in used)
    tout = sum(u["completion_tokens"] for u in used)
    pin, pout = PRICES.get(model, (None, None))
    lines = [f"Tokens: {tin} in / {tout} out over {n} sheets "
             f"(avg {tin / n:.0f} in / {tout / n:.0f} out per sheet)"]
    if pin is None:
        lines.append("Cost: no price configured for this model")
    else:
        per_sheet = (tin * pin + tout * pout) / 1e6 / n
        lines.append(f"Cost per sheet: ${per_sheet:.5f}  | per {SHEETS} sheets: ${per_sheet * SHEETS:.2f}")
    return lines + [""]


def percentile(values, p):
    """Linear-interpolated percentile (p in 0..100) of a non-empty list."""
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def quality_lines(entries):
    """Share of ok / corrected / failed / error over all processed sheets (pydantic validation)."""
    n = len(entries)
    counts = {s: sum(1 for e in entries if e.get("status", "error") == s)
              for s in ("ok", "corrected", "failed", "error")}
    shares = "  ".join(f"{s} {c / n:.0%} ({c})" for s, c in counts.items())
    valid = counts["ok"] + counts["corrected"]
    lines = [f"Pydantic validation ({n} sheets): {shares}",
             f"Valid data (ok + corrected): {valid / n:.0%}"]
    lat = [e["latency_s"] for e in entries if "latency_s" in e]
    if lat:
        lines.append(f"Latency per sheet: p50 {percentile(lat, 50):.1f} s  | "
                     f"p95 {percentile(lat, 95):.1f} s")
    return lines


def write_log(path, all_stats, results):
    lines = [f"Prompt file: {PROMPT_FILE}", f"Golden file: data/golden.json", ""]
    for model, (stats, diffs, skipped, statuses) in all_stats.items():
        ok = sum(v[0] for v in stats.values())
        total = sum(v[1] for v in stats.values())
        lines += [f"=== Model: {model} ===",
                  f"Settings: temperature={sampling(model)['temperature']}  "
                  f"top_p={sampling(model)['top_p']}  prompt={PROMPT_FILE}  "
                  f"PDF_READER={PDF_READER}  UNSTRUCTURED_STRATEGY={UNSTRUCTURED_STRATEGY}",
                  *quality_lines(results[model]),
                  f"Status (golden only): {statuses}  | not in golden (skipped): {skipped}",
                  f"{'Category':<28} {'correct':>8} {'total':>7} {'Rate':>7}"]
        for cat, (a, n) in stats.items():
            lines.append(f"{cat:<28} {a:>8} {n:>7} {a / n:>7.0%}" if n else f"{cat:<28} -")
        lines += [f"{'TOTAL':<28} {ok:>8} {total:>7} {ok / total:>7.0%}" if total else "TOTAL -",
                  ""] + cost_lines(model, results[model]) + \
                 ["Deviations:"] + (diffs or ["  none"]) + [""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


files = sorted(glob.glob("data/*.pdf"))
results = {}

for name, base_url, key, model, timeout in TARGETS:
    if not key:
        print(f"{name}: немає OPENROUTER_API_KEY у .env")
        continue

    client = OpenAI(base_url=base_url, api_key=key,
                    timeout=timeout, max_retries=2)
    results[model] = []

    for path in files:
        start = time.perf_counter()
        messages = [{"role": "user", "content": PROMPT + pdf_text(path)}]
        status, content, err = "ok", None, None
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            for attempt in range(2):  # 0 = перша спроба, 1 = корекція
                r = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **sampling(model),
                    response_format={"type": "json_object"},
                )
                content = r.choices[0].message.content
                if r.usage:  # summed over the first try and the correction
                    usage["prompt_tokens"] += r.usage.prompt_tokens
                    usage["completion_tokens"] += r.usage.completion_tokens
                try:
                    Part.model_validate_json(content)
                    status = "ok" if attempt == 0 else "corrected"
                    err = None
                    break
                except ValidationError as e:
                    err = errors_text(e)
                    status = "failed"
                    messages += [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content":
                            f"Validation failed: {err}. Check the datasheet again "
                            "and answer ONLY with the corrected JSON object."},
                    ]
        except Exception as e:
            print(f"{name} | {path}: error  {type(e).__name__}: {e}")
            results[model].append({"file": os.path.basename(path), "error": str(e)})
            continue
        sec = time.perf_counter() - start

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"raw": content}
        entry = {"file": os.path.basename(path), "status": status, "usage": usage,
                 "latency_s": round(sec, 2), **data}
        if err:
            entry["validation_error"] = err
        results[model].append(entry)
        print(f"{name} | {model} | {sec:.1f} с | {os.path.basename(path)} | "
              f"{status.upper()} | {data.get('max_temperature_c')}, "
              f"{data.get('max_voltage')}, {data.get('max_current')}"
              + (f" | {err}" if err else ""))

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

golden = json.load(open("data/golden.json", encoding="utf-8"))
write_log(f"{OUT_DIR}/log.txt", {m: evaluate(r, golden) for m, r in results.items()}, results)
print(f"Результати та лог збережено в {OUT_DIR}/")
