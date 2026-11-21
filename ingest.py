""".

    python starter/ingest.py             # ingest data/raw -> output/
    python starter/ingest.py --full      # ignore the manifest and re-ingest everything
    python starter/check.py              # check the output against data/expected.json

Pipeline: extract text -> normalize -> deduplicate -> chunk -> attach metadata -> manifest.

"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from charset_normalizer import from_bytes
import pymupdf
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data"
ACCESS = ROOT / "data" / "access.json"
OUT = ROOT / "output" / "ingest"

PIPELINE_VERSION = "ingest-v1"
CHUNK_TOKENS = 120          # max tokens per chunk (small, because the demo documents are short)
FIXED_OVERLAP = 20          # overlap for the fixed-size baseline
NEAR_DUP_JACCARD = 0.8      # shingle overlap above which two documents are near-duplicates
MIN_CHARS_PER_PAGE = 50     # a PDF page with less text than this has no usable text layer
GARBLED_RATIO = 0.2         # share of control characters / glyph names above which a PDF is unreadable
PAGE_NUMBER = (1,)        # 1-based pages to read
READ_TABLE = False          # True: keep tables in PDF text; False: leave them out
CHUNK_FIXED = False         # True: write fixed-size token chunks; False: write chunks by pages

ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


@dataclass
class Doc:
    source: str                       # path relative to data/raw, e.g. "it/vpn_setup.docx"
    raw_hash: str                     # sha256 of the file bytes: did the file change?
    text: str = ""
    title: str = ""
    effective: str | None = None      # "Effective YYYY-MM-DD" found in the document
    status: str = "ok"                # ok | ocr_required | exact_duplicate | near_duplicate | error
    note: str = ""
    doc_hash: str = ""                # sha256 of the normalized text: is the content the same?
    unchanged: bool = False           # same file as in the last run: reuse its chunks
    chunks: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- 1. extract

def read_text_file(path: Path) -> str:
    """TODO 1: .txt / .md. Detect the encoding instead of assuming UTF-8."""
    # Hint: charset_normalizer.from_bytes(data).best(); strip the BOM ("\ufeff").
    data = path.read_bytes()
    best = from_bytes(data).best()
    text = str(best) if best is not None else data.decode("utf-8", errors="replace")
    text = text.lstrip("\ufeff")
    pages = text.split("\f")
    return "\f".join(p for i, p in enumerate(pages, start=1) if i in PAGE_NUMBER)


def _pdf_pages_without_tables(path: Path) -> list[str]:
    """Text of the selected pages with every detected table region left out."""
    pages = []
    with pymupdf.open(path) as doc:
        for number in sorted(PAGE_NUMBER):
            if not 1 <= number <= len(doc):
                continue
            page = doc[number - 1]
            tables = [pymupdf.Rect(t.bbox) for t in page.find_tables().tables]
            blocks = [b for b in page.get_text("blocks")
                      if b[6] == 0 and not any(pymupdf.Rect(b[:4]).intersects(t) for t in tables)]
            pages.append("\n".join(b[4].strip() for b in blocks))
    return pages


def extract_pdf(path: Path) -> tuple[str, bool]:
    """TODO 3: text per page, pages separated by \\f. Returns (text, needs_ocr).
    A scan has pages but (almost) no text layer: we flag it instead of indexing nothing.
    Only the pages in PAGE_NUMBER are read; tables are read only if READ_TABLE is True."""
    # Hint: compare the number of characters with MIN_CHARS_PER_PAGE * number of pages.
    if READ_TABLE:
        reader = PdfReader(path)
        pages = [page.extract_text() or ""
                 for i, page in enumerate(reader.pages, start=1) if i in PAGE_NUMBER]
    else:
        pages = _pdf_pages_without_tables(path)
    chars = sum(len(p.strip()) for p in pages)
    needs_ocr = len(pages) > 0 and (chars < MIN_CHARS_PER_PAGE * len(pages) or is_garbled("".join(pages)))
    return "\f".join(pages), needs_ocr


def is_garbled(text: str) -> bool:
    """A font without a character map gives control characters or raw glyph names (/G72/G69...)
    instead of text: it looks like a text layer but is unreadable, so it needs OCR."""
    text = text.strip()
    if not text:
        return False
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\f\r")
    glyph_names = sum(len(m) for m in re.findall(r"/G\d+", text))
    return (control + glyph_names) / len(text) > GARBLED_RATIO





def extract(path: Path) -> tuple[str, bool]:
    """Dispatch by file type. Returns (text, needs_ocr)."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return read_text_file(path), False
    if suffix == ".pdf":
        return extract_pdf(path)
    raise ValueError(f"unsupported file type: {suffix}")


# ---------------------------------------------------------------- 2. normalize

PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d+\s*(of|/)\s*\d+\s*$", re.IGNORECASE)
COPYRIGHT_LINE_RE = re.compile(r"^\s*(©|\(c\)|all rights reserved\.?\s*$)", re.IGNORECASE)
COPYRIGHT_INLINE_RE = re.compile(r"(©\s*)?(\d{4}\s+)?copyright\b.*?all rights reserved\.?\s*", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6} |\d+\.\s+[A-ZА-ЯІЇЄҐ])")


def is_heading(line: str) -> bool:
    """A markdown heading, or a short numbered title like "2. Eligibility" (a PDF has no styles).
    "1. Open the Software Center." is a list item, not a heading: it is a sentence."""
    line = line.strip()
    if line.startswith("#"):
        return bool(HEADING.match(line))
    return bool(HEADING.match(line)) and len(line) < 60 and not line.endswith((".", ":", ";"))




def unwrap(text: str) -> str:
    """Join lines that a PDF broke inside a sentence; keep headings, list items and table rows."""
    out: list[str] = []
    for line in text.split("\n"):
        prev = out[-1] if out else ""
        # A wrapped line is long and does not end a sentence; a short line is a title or a label.
        joinable = (prev and line and len(prev) >= 60 and not HEADING.match(line) and not HEADING.match(prev)
                    and not line.lstrip().startswith(("- ", "|")) and not prev.lstrip().startswith("|")
                    and not re.match(r"^\d+\.\s", line) and not prev.rstrip().endswith((".", ":", "!", "?")))
        if joinable:
            out[-1] = f"{prev.rstrip()} {line.lstrip()}"
        else:
            out.append(line)
    return "\n".join(out)


def normalize(text: str) -> str:
    """TODO 5: one canonical text form, so that hashing, dedup and chunking see the same thing."""
    # Steps, in this order:
    # 1. unicodedata.normalize("NFKC", ...), CRLF -> LF, remove zero-width characters and the BOM
    # 2. split pages on "\f", remove lines that repeat on most pages (headers, footers)
    # 3. remove page numbers ("Page 2 of 3") with PAGE_NUMBER_RE
    # 4. join words hyphenated across lines: "secur-\nity" -> "security"
    # 5. strip trailing spaces, collapse runs of spaces, unwrap(text), max one blank line
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub("[​‌‍⁠﻿]", "", text)

    pages = [[line.rstrip() for line in page.split("\n")] for page in text.split("\f")]
    if len(pages) > 1:
        counts: dict[str, int] = {}
        for page in pages:
            for line in {l.strip() for l in page if l.strip()}:
                counts[line] = counts.get(line, 0) + 1
        threshold = max(2, len(pages) / 2)
        repeated = {line for line, n in counts.items() if n >= threshold}
        pages = [[l for l in page if l.strip() not in repeated] for page in pages]

    out_pages = []
    for page in pages:          # pages stay separated by \f so that chunk_by_pages can use them
        lines = [COPYRIGHT_INLINE_RE.sub("", l) for l in page if not COPYRIGHT_LINE_RE.match(l)]
        lines = [l for l in lines if not PAGE_NUMBER_RE.match(l)]
        page_text = "\n".join(lines)
        page_text = re.sub(r"(\w)-\n([a-zа-яіїєґ])", r"\1\2", page_text)
        page_text = "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in page_text.split("\n"))
        page_text = unwrap(page_text)
        out_pages.append(re.sub(r"\n{3,}", "\n\n", page_text).strip())
    return "\f".join(out_pages)


# ---------------------------------------------------------------- 3. deduplicate

def content_hash(text: str) -> str:
    """TODO 6: the same content gives the same hash, whatever the line endings or spacing."""
    # Hint: hash a canonical form, not the raw string.
    canonical = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").casefold()
    canonical = re.sub(r"\s+", " ", canonical).strip()  # line breaks and spacing are not content
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def near_duplicates(docs: list[Doc]) -> list[tuple[Doc, Doc, float]]:
    """TODO 7: pairs of documents whose word 5-shingles overlap above NEAR_DUP_JACCARD.
    O(n^2) is fine for a folder; for millions of documents use MinHash + LSH."""
    # Hint: shingles() and jaccard() are ready; compare every pair once.
    sets = [shingles(d.text) for d in docs]
    pairs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            score = jaccard(sets[i], sets[j])
            if score > NEAR_DUP_JACCARD:
                pairs.append((docs[i], docs[j], score))
    return pairs


# ---------------------------------------------------------------- 4. chunk

def chunk_fixed(text: str, size: int = CHUNK_TOKENS, overlap: int = FIXED_OVERLAP) -> list[dict]:
    """Baseline: a sliding window of tokens. Simple, but cuts sentences, tables and sections."""
    tokens = ENC.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        piece = ENC.decode(tokens[start:start + size])
        chunks.append({"text": piece, "section": ""})
        if start + size >= len(tokens):
            break
        start += size - overlap
    return chunks


def chunk_by_pages(text: str, max_tokens: int = CHUNK_TOKENS) -> list[dict]:
    """One or more chunks per page (pages are separated by \\f); a chunk never spans two pages.
    Inside a page, whole paragraphs are packed up to max_tokens; a table is never cut in the middle."""
    page_labels = sorted(PAGE_NUMBER)
    chunks: list[dict] = []
    for index, page in enumerate(text.split("\f")):
        number = page_labels[index] if index < len(page_labels) else index + 1
        section = f"page {number}"
        current: list[str] = []
        after_heading = False       # a heading is never left alone: the next piece joins it
        for para in re.split(r"\n\s*\n", page):
            if not para.strip():
                continue
            for piece in split_paragraph(para.strip(), max_tokens):
                if (current and not after_heading
                        and n_tokens("\n\n".join(current + [piece])) > max_tokens):
                    chunks.append({"text": "\n\n".join(current), "section": section})
                    current = []
                current.append(piece)
                after_heading = is_heading(piece)
        if current:
            chunks.append({"text": "\n\n".join(current), "section": section})
    return chunks


def split_paragraph(para: str, max_tokens: int) -> list[str]:
    """A paragraph that fits stays whole; an oversized one is split on sentence ends. A table is never split."""
    if n_tokens(para) <= max_tokens or para.lstrip().startswith("|"):
        return [para]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", para):
        if current and n_tokens(f"{current} {sentence}") > max_tokens:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return pieces + [current] if current else pieces


def chunk_stats(chunks: list[dict]) -> dict:
    """How often a chunk starts mid-sentence or cuts a table: the cost of naive chunking."""
    mid = sum(1 for c in chunks if c["text"][:1].islower())
    cut_tables = sum(1 for c in chunks if any("|" in line and not (line.startswith("|") and line.endswith("|"))
                                              for line in c["text"].strip().split("\n")))
    sizes = [n_tokens(c["text"]) for c in chunks] or [0]
    return {"chunks": len(chunks), "starts_mid_sentence": mid, "cut_tables": cut_tables,
            "avg_tokens": round(sum(sizes) / len(sizes)), "max_tokens": max(sizes)}


# ---------------------------------------------------------------- 5. metadata

def load_access() -> dict[str, list[str]]:
    rules = json.loads(ACCESS.read_text(encoding="utf-8")) if ACCESS.exists() else {}
    return {k: v for k, v in rules.items() if not k.startswith("_")}


def access_for(source: str, rules: dict[str, list[str]]) -> list[str]:
    """An explicit access.json rule (longest non-empty path prefix) wins. Otherwise the directory decides:
    file directly in data/ -> rules[""] or ["all"]; file in a subfolder -> [name of the first-level subfolder]."""
    matches = [p for p in rules if p and source.startswith(p)]
    if matches:
        return rules[max(matches, key=len)]
    parts = source.split("/")
    return [parts[0]] if len(parts) > 1 else rules.get("", ["all"])


def title_of(doc: Doc) -> str:
    for line in doc.text.splitlines():
        if line.strip():
            return line.lstrip("#").strip()[:120]
    return Path(doc.source).stem


def pdf_title(path: Path) -> str:
    """A PDF often carries its real title in the metadata; the first line may be a stamp."""
    if path.suffix.lower() != ".pdf":
        return ""
    meta = PdfReader(path).metadata
    return (meta.title or "").strip() if meta else ""


def chunk_records(doc: Doc, chunks: list[dict], rules: dict[str, list[str]], now: str) -> list[dict]:
    """TODO 9: everything retrieval will need later: where it came from, who may see it,
    which version of the document and of the pipeline produced it."""
    # Fields: chunk_id, source, title, section, access (access_for), effective, doc_hash,
    # chunk_index, n_tokens, text, pipeline_version, ingested_at
    access = access_for(doc.source, rules)
    return [{"chunk_id": f"{doc.source}#{i}", "source": doc.source, "title": doc.title,
             "section": c.get("section", ""), "access": access, "effective": doc.effective,
             "doc_hash": doc.doc_hash, "chunk_index": i, "n_tokens": n_tokens(c["text"]), "text": c["text"],
             "pipeline_version": PIPELINE_VERSION, "ingested_at": now}
            for i, c in enumerate(chunks)]


# ---------------------------------------------------------------- 6. manifest, incremental runs, audit

def load_manifest() -> dict:
    path = OUT / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": {}}


def dataset_version(files: dict) -> str:
    """TODO 10: one id for the exact set of inputs + pipeline version: same inputs, same id."""
    # Hint: sha256 of json.dumps(..., sort_keys=True) over PIPELINE_VERSION and each file's raw_hash.
    payload = {"pipeline_version": PIPELINE_VERSION,
               "files": {source: f["raw_hash"] for source, f in files.items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def audit(event: str, source: str, **extra) -> None:
    """TODO 10: record what happened to each file, so any chunk can be traced back."""
    # One JSON line per event in OUT / "audit.jsonl": ts, event, source and the extra fields.
    record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, "source": source, **extra}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- run

def run(full: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rules = load_access()
    manifest = {"files": {}} if full else load_manifest()
    previous = manifest["files"]
    old_chunks = {}
    if not full and (OUT / "chunks.jsonl").exists():
        for line in (OUT / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            old_chunks.setdefault(c["source"], []).append(c)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    docs: list[Doc] = []
    skipped = 0
    for path in sorted(p for p in RAW.rglob("*") if p.is_file() and not p.name.startswith(".")
                       and p.suffix.lower() in (".txt", ".md", ".pdf")):
        source = path.relative_to(RAW).as_posix()
        doc = Doc(source=source, raw_hash=hashlib.sha256(path.read_bytes()).hexdigest())
        prev = previous.get(source)
        if prev and prev["raw_hash"] == doc.raw_hash and prev.get("pipeline_version") == PIPELINE_VERSION:
            doc.status, doc.note, doc.doc_hash = prev["status"], prev.get("note", ""), prev.get("doc_hash", "")
            doc.title, doc.effective = prev.get("title", ""), prev.get("effective")
            doc.chunks = old_chunks.get(source, [])
            doc.text = "\n".join(c["text"] for c in doc.chunks)
            doc.unchanged = True
            skipped += 1
            docs.append(doc)
            continue
        try:
            raw_text, needs_ocr = extract(path)
        except Exception as e:  # one broken file must not stop the batch
            doc.status, doc.note = "error", f"{type(e).__name__}: {e}"
            audit("error", source, error=doc.note)
            docs.append(doc)
            continue
        if needs_ocr:
            doc.status, doc.note = "ocr_required", "no text layer: send to OCR, not to the index"
            audit("ocr_required", source)
            docs.append(doc)
            continue
        m = re.search(r"Effective (\d{4}-\d{2}-\d{2})", raw_text)
        doc.effective = m.group(1) if m else None
        doc.text = normalize(raw_text)
        doc.doc_hash = content_hash(doc.text)
        doc.title = pdf_title(path) or title_of(doc)
        docs.append(doc)

    # exact duplicates: same normalized content -> keep the first path, drop the rest
    seen: dict[str, Doc] = {}
    for doc in docs:
        if doc.status != "ok" or doc.unchanged:
            if doc.status == "ok":
                seen.setdefault(doc.doc_hash, doc)
            continue
        if doc.doc_hash in seen:
            doc.status, doc.note = "exact_duplicate", f"same content as {seen[doc.doc_hash].source}"
            audit("exact_duplicate", doc.source, of=seen[doc.doc_hash].source)
        else:
            seen[doc.doc_hash] = doc

    # near duplicates: keep the newer version (by effective date), drop the older one
    active = [d for d in docs if d.status == "ok"]
    for a, b, score in near_duplicates(active):
        older, newer = sorted([a, b], key=lambda d: d.effective or "")
        older.status, older.note = "near_duplicate", f"older version of {newer.source} (jaccard {score:.2f})"
        older.chunks = []
        audit("near_duplicate", older.source, kept=newer.source, jaccard=round(score, 2))

    fixed_all, page_all = [], []
    for doc in docs:
        if doc.status != "ok" or doc.unchanged:
            continue
        fixed = chunk_fixed(doc.text.replace("\f", "\n\n"))
        pages = chunk_by_pages(doc.text)
        fixed_all += fixed
        page_all += pages
        chunks = fixed if CHUNK_FIXED else pages
        doc.chunks = chunk_records(doc, chunks, rules, now)
        audit("ingested", doc.source, chunks=len(doc.chunks), doc_hash=doc.doc_hash[:12])

    files = {d.source: {"raw_hash": d.raw_hash, "doc_hash": d.doc_hash, "status": d.status, "note": d.note,
                        "title": d.title, "effective": d.effective, "chunks": len(d.chunks),
                        "pipeline_version": PIPELINE_VERSION} for d in docs}
    version = dataset_version(files)
    (OUT / "manifest.json").write_text(json.dumps({"dataset_version": version, "pipeline_version": PIPELINE_VERSION,
                                                   "created_at": now, "files": files}, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    all_chunks = [c for d in docs if d.status == "ok" for c in d.chunks]
    with (OUT / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report = {"dataset_version": version, "files": len(docs), "skipped_unchanged": skipped,
              "docs": {d.source: {"status": d.status, "note": d.note, "chunks": len(d.chunks)} for d in docs},
              "chunking": {"fixed": chunk_stats(fixed_all), "by_pages": chunk_stats(page_all)}}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_report(report: dict) -> None:
    print(f"{'file':<40} {'status':<16} chunks  note")
    for source, d in report["docs"].items():
        print(f"{source:<40} {d['status']:<16} {d['chunks']:>6}  {d['note']}")
    fixed, pages = report["chunking"]["fixed"], report["chunking"]["by_pages"]
    if fixed["chunks"]:
        print("\nchunking            chunks  mid-sentence  cut tables  avg tok  max tok")
        for name, s in (("fixed 120/20", fixed), ("by pages", pages)):
            print(f"{name:<18} {s['chunks']:>7} {s['starts_mid_sentence']:>13} {s['cut_tables']:>11} "
                  f"{s['avg_tokens']:>8} {s['max_tokens']:>8}")
    ok = sum(1 for d in report["docs"].values() if d["status"] == "ok")
    n_chunks = sum(d["chunks"] for d in report["docs"].values())
    print(f"\nFOR CHAT: files {report['files']} | indexed {ok} | chunks {n_chunks} | "
          f"skipped unchanged {report['skipped_unchanged']} | dataset {report['dataset_version']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="ignore the manifest and re-ingest every file")
    args = ap.parse_args()
    if not RAW.exists():
        sys.exit(f"no input folder {RAW}")
    print_report(run(args.full))
