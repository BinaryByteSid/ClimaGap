"""Stage 1: turn every PDF/TXT in data/raw into paragraph-sized passages.

File naming: <Region>__<anything>.pdf|txt. data/sources.csv can override the
inferred metadata (doc_id, region, title, year, kind, url) for a file.
"""
import csv
import json
import re
from collections import Counter

from common import DATA, PASSAGES, PROCESSED, RAW, entity_names, slug, write_jsonl

MIN_CHARS, MAX_CHARS = 200, 1200
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def load_sources():
    path = DATA / "sources.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def metadata_for(path, sources):
    if path.name in sources:
        return sources[path.name]
    prefix = path.stem.split("__", 1)[0]
    region = next((r for r in entity_names("Region") if r.lower() == prefix.lower()), "")
    return {"doc_id": slug(path.stem), "region": region, "title": path.stem.replace("_", " "),
            "year": "", "kind": "real", "url": ""}


def pages_of(path):
    """(page_number, text) pairs. TXT files may use form feeds as page breaks."""
    if path.suffix.lower() == ".pdf":
        # PDF text extraction is slow (minutes for a 300-page plan), so cache it per file version.
        stat = path.stat()
        cache = PROCESSED / "pdf_text" / f"{path.stem}.json"
        stamp = f"{stat.st_size}-{int(stat.st_mtime)}"
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached["stamp"] == stamp:
                return [tuple(p) for p in cached["pages"]]
        from pypdf import PdfReader
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(PdfReader(path).pages)]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"stamp": stamp, "pages": pages}, ensure_ascii=False), encoding="utf-8")
        return pages
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    return list(enumerate(text.split("\f"), start=1))


def line_key(line):
    """Normalise a line so running headers/footers match across pages (page numbers vary)."""
    return re.sub(r"\d+", "#", line.strip().lower())


def repeated_lines(pages, min_share=0.2):
    """Lines that recur on many pages of one document: running headers, footers, watermarks."""
    if len(pages) < 5:
        return set()
    counts = Counter()
    for _, text in pages:
        counts.update({line_key(l) for l in text.splitlines() if l.strip()})
    return {k for k, c in counts.items() if c >= max(3, min_share * len(pages)) and len(k) > 3}


def clean(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)                      # re-join hyphenated line breaks
    text = re.sub(r"^\s*(page\s*)?\d+\s*$", "", text, flags=re.I | re.M)  # bare page numbers
    text = re.sub(r"[ \t]+", " ", text)
    return text


def pack_sentences(text):
    chunks, current = [], ""
    for sentence in SENT_SPLIT.split(text):
        if current and len(current) + len(sentence) > MAX_CHARS:
            chunks.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def paragraphs(text):
    """Blank-line paragraphs; short ones (headings) merge forward, long ones split by sentence."""
    raw = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    raw = [p for p in raw if p]
    merged, carry = [], ""
    for p in raw:
        p = f"{carry} {p}".strip()
        if len(p) < MIN_CHARS:
            carry = p
            continue
        carry = ""
        merged.extend(pack_sentences(p) if len(p) > MAX_CHARS else [p])
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def ingest():
    sources = load_sources()
    rows = []
    for path in sorted(RAW.iterdir()):
        if path.suffix.lower() not in (".pdf", ".txt"):
            continue
        meta = metadata_for(path, sources)
        n = 0
        pages = pages_of(path)
        boilerplate = repeated_lines(pages)
        for page, text in pages:
            text = "\n".join(l for l in text.splitlines() if line_key(l) not in boilerplate)
            for para in paragraphs(clean(text)):
                n += 1
                rows.append({
                    "id": f"{meta['doc_id']}:p{page}:{n}",
                    "doc_id": meta["doc_id"],
                    "region": meta["region"],
                    "title": meta["title"],
                    "kind": meta["kind"],
                    "page": page,
                    "text": para,
                })
        print(f"  {path.name}: {n} passages")
    write_jsonl(PASSAGES, rows)
    return rows


def main():
    print("Ingesting data/raw ...")
    rows = ingest()
    print(f"Wrote {len(rows)} passages -> {PASSAGES.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
