from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "src" else Path.cwd()
# The script is intended to be copied to D:\research-paper-rag\src\retrieval\build_corpus.py.
# Resolve project root from this script location, with a cwd fallback for convenience.
if (Path(__file__).resolve().parent.name == "retrieval" and Path(__file__).resolve().parent.parent.name == "src"):
    ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
GROBID_DIR = PROCESSED_DIR / "grobid"
OUTPUT_DIR = PROCESSED_DIR
OUTPUT_FILE = OUTPUT_DIR / "corpus.jsonl"
REPORT_FILE = OUTPUT_DIR / "corpus_report.json"

MAX_CHARS = 1200
OVERLAP_CHARS = 180


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split page text into bounded chunks while preferring paragraph/sentence boundaries."""
    text = text.replace("\x00", " ").strip()
    if not text:
        return []

    # Preserve visible paragraph boundaries but collapse line wraps inside paragraphs.
    paragraphs = []
    for part in re.split(r"\n\s*\n+", text):
        part = normalize_spaces(part)
        if part:
            paragraphs.append(part)
    if not paragraphs:
        paragraphs = [normalize_spaces(text)]

    chunks: list[str] = []
    current = ""

    def emit_long(paragraph: str) -> list[str]:
        results = []
        remaining = paragraph
        while len(remaining) > max_chars:
            cut = max_chars
            # Prefer the final sentence/word boundary in the second half of the window.
            floor = int(max_chars * 0.55)
            candidates = [remaining.rfind(mark, floor, max_chars) for mark in (". ", "; ", ": ", " ")]
            best = max(candidates)
            if best >= floor:
                cut = best + (1 if remaining[best:best + 2] in {". ", "; ", ": "} else 0)
            piece = remaining[:cut].strip()
            if piece:
                results.append(piece)
            next_start = max(1, cut - overlap)
            remaining = remaining[next_start:].strip()
        if remaining:
            results.append(remaining)
        return results

    for paragraph in paragraphs:
        pieces = emit_long(paragraph) if len(paragraph) > max_chars else [paragraph]
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current += " " + piece
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)

    # Deduplicate exact adjacent chunks, preserving order.
    unique: list[str] = []
    for chunk in chunks:
        if chunk and (not unique or chunk != unique[-1]):
            unique.append(chunk)
    return unique


def make_record(
    *,
    paper_id: str,
    source_type: str,
    text: str,
    source_id: str,
    page_number: int | None = None,
    section: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": source_id,
        "paper_id": paper_id,
        "source_type": source_type,
        "text": normalize_spaces(text),
        "page_number": page_number,
        "section": section,
        "metadata": metadata or {},
    }


def paper_section_hint(page: dict[str, Any]) -> str | None:
    headings = [normalize_spaces(h) for h in page.get("headings", []) if normalize_spaces(h)]
    if not headings:
        return None
    # Keep a short page-local hint; this is not claimed to be the globally active section.
    return " / ".join(headings[:3])


def build_page_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    paper_id = data["paper_id"]

    for page in data.get("pages", []):
        page_num = page.get("page_number")
        page_text = page.get("text", "")
        section_hint = paper_section_hint(page)

        for i, chunk in enumerate(split_text(page_text), start=1):
            records.append(make_record(
                paper_id=paper_id,
                source_type="page_text",
                text=chunk,
                source_id=f"{paper_id}:p{page_num}:text{i}",
                page_number=page_num,
                section=section_hint,
                metadata={
                    "headings_on_page": page.get("headings", []),
                    "numeric_citation_markers": page.get("numeric_citation_markers", []),
                    "source_pdf": data.get("source_pdf"),
                },
            ))

        # Table chunks retain row boundaries and exact page-level provenance.
        for table_index, table in enumerate(page.get("tables", []), start=1):
            rows = table.get("rows", [])
            row_lines = [" | ".join(str(cell or "") for cell in row) for row in rows]
            table_text = "\n".join(line for line in row_lines if line.strip())
            if table_text:
                records.append(make_record(
                    paper_id=paper_id,
                    source_type="table",
                    text=f"Table extracted from page {page_num}.\n{table_text}",
                    source_id=f"{paper_id}:p{page_num}:table{table_index}",
                    page_number=page_num,
                    section=section_hint,
                    metadata={
                        "table_number_on_page": table.get("table_number_on_page", table_index),
                        "bbox": table.get("bbox"),
                        "rendered_page": page.get("rendered_page"),
                        "source_pdf": data.get("source_pdf"),
                    },
                ))

        # Caption-only records preserve a link to the page/image, but do not claim
        # that the underlying visual has been interpreted by a vision model yet.
        images = page.get("images", [])
        for caption_index, caption in enumerate(page.get("caption_candidates", []), start=1):
            caption = normalize_spaces(caption)
            if not caption:
                continue
            records.append(make_record(
                paper_id=paper_id,
                source_type="figure_caption",
                text=f"Figure/table caption candidate on page {page_num}: {caption}",
                source_id=f"{paper_id}:p{page_num}:caption{caption_index}",
                page_number=page_num,
                section=section_hint,
                metadata={
                    "caption": caption,
                    "image_paths_on_page": [item.get("path") for item in images if item.get("path")],
                    "rendered_page": page.get("rendered_page"),
                    "visual_description_status": "caption_only_not_vision_verified",
                    "source_pdf": data.get("source_pdf"),
                },
            ))

    return records


def build_grobid_records(grobid: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    paper_id = grobid["paper_id"]

    # Prefer GROBID's section-aware paragraph order for concept questions.
    # PyMuPDF page text remains available for exact page-level provenance, but
    # multi-column PDFs can have interleaved reading order in plain text output.
    document = grobid.get("document", {})
    abstract = normalize_spaces(document.get("abstract", ""))
    if abstract:
        for part_index, part in enumerate(split_text(abstract), start=1):
            records.append(make_record(
                paper_id=paper_id,
                source_type="abstract",
                text=part,
                source_id=f"{paper_id}:abstract:{part_index}",
                section="Abstract",
                metadata={
                    "section_path": ["Abstract"],
                    "source_pdf": grobid.get("source_pdf"),
                    "page_number_note": "GROBID abstract record; original PDF page not assigned.",
                },
            ))

    for section_index, section_record in enumerate(grobid.get("sections", []), start=1):
        section_text = normalize_spaces(section_record.get("text", ""))
        if not section_text:
            continue
        section_path = section_record.get("section_path", [])
        section = " / ".join(str(item) for item in section_path if str(item).strip()) or None
        for chunk_index, chunk in enumerate(split_text(section_text), start=1):
            records.append(make_record(
                paper_id=paper_id,
                source_type="section_text",
                text=chunk,
                source_id=f"{paper_id}:grobid_section:{section_index}:chunk{chunk_index}",
                section=section,
                metadata={
                    "section_path": section_path,
                    "source_pdf": grobid.get("source_pdf"),
                    "page_number_note": "GROBID section paragraph; exact PDF page not mapped.",
                },
            ))

    citations_by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in grobid.get("citation_occurrences", []):
        citations_by_ref[occurrence.get("reference_id", "")].append(occurrence)

    # Each bibliography item is individually retrievable and keeps its stable GROBID ID.
    for index, ref in enumerate(grobid.get("bibliography", []), start=1):
        ref_id = ref.get("reference_id") or f"reference_{index}"
        authors = ref.get("authors", [])
        authors_text = ", ".join(authors)
        fields = [
            f"Reference cited by {grobid.get('document', {}).get('title', paper_id)}",
            f"Reference ID: {ref_id}",
            f"Authors: {authors_text}" if authors_text else "",
            f"Title: {ref.get('title')}" if ref.get("title") else "",
            f"Year: {ref.get('year')}" if ref.get("year") else "",
            f"Venue: {ref.get('venue')}" if ref.get("venue") else "",
            f"DOI: {ref.get('doi')}" if ref.get("doi") else "",
            f"URL: {ref.get('url')}" if ref.get("url") else "",
            f"Raw bibliography entry: {ref.get('raw_text')}" if ref.get("raw_text") else "",
        ]
        contexts = []
        seen_contexts = set()
        for occ in citations_by_ref.get(ref_id, []):
            context = normalize_spaces(occ.get("context", ""))
            if context and context not in seen_contexts:
                seen_contexts.add(context)
                section_path = occ.get("section_path", [])
                section = " / ".join(section_path) if section_path else ""
                contexts.append(f"Citation context{f' (section: {section})' if section else ''}: {context}")
            if len(contexts) >= 3:
                break
        fields.extend(contexts)
        text = "\n".join(item for item in fields if item)
        records.append(make_record(
            paper_id=paper_id,
            source_type="bibliography_reference",
            text=text,
            source_id=f"{paper_id}:ref:{ref_id}",
            metadata={
                "reference_id": ref_id,
                "label": ref.get("label"),
                "title": ref.get("title"),
                "authors": authors,
                "year": ref.get("year"),
                "venue": ref.get("venue"),
                "doi": ref.get("doi"),
                "url": ref.get("url"),
                "citation_occurrence_count": len(citations_by_ref.get(ref_id, [])),
                "source_pdf": grobid.get("source_pdf"),
                "page_number_note": "Bibliography page not assigned by GROBID corpus builder.",
            },
        ))

    # Index citation contexts separately so a question like “why does the author cite X?”
    # can retrieve the actual surrounding paragraph, rather than only the bibliography row.
    for index, occurrence in enumerate(grobid.get("citation_occurrences", []), start=1):
        context = normalize_spaces(occurrence.get("context", ""))
        if not context:
            continue
        section_path = occurrence.get("section_path", [])
        section = " / ".join(section_path) if section_path else None
        marker = occurrence.get("marker_text", "")
        ref_id = occurrence.get("reference_id", "")
        records.append(make_record(
            paper_id=paper_id,
            source_type="citation_context",
            text=f"Citation marker {marker} points to reference {ref_id}. {context}",
            source_id=f"{paper_id}:citation:{index}",
            page_number=None,
            section=section,
            metadata={
                "reference_id": ref_id,
                "marker_text": marker,
                "section_path": section_path,
                "page_number_note": "GROBID citation context has no page mapping in this version.",
            },
        ))

    return records


def main() -> None:
    if not PROCESSED_DIR.exists():
        raise SystemExit(f"Processed data directory not found: {PROCESSED_DIR}")

    records: list[dict[str, Any]] = []
    summary = {"papers": [], "source_type_counts": {}, "total_chunks": 0}

    parsed_files = sorted(
        path for path in PROCESSED_DIR.glob("*.json")
        if path.name != "ingestion_report.json"
    )
    if not parsed_files:
        raise SystemExit("No PyMuPDF JSON files found; run pdf_parser.py first.")

    for parsed_path in parsed_files:
        data = load_json(parsed_path)
        paper_id = data.get("paper_id")
        if not paper_id:
            print(f"Skipping file without paper_id: {parsed_path}")
            continue

        before = len(records)
        records.extend(build_page_records(data))

        grobid_path = GROBID_DIR / f"{paper_id}.grobid.json"
        if grobid_path.exists():
            grobid = load_json(grobid_path)
            records.extend(build_grobid_records(grobid))
        else:
            print(f"WARNING: No GROBID data for {paper_id}; bibliography/citation records omitted.")

        summary["papers"].append({
            "paper_id": paper_id,
            "page_count": data.get("document", {}).get("page_count"),
            "chunks_added": len(records) - before,
            "grobid_available": grobid_path.exists(),
        })

    # Final guard: no empty records and no duplicate chunk IDs.
    records = [r for r in records if r.get("text", "").strip()]
    ids = [r["chunk_id"] for r in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted({chunk_id for chunk_id in ids if ids.count(chunk_id) > 1})
        raise RuntimeError(f"Duplicate chunk IDs generated: {duplicates[:10]}")

    OUTPUT_FILE.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["source_type"]] += 1
    summary["source_type_counts"] = dict(sorted(counts.items()))
    summary["total_chunks"] = len(records)
    REPORT_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCorpus written to: {OUTPUT_FILE}")
    print(f"Report written to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
