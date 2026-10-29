from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "data" / "selected_papers"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGES_DIR = ROOT / "data" / "images"

REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:references|bibliography|works cited)\s*$",
    re.IGNORECASE,
)

SECTION_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|background|related work|"
    r"literature review|methodology|methods|proposed method|"
    r"experimental setup|experiments|results|discussion|"
    r"limitations|conclusion|conclusions|future work|"
    r"references|bibliography|appendix)\s*:?\s*$",
    re.IGNORECASE,
)

CAPTION_PATTERN = re.compile(
    r"^\s*(?:figure|fig\.?|table|algorithm|chart)\s*"
    r"[A-Z]?\d+(?:\.\d+)*\s*[:.)\-]?",
    re.IGNORECASE,
)

NUMERIC_CITATION_PATTERN = re.compile(
    r"\[(\d{1,3}(?:\s*[-,;]\s*\d{1,3})*)\]"
)


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def normalize_heading(text: str) -> str:
    """Normalize PDF text for robust heading matching."""
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", text)
    return text.strip(" .:-").casefold()


def is_reference_heading(line: str) -> bool:
    return normalize_heading(line) in {"references", "bibliography", "works cited"}


def is_appendix_heading(line: str) -> bool:
    normalized = normalize_heading(line)
    return bool(re.fullmatch(r"appendices?(?:\s+for\b.*)?", normalized))


def extract_layout_lines(page) -> list[dict]:
    """Keep PDF line/span layout so headings can be detected despite column interleaving."""
    result = []
    page_dict = page.get_text("dict", sort=True)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans).strip()
            if not line_text:
                continue

            span_texts = [span.get("text", "") for span in spans]
            sizes = [float(span.get("size", 0.0)) for span in spans]
            bbox = line.get("bbox", (0, 0, 0, 0))
            result.append({
                "text": line_text,
                "bbox": [round(float(value), 2) for value in bbox],
                "max_font_size": max(sizes, default=0.0),
                # Some multi-column PDFs place the heading span beside unrelated text.
                "reference_heading": (
                    is_reference_heading(line_text)
                    or any(
                        is_reference_heading(span.get("text", ""))
                        and (
                            (int(span.get("flags", 0)) & 16) != 0
                            or float(span.get("size", 0.0)) >= 10.5
                        )
                        for span in spans
                    )
                ),
                "appendix_heading": (
                    is_appendix_heading(line_text)
                    or any(is_appendix_heading(value) for value in span_texts)
                ),
            })

    return result


def extract_pdf_title(page, fallback: str) -> str:
    """Infer a title from large-font text near the top of page one."""
    lines = extract_layout_lines(page)
    upper_lines = []
    upper_limit = float(page.rect.height) * 0.36

    for line in lines:
        text = line["text"].strip()
        normalized = normalize_heading(text)
        if line["bbox"][1] > upper_limit or len(text) < 4:
            continue
        if normalized in {"abstract", "contents", "table of contents"}:
            continue
        if "@" in text or normalized.startswith(("arxiv:", "doi:")):
            continue
        if not any(char.isalpha() for char in text):
            continue
        upper_lines.append(line)

    if not upper_lines:
        return fallback.replace("_", " ").strip()

    max_size = max(line["max_font_size"] for line in upper_lines)
    threshold = max(10.0, max_size * 0.78)
    candidates = [line for line in upper_lines if line["max_font_size"] >= threshold]
    candidates.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))

    chosen = []
    for line in candidates:
        if not chosen:
            chosen.append(line)
            continue
        vertical_gap = line["bbox"][1] - chosen[-1]["bbox"][1]
        if vertical_gap <= max(30.0, max_size * 1.8):
            chosen.append(line)
        else:
            break

    title = " ".join(line["text"] for line in chosen).strip()
    title = re.sub(r"\s+", " ", title)
    return title or fallback.replace("_", " ").strip()


def clean_table_rows(rows):
    return [
        [str(cell).strip() if cell is not None else None for cell in row]
        for row in rows
    ]


def parse_pdf(pdf_path: Path) -> dict:
    paper_id = pdf_path.stem
    image_dir = IMAGES_DIR / paper_id
    image_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pages = []
    warnings = []
    all_image_xrefs = set()

    with pymupdf.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError(f"Encrypted PDF cannot be opened: {pdf_path.name}")

        metadata = document.metadata or {}
        toc = document.get_toc()

        document_title = (metadata.get("title") or "").strip()
        if not document_title or document_title == pdf_path.stem:
            document_title = extract_pdf_title(document[0], pdf_path.stem)

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            text = page.get_text("text", sort=True).strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            layout_lines = extract_layout_lines(page)

            headings = [
                line for line in lines
                if SECTION_HEADING.fullmatch(line)
            ]

            captions = [
                line for line in lines
                if CAPTION_PATTERN.match(line)
            ]

            citation_markers = sorted(
                {
                    match.group(1).strip()
                    for match in NUMERIC_CITATION_PATTERN.finditer(text)
                }
            )

            extracted_images = []

            for image_info in page.get_images(full=True):
                xref = image_info[0]

                try:
                    image_data = document.extract_image(xref)
                    width = image_data.get("width", 0)
                    height = image_data.get("height", 0)

                    # Exclude tiny decorative images from the main index.
                    if width * height < 10000:
                        continue

                    extension = image_data.get("ext", "png")
                    image_path = image_dir / f"image_{xref}.{extension}"

                    if xref not in all_image_xrefs:
                        image_path.write_bytes(image_data["image"])
                        all_image_xrefs.add(xref)

                    extracted_images.append({
                        "xref": xref,
                        "width": width,
                        "height": height,
                        "path": relative_path(image_path),
                    })

                except Exception as exc:
                    warnings.append(
                        f"Page {page_number}: image {xref} extraction failed: {exc}"
                    )

            extracted_tables = []

            try:
                table_finder = page.find_tables()

                for table_number, table in enumerate(
                    table_finder.tables, start=1
                ):
                    rows = clean_table_rows(table.extract())

                    if not rows:
                        continue

                    extracted_tables.append({
                        "table_number_on_page": table_number,
                        "bbox": list(table.bbox),
                        "rows": rows,
                    })

            except Exception as exc:
                warnings.append(
                    f"Page {page_number}: table extraction failed: {exc}"
                )

            # Save a preview for pages likely to contain visual evidence.
            drawing_count = 0
            try:
                drawing_count = len(page.get_drawings())
            except Exception:
                pass

            rendered_page = None

            if (
                extracted_images
                or extracted_tables
                or captions
                or drawing_count >= 12
            ):
                rendered_path = image_dir / f"page_{page_number:04d}.png"

                if not rendered_path.exists():
                    try:
                        pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(1.4, 1.4),
                            alpha=False,
                        )
                        pixmap.save(rendered_path)
                    except Exception as exc:
                        warnings.append(
                            f"Page {page_number}: preview rendering failed: {exc}"
                        )

                if rendered_path.exists():
                    rendered_page = relative_path(rendered_path)

            pages.append({
                "page_number": page_number,
                "text": text,
                "layout_lines": layout_lines,
                "headings": headings,
                "numeric_citation_markers": citation_markers,
                "caption_candidates": captions,
                "images": extracted_images,
                "tables": extracted_tables,
                "drawing_count": drawing_count,
                "rendered_page": rendered_page,
                "character_count": len(text),
            })

    # Preserve the candidate references section for later structured parsing.
    references_started = False
    references_start_page = None
    reference_lines = []

    for page_data in pages:
        for line_data in page_data.get("layout_lines", []):
            line = line_data["text"].strip()

            if not references_started and line_data.get("reference_heading"):
                references_started = True
                if references_start_page is None:
                    references_start_page = page_data["page_number"]
                continue

            # Avoid treating appendices as bibliography content.
            if references_started and line_data.get("appendix_heading"):
                references_started = False
                break

            if references_started and line:
                reference_lines.append(line)

    output = {
        "schema_version": "1.0",
        "paper_id": paper_id,
        "source_pdf": relative_path(pdf_path),
        "document": {
            "title": document_title,
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
            "page_count": len(pages),
            "table_of_contents": toc,
        },
        "pages": pages,
        "reference_section": {
            "start_page": references_start_page,
            "raw_text": "\n".join(reference_lines),
            "extraction_method": "heading-based candidate extraction",
            "warning": (
                "Candidate references only; citation-to-reference "
                "matching requires structured validation."
            ),
        },
        "warnings": warnings,
    }

    output_path = PROCESSED_DIR / f"{paper_id}.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_tables = sum(len(page["tables"]) for page in pages)
    total_captions = sum(len(page["caption_candidates"]) for page in pages)
    total_chars = sum(page["character_count"] for page in pages)
    pages_with_text = sum(bool(page["text"]) for page in pages)
    pages_with_preview = sum(bool(page["rendered_page"]) for page in pages)

    return {
        "paper": pdf_path.name,
        "pages": len(pages),
        "pages_with_text": pages_with_text,
        "characters": total_chars,
        "image_files": len(all_image_xrefs),
        "tables_detected": total_tables,
        "caption_candidates": total_captions,
        "reference_start_page": references_start_page,
        "preview_pages": pages_with_preview,
        "warnings": len(warnings),
        "output": relative_path(output_path),
    }


def main():
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in: {PAPERS_DIR}")
        sys.exit(1)

    reports = []

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            report = parse_pdf(pdf_path)
            reports.append(report)
            print(json.dumps(report, indent=2))

        except Exception as exc:
            print(f"ERROR processing {pdf_path.name}: {exc}")
            raise

    report_path = PROCESSED_DIR / "ingestion_report.json"
    report_path.write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )

    print(f"\nIngestion report saved to: {relative_path(report_path)}")


if __name__ == "__main__":
    main()
