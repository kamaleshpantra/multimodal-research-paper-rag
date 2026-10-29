from pathlib import Path

path = Path('src/ingestion/pdf_parser.py')
if not path.exists():
    raise SystemExit('Run this script from the D:\\research-paper-rag project root; src/ingestion/pdf_parser.py was not found.')

source = path.read_text(encoding='utf-8-sig')

if 'import unicodedata' not in source:
    source = source.replace('import sys\n', 'import sys\nimport unicodedata\n', 1)

old_helpers = '''def is_reference_heading(line: str) -> bool:
    return bool(REFERENCE_HEADING.fullmatch(line.strip()))


def clean_table_rows(rows):
'''
new_helpers = r'''def normalize_heading(text: str) -> str:
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
                    or any(is_reference_heading(value) for value in span_texts)
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
'''
if old_helpers not in source:
    raise SystemExit('Could not find the expected helper section. No changes written; inspect src/ingestion/pdf_parser.py before retrying.')
source = source.replace(old_helpers, new_helpers, 1)

old_lines = '''            text = page.get_text("text", sort=True).strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            headings = [
'''
new_lines = '''            text = page.get_text("text", sort=True).strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            layout_lines = extract_layout_lines(page)

            headings = [
'''
if old_lines not in source:
    raise SystemExit('Could not find the expected page text extraction section. No changes written.')
source = source.replace(old_lines, new_lines, 1)

old_page_field = '''                "page_number": page_number,
                "text": text,
                "headings": headings,
'''
new_page_field = '''                "page_number": page_number,
                "text": text,
                "layout_lines": layout_lines,
                "headings": headings,
'''
if old_page_field not in source:
    raise SystemExit('Could not find the expected pages dictionary. No changes written.')
source = source.replace(old_page_field, new_page_field, 1)

old_title = '''            "title": metadata.get("title") or pdf_path.stem,
'''
new_title = '''            "title": (
                (metadata.get("title") or "").strip()
                if (metadata.get("title") or "").strip()
                and (metadata.get("title") or "").strip() != pdf_path.stem
                else extract_pdf_title(document[0], pdf_path.stem)
            ),
'''
if old_title not in source:
    raise SystemExit('Could not find the expected document title field. No changes written.')
source = source.replace(old_title, new_title, 1)

old_reference_loop = '''    for page_data in pages:
        for line in page_data["text"].splitlines():
            if is_reference_heading(line):
                references_started = True
                if references_start_page is None:
                    references_start_page = page_data["page_number"]
                continue

            if references_started and line.strip():
                reference_lines.append(line.strip())
'''
new_reference_loop = '''    for page_data in pages:
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
'''
if old_reference_loop not in source:
    raise SystemExit('Could not find the expected reference extraction loop. No changes written.')
source = source.replace(old_reference_loop, new_reference_loop, 1)

path.write_text(source, encoding='utf-8')
print('Patched src/ingestion/pdf_parser.py successfully.')
print('Added layout-aware heading detection, title inference, and appendix cutoff for reference candidates.')
