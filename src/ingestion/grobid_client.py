from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "data" / "selected_papers"
OUTPUT_DIR = ROOT / "data" / "processed" / "grobid"

# Demo endpoint for one-time preprocessing only.
# For production, configure your own GROBID service.
GROBID_URL = os.getenv(
    "GROBID_URL",
    "https://grobidOrg-grobid-full.hf.space",
).rstrip("/")

NS_URI = "http://www.tei-c.org/ns/1.0"
XML_URI = "http://www.w3.org/XML/1998/namespace"
NS = {"tei": NS_URI}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalized_text(element) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def first_text(element, paths: list[str]) -> str:
    for path in paths:
        found = element.find(path, NS)
        value = normalized_text(found)
        if value:
            return value
    return ""


def get_year(element, raw_text: str) -> str | None:
    date_element = element.find(".//tei:date", NS)

    if date_element is not None:
        when = date_element.get("when", "")
        match = re.search(r"\b(?:19|20)\d{2}\b", when)
        if match:
            return match.group(0)

        match = re.search(
            r"\b(?:19|20)\d{2}\b",
            normalized_text(date_element),
        )
        if match:
            return match.group(0)

    # Fallback only; this could match a year in a title or URL.
    match = re.search(r"\b(?:19|20)\d{2}\b", raw_text)
    return match.group(0) if match else None


def parse_authors(bibl) -> list[str]:
    authors = []

    for author in bibl.findall("./tei:analytic/tei:author", NS):
        pass

    candidates = (
        bibl.findall("./tei:analytic/tei:author", NS)
        + bibl.findall("./tei:monogr/tei:author", NS)
    )

    for author in candidates:
        person = author.find(".//tei:persName", NS)

        if person is not None:
            given = " ".join(
                normalized_text(node)
                for node in person.findall("./tei:forename", NS)
                if normalized_text(node)
            )
            surname = " ".join(
                normalized_text(node)
                for node in person.findall("./tei:surname", NS)
                if normalized_text(node)
            )
            name = " ".join(part for part in (given, surname) if part)
        else:
            name = first_text(author, ["./tei:orgName"])

        if name and name not in authors:
            authors.append(name)

    return authors


def extract_bibliography(root) -> list[dict]:
    bibliography = []

    for index, bibl in enumerate(
        root.findall(".//tei:listBibl/tei:biblStruct", NS),
        start=1,
    ):
        raw = normalized_text(bibl)

        title = first_text(
            bibl,
            [
                "./tei:analytic/tei:title",
                "./tei:monogr/tei:title",
            ],
        )

        doi = first_text(bibl, ['.//tei:idno[@type="DOI"]'])
        url = first_text(bibl, ['.//tei:idno[@type="URL"]'])

        if not url:
            ptr = bibl.find(".//tei:ptr[@type='url']", NS)
            if ptr is not None:
                url = ptr.get("target", "")

        label = first_text(bibl, ["./tei:label"])

        record_id = bibl.get(f"{{{XML_URI}}}id")
        venue = first_text(bibl, ["./tei:monogr/tei:title"])

        bibliography.append({
            "reference_id": record_id or f"reference_{index}",
            "label": label,
            "title": title,
            "authors": parse_authors(bibl),
            "year": get_year(bibl, raw),
            "venue": venue,
            "doi": doi or None,
            "url": url or None,
            "raw_text": raw,
        })

    return bibliography


def extract_body(root) -> tuple[list[dict], list[dict]]:
    sections = []
    citations = []

    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return sections, citations

    def visit_div(div, parent_path: list[str]) -> None:
        heading = first_text(div, ["./tei:head"])
        current_path = parent_path + ([heading] if heading else [])

        # Process only direct paragraphs to avoid duplicating nested sections.
        for paragraph in div.findall("./tei:p", NS):
            paragraph_text = normalized_text(paragraph)

            if not paragraph_text:
                continue

            section_record = {
                "section_path": current_path,
                "text": paragraph_text,
            }
            sections.append(section_record)

            for ref in paragraph.iter():
                if local_name(ref.tag) != "ref":
                    continue
                if ref.get("type") != "bibr":
                    continue

                marker = normalized_text(ref)
                targets = re.findall(r"#([^\s#]+)", ref.get("target", ""))

                for target in targets:
                    citations.append({
                        "reference_id": target,
                        "marker_text": marker,
                        "section_path": current_path,
                        "context": paragraph_text,
                    })

        for child in list(div):
            if local_name(child.tag) == "div":
                visit_div(child, current_path)

    for child in list(body):
        if local_name(child.tag) == "div":
            visit_div(child, [])

        elif local_name(child.tag) == "p":
            text = normalized_text(child)
            if text:
                sections.append({"section_path": [], "text": text})

    return sections, citations


def extract_figures(root) -> list[dict]:
    records = []

    for index, figure in enumerate(root.findall(".//tei:figure", NS), start=1):
        graphics = []

        for node in figure.iter():
            if local_name(node.tag) == "graphic":
                target = node.get("url") or node.get("target")
                if target:
                    graphics.append(target)

        records.append({
            "figure_id": (
                figure.get(f"{{{XML_URI}}}id")
                or figure.get("id")
                or f"figure_{index}"
            ),
            "label": first_text(figure, ["./tei:label"]),
            "caption": first_text(
                figure,
                ["./tei:head", "./tei:figDesc"],
            ),
            "description": first_text(figure, ["./tei:figDesc"]),
            "graphic_targets": graphics,
            "raw_text": normalized_text(figure),
        })

    return records


def extract_tables(root) -> list[dict]:
    records = []

    for index, table in enumerate(
        [node for node in root.iter() if local_name(node.tag) == "table"],
        start=1,
    ):
        rows = []

        for row in table.iter():
            if local_name(row.tag) != "row":
                continue

            cells = [
                normalized_text(cell)
                for cell in row.iter()
                if local_name(cell.tag) in {"cell", "td", "th"}
                and normalized_text(cell)
            ]

            if cells:
                rows.append(cells)

        records.append({
            "table_id": f"table_{index}",
            "rows": rows,
            "raw_text": normalized_text(table),
        })

    return records


def parse_tei(tei_path: Path, pdf_path: Path) -> dict:
    root = ET.parse(tei_path).getroot()

    title = first_text(
        root,
        [
            ".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title",
            ".//tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:title",
        ],
    )

    abstract = first_text(
        root,
        [
            ".//tei:teiHeader/tei:profileDesc/tei:abstract",
            ".//tei:text/tei:front/tei:div[@type='abstract']",
        ],
    )

    bibliography = extract_bibliography(root)
    sections, citations = extract_body(root)
    figures = extract_figures(root)
    tables = extract_tables(root)

    known_ids = {item["reference_id"] for item in bibliography}
    unresolved = sorted({
        citation["reference_id"]
        for citation in citations
        if citation["reference_id"] not in known_ids
    })

    return {
        "schema_version": "1.0",
        "paper_id": pdf_path.stem,
        "source_pdf": pdf_path.relative_to(ROOT).as_posix(),
        "extraction_method": "GROBID TEI",
        "document": {
            "title": title or pdf_path.stem,
            "abstract": abstract,
        },
        "sections": sections,
        "bibliography": bibliography,
        "citation_occurrences": citations,
        "figures": figures,
        "tables": tables,
        "validation": {
            "bibliography_entries": len(bibliography),
            "citation_occurrences": len(citations),
            "linked_citation_occurrences": sum(
                citation["reference_id"] in known_ids
                for citation in citations
            ),
            "unresolved_reference_ids": unresolved,
        },
    }


def fetch_grobid(pdf_path: Path, tei_path: Path) -> None:
    last_error = None
    endpoint = f"{GROBID_URL}/api/processFulltextDocument"

    for attempt in range(1, 4):
        try:
            with pdf_path.open("rb") as handle:
                response = requests.post(
                    endpoint,
                    files={
                        "input": (
                            pdf_path.name,
                            handle,
                            "application/pdf",
                        )
                    },
                    data={
                        "consolidateHeader": "0",
                        "consolidateCitations": "0",
                        "includeRawCitations": "1",
                    },
                    headers={"Accept": "application/xml"},
                    timeout=(20, 240),
                )

            if response.status_code == 200 and response.text.strip():
                # Validate before saving the result.
                ET.fromstring(response.content)
                tei_path.write_bytes(response.content)
                return

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        except (requests.RequestException, ET.ParseError) as exc:
            last_error = str(exc)

        if attempt < 3:
            time.sleep(4 * attempt)

    raise RuntimeError(f"GROBID request failed: {last_error}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdfs:
        raise SystemExit(f"No PDFs found in {PAPERS_DIR}")

    reports = []

    for pdf_path in pdfs:
        tei_path = OUTPUT_DIR / f"{pdf_path.stem}.tei.xml"
        json_path = OUTPUT_DIR / f"{pdf_path.stem}.grobid.json"

        print(f"\nProcessing: {pdf_path.name}")

        try:
            if not tei_path.exists():
                print("  Requesting structured extraction...")
                fetch_grobid(pdf_path, tei_path)
            else:
                print("  Reusing saved TEI XML.")

            result = parse_tei(tei_path, pdf_path)
            json_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            report = {
                "paper": pdf_path.name,
                **result["validation"],
                "title": result["document"]["title"],
                "tei": tei_path.relative_to(ROOT).as_posix(),
                "json": json_path.relative_to(ROOT).as_posix(),
            }
            reports.append(report)
            print(json.dumps(report, indent=2))

        except Exception as exc:
            print(f"  ERROR: {exc}")
            reports.append({
                "paper": pdf_path.name,
                "error": str(exc),
            })

    report_path = OUTPUT_DIR / "grobid_report.json"
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {report_path.relative_to(ROOT)}")

    if any("error" in item for item in reports):
        raise SystemExit(
            "One or more papers failed. Successful results were retained."
        )


if __name__ == "__main__":
    main()
