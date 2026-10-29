from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pymupdf
import requests

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "src" else Path.cwd()
if Path(__file__).resolve().parent.name == "ingestion" and Path(__file__).resolve().parent.parent.name == "src":
    ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
PAPERS_DIR = ROOT / "data" / "selected_papers"
VISION_DIR = PROCESSED_DIR / "vision"
CANDIDATES_FILE = VISION_DIR / "vision_candidates.jsonl"
REPORT_FILE = VISION_DIR / "vision_report.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
VISION_NUM_PREDICT = int(os.getenv("VISION_NUM_PREDICT", "240"))
VISION_TIMEOUT_SECONDS = int(os.getenv("VISION_TIMEOUT_SECONDS", "300"))

VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "figure_type": {
            "type": "string",
            "enum": ["architecture diagram", "flowchart", "line chart", "bar chart", "scatter plot", "table", "other", "unclear"],
        },
        "visible_labels": {
            "type": "array", "maxItems": 6,
            "items": {"type": "string", "maxLength": 45},
        },
        "relationships": {
            "type": "array", "maxItems": 3,
            "items": {"type": "string", "maxLength": 85},
        },
        "chart_axes_and_units": {
            "type": "array", "maxItems": 3,
            "items": {"type": "string", "maxLength": 70},
        },
        "uncertainties": {
            "type": "array", "maxItems": 2,
            "items": {"type": "string", "maxLength": 75},
        },
    },
    "required": ["figure_type", "visible_labels", "relationships", "chart_axes_and_units", "uncertainties"],
    "additionalProperties": False,
}

PROMPT = """Describe this scientific paper figure for a research assistant. Return only JSON matching the schema.
This is extraction, not question answering: do not answer example questions shown inside the visual.
- Identify the main figure type.
- Give at most 6 important, readable labels and at most 3 clearly visible relationships.
- For charts, list only readable axes, units, legends or visible trends in chart_axes_and_units. Never guess numeric values.
- For diagrams, put arrows/data flows in relationships and leave chart_axes_and_units empty.
- Do not reproduce or invent the caption; it is stored separately from the original PDF.
- If a relationship or label is uncertain, use uncertainties rather than guessing.
- Keep strings short. Do not add a summary sentence."""

CAPTION_PREFIX = re.compile(r"^\s*(figure|fig\.?|table|algorithm|chart)\s*([A-Z]?\d+(?:\.\d+)*)", re.IGNORECASE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def caption_key(caption: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", caption.casefold())


def locate_caption(page: pymupdf.Page, caption: str) -> pymupdf.Rect | None:
    # Full caption matching works best when the PDF has an intact text span.
    try:
        boxes = page.search_for(caption)
        if boxes:
            return boxes[0]
    except Exception:
        pass

    match = CAPTION_PREFIX.match(caption)
    if match:
        label = f"{match.group(1)} {match.group(2)}"
        for candidate in (label, label.replace("Figure", "Fig."), label.replace("Fig.", "Figure")):
            try:
                boxes = page.search_for(candidate)
                if boxes:
                    return boxes[0]
            except Exception:
                continue
    return None


def make_crop(pdf_path: Path, page_num: int, caption: str, out_path: Path) -> tuple[Path, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as doc:
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Invalid page {page_num} for {pdf_path.name} (pages={len(doc)})")
        page = doc[page_num - 1]
        box = locate_caption(page, caption)
        if box is None:
            # Fail open to a full-page crop, but record that less precise fallback was used.
            clip = page.rect
            crop_method = "full_page_fallback_caption_not_located"
        else:
            # Most scientific figure captions are below their figure. Crop the area above and just below it.
            clip = pymupdf.Rect(
                0,
                max(page.rect.y0, box.y0 - 480),
                page.rect.x1,
                min(page.rect.y1, box.y1 + 26),
            )
            crop_method = "caption_anchored_480pt_above"
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8), clip=clip, alpha=False)
        pix.save(out_path)
    return out_path, crop_method


def call_vision(image_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": VISION_MODEL,
        "stream": False,
        "format": VISION_SCHEMA,
        "keep_alive": "10m",
        "messages": [{
            "role": "user",
            "content": PROMPT,
            "images": [base64.b64encode(image_path.read_bytes()).decode("ascii")],
        }],
        "options": {
            "temperature": 0,
            "num_predict": VISION_NUM_PREDICT,
            "num_ctx": 4096,
        },
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=(15, VISION_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    result = response.json()
    raw = result.get("message", {}).get("content", "").strip()
    parsed = json.loads(raw)
    return parsed, {
        "done_reason": result.get("done_reason"),
        "total_duration_seconds": round(result.get("total_duration", 0) / 1e9, 2),
        "prompt_tokens": result.get("prompt_eval_count"),
        "output_tokens": result.get("eval_count"),
    }


def candidate_text(caption: str, parsed: dict[str, Any]) -> str:
    parts = [f"Scientific visual from paper. Original PDF caption: {caption}"]
    figure_type = normalize(parsed.get("figure_type"))
    if figure_type:
        parts.append(f"Visual type: {figure_type}.")
    labels = [normalize(x) for x in parsed.get("visible_labels", []) if normalize(x)]
    relationships = [normalize(x) for x in parsed.get("relationships", []) if normalize(x)]
    chart_info = [normalize(x) for x in parsed.get("chart_axes_and_units", []) if normalize(x)]
    uncertainties = [normalize(x) for x in parsed.get("uncertainties", []) if normalize(x)]
    if labels:
        parts.append("Visible labels: " + "; ".join(labels) + ".")
    if relationships:
        parts.append("Visible relationships: " + "; ".join(relationships) + ".")
    if chart_info:
        parts.append("Chart axes, units, legends or trends: " + "; ".join(chart_info) + ".")
    if uncertainties:
        parts.append("Vision-model uncertainties: " + "; ".join(uncertainties) + ".")
    return "\n".join(parts)


def quality_flags(parsed: dict[str, Any], stats: dict[str, Any], crop_method: str) -> list[str]:
    flags: list[str] = []
    labels = [normalize(x).casefold() for x in parsed.get("visible_labels", []) if normalize(x)]
    if len(labels) != len(set(labels)):
        flags.append("duplicate_visible_labels")
    if parsed.get("figure_type") in (None, "unclear"):
        flags.append("figure_type_unclear")
    if not parsed.get("visible_labels") and not parsed.get("chart_axes_and_units"):
        flags.append("no_labels_or_chart_details")
    if crop_method.endswith("fallback_caption_not_located"):
        flags.append("caption_not_located_full_page_used")
    if stats.get("done_reason") == "length":
        flags.append("output_token_limit_reached")
    if not parsed.get("relationships") and parsed.get("figure_type") in {"architecture diagram", "flowchart"}:
        flags.append("diagram_relationships_missing")
    return flags


def collect_candidates(paper_filter: str | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for json_path in sorted(PROCESSED_DIR.glob("*.json")):
        if json_path.name == "ingestion_report.json":
            continue
        data = load_json(json_path)
        paper_id = data.get("paper_id")
        if not paper_id or (paper_filter and paper_id != paper_filter):
            continue
        for page in data.get("pages", []):
            page_num = int(page.get("page_number") or 0)
            seen = set()
            for index, caption in enumerate(page.get("caption_candidates", []), start=1):
                caption = normalize(caption)
                if not caption or caption_key(caption) in seen:
                    continue
                seen.add(caption_key(caption))
                if not CAPTION_PREFIX.match(caption):
                    continue
                candidates.append({
                    "paper_id": paper_id,
                    "paper_title": data.get("document", {}).get("title") or paper_id,
                    "page_number": page_num,
                    "caption_index_on_page": index,
                    "caption": caption,
                    "source_pdf": data.get("source_pdf") or f"data/selected_papers/{paper_id}.pdf",
                })
    return candidates


def read_existing() -> dict[str, dict[str, Any]]:
    if not CANDIDATES_FILE.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for line in CANDIDATES_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            existing[record.get("chunk_id", "")] = record
        except json.JSONDecodeError:
            continue
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description="Create reviewed visual-evidence candidates from paper figures/tables.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum visuals to process in this run (default: 3).")
    parser.add_argument("--paper", type=str, default=None, help="Optional paper_id filter.")
    parser.add_argument("--force", action="store_true", help="Reprocess candidates already present in output.")
    args = parser.parse_args()

    VISION_DIR.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates(args.paper)
    if not candidates:
        print(f"No figure/table caption candidates found in {PROCESSED_DIR}")
        return 1

    existing = read_existing()
    pending_by_paper: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        key = f"{item['paper_id']}:visual:p{item['page_number']}:c{item['caption_index_on_page']}"
        if not args.force and key in existing and existing[key].get("metadata", {}).get("vision"):
            continue
        item["chunk_id"] = key
        pending_by_paper.setdefault(str(item["paper_id"]), []).append(item)

    # Round-robin by paper so a small time-limited batch samples the whole corpus
    # instead of processing every early caption from the first PDF.
    selected: list[dict[str, Any]] = []
    paper_order = sorted(pending_by_paper)
    while len(selected) < max(args.limit, 1) and any(pending_by_paper.values()):
        for paper_id in paper_order:
            queue = pending_by_paper.get(paper_id, [])
            if queue and len(selected) < max(args.limit, 1):
                selected.append(queue.pop(0))

    if not selected:
        print("No new visuals to process. Use --force to reprocess existing candidates.")
        return 0

    print(f"Visual caption candidates available: {len(candidates)}")
    print(f"Processing now: {len(selected)} (limit={args.limit})")
    print(f"Vision model: {VISION_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_BASE_URL}")

    results = dict(existing)
    failures = []
    for i, item in enumerate(selected, start=1):
        print(f"\n[{i}/{len(selected)}] {item['paper_id']} p.{item['page_number']}: {item['caption'][:110]}")
        pdf_path = ROOT / item["source_pdf"]
        if not pdf_path.exists():
            # Fall back to the canonical papers directory if source_pdf metadata was stale.
            pdf_path = PAPERS_DIR / f"{item['paper_id']}.pdf"
        safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", item["chunk_id"])
        crop_path = VISION_DIR / "crops" / f"{safe_stem}.png"
        try:
            if args.force or not crop_path.exists():
                crop_path, crop_method = make_crop(pdf_path, item["page_number"], item["caption"], crop_path)
            else:
                crop_method = "cached_crop"
            print(f"  Crop: {crop_path.relative_to(ROOT).as_posix()}")
            parsed, stats = call_vision(crop_path)
            flags = quality_flags(parsed, stats, crop_method)
            rel_image = crop_path.relative_to(ROOT).as_posix()
            text = candidate_text(item["caption"], parsed)
            record = {
                "chunk_id": item["chunk_id"],
                "paper_id": item["paper_id"],
                "paper_title": item["paper_title"],
                "source_type": "visual_description",
                "page_number": item["page_number"],
                "section": None,
                "text": text,
                "metadata": {
                    "caption": item["caption"],
                    "image_path": rel_image,
                    "source_pdf": item["source_pdf"],
                    "vision_model": VISION_MODEL,
                    "crop_method": crop_method,
                    "vision": parsed,
                    "inference": stats,
                    "quality_flags": flags,
                    "review_status": "pending_manual_review",
                    "approved_for_index": False,
                },
            }
            results[item["chunk_id"]] = record
            print(f"  Completed in {stats['total_duration_seconds']}s; flags: {flags or 'none'}")
        except (requests.RequestException, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            failures.append({"chunk_id": item["chunk_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    CANDIDATES_FILE.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in results.values()) + ("\n" if results else ""),
        encoding="utf-8",
    )
    records = list(results.values())
    report = {
        "candidate_count_total": len(candidates),
        "vision_records_saved": len(records),
        "pending_manual_review": sum(1 for r in records if r.get("metadata", {}).get("review_status") == "pending_manual_review"),
        "approved_for_index": sum(1 for r in records if r.get("metadata", {}).get("approved_for_index", False)),
        "failures_this_run": failures,
        "vision_model": VISION_MODEL,
        "output_file": CANDIDATES_FILE.relative_to(ROOT).as_posix(),
        "note": "Visual descriptions remain out of the main corpus until reviewed and approved.",
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nVISION INDEX REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
