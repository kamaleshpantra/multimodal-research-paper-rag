from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_FILE = ROOT / "data" / "processed" / "corpus.jsonl"
VISION_FILE = ROOT / "data" / "processed" / "vision" / "vision_candidates.jsonl"
REPORT_FILE = ROOT / "data" / "processed" / "vision" / "merge_report.json"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    if not CORPUS_FILE.exists():
        raise SystemExit(f"Corpus not found: {CORPUS_FILE}. Run build_corpus.py first.")
    existing = read_jsonl(CORPUS_FILE)
    visual = read_jsonl(VISION_FILE)
    approved = [r for r in visual if r.get("metadata", {}).get("approved_for_index") is True]
    existing_by_id = {r.get("chunk_id"): r for r in existing}
    added_or_updated = 0
    for record in approved:
        if not record.get("chunk_id") or not record.get("text", "").strip():
            continue
        existing_by_id[record["chunk_id"]] = record
        added_or_updated += 1
    merged = list(existing_by_id.values())
    CORPUS_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n",
        encoding="utf-8",
    )
    report = {
        "base_corpus_records_before_merge": len(existing),
        "visual_records_saved": len(visual),
        "approved_visual_records": len(approved),
        "records_added_or_updated": added_or_updated,
        "merged_corpus_records": len(merged),
        "note": "If build_corpus.py is rerun, it regenerates corpus.jsonl; then rerun this merge script to add approved visual records back.",
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
