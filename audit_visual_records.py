from __future__ import annotations

"""Generic visual-candidate audit report. Never auto-approves model-generated descriptions."""
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "data" / "processed" / "vision" / "vision_candidates.jsonl"
REPORT = ROOT / "data" / "processed" / "vision" / "vision_quality_audit.json"


def main() -> int:
    if not CANDIDATES.exists():
        raise SystemExit(f"Visual candidate file not found: {CANDIDATES}. Run src/ingestion/vision_index.py first.")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(CANDIDATES.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at {CANDIDATES}:{line_no}: {exc}") from exc
    status_counts = Counter()
    paper_counts: dict[str, Counter] = {}
    flagged = []
    for record in records:
        meta = record.get("metadata") or {}
        status = str(meta.get("review_status") or "unreviewed")
        paper = str(record.get("paper_id") or "unknown")
        status_counts[status] += 1
        paper_counts.setdefault(paper, Counter())[status] += 1
        flags = list(meta.get("quality_flags") or [])
        if meta.get("approved_for_index") is not True or flags:
            flagged.append({
                "chunk_id": record.get("chunk_id"), "paper_id": paper,
                "page_number": record.get("page_number"), "caption": meta.get("caption"),
                "review_status": status, "approved_for_index": meta.get("approved_for_index", False),
                "quality_flags": flags,
                "uncertainties": ((meta.get("vision") or {}).get("uncertainties") or []),
                "instruction": "Inspect the original crop; approve manually only if labels, relationships, axes/units, and values are supported.",
            })
    report = {
        "candidate_count": len(records), "review_status_counts": dict(status_counts),
        "by_paper": {paper: dict(counts) for paper, counts in paper_counts.items()},
        "records_requiring_manual_review_or_flag_review": flagged,
        "note": "This report does not approve, reject, or correct visual descriptions automatically.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
