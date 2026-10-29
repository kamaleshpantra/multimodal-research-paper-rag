from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "src" else Path.cwd()
if Path(__file__).resolve().parent.name == "ingestion" and Path(__file__).resolve().parent.parent.name == "src":
    ROOT = Path(__file__).resolve().parents[2]
VISION_DIR = ROOT / "data" / "processed" / "vision"
CANDIDATES_FILE = VISION_DIR / "vision_candidates.jsonl"
REPORT_FILE = VISION_DIR / "vision_report.json"


def main() -> int:
    if not CANDIDATES_FILE.exists():
        print("No visual candidates found. Run vision_index.py first.")
        return 1

    records = []
    for line in CANDIDATES_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    pending = [
        r for r in records
        if not r.get("metadata", {}).get("approved_for_index", False)
        and not str(r.get("metadata", {}).get("review_status", "")).casefold().startswith("rejected")
    ]
    if not pending:
        print("All visual records have already been approved or rejected.")
        return 0

    print(f"Reviewing {len(pending)} visual records. Approve only if labels and relationships match the image.")
    by_id = {r["chunk_id"]: r for r in records}
    for idx, record in enumerate(pending, start=1):
        meta = record.get("metadata", {})
        vision = meta.get("vision", {})
        print("\n" + "=" * 76)
        print(f"[{idx}/{len(pending)}] {record['chunk_id']}")
        print("Paper:", record.get("paper_title"))
        print("Page:", record.get("page_number"))
        print("Caption:", meta.get("caption"))
        print("Image:", meta.get("image_path"))
        print("Type:", vision.get("figure_type"))
        print("Labels:", vision.get("visible_labels"))
        print("Relationships:", vision.get("relationships"))
        print("Chart details:", vision.get("chart_axes_and_units"))
        print("Uncertainties:", vision.get("uncertainties"))
        print("Quality flags:", meta.get("quality_flags"))
        decision = input("Approve for retrieval index? [y/N/r=reject]: ").strip().lower()
        if decision == "y":
            meta["approved_for_index"] = True
            meta["review_status"] = "approved_by_user"
            print("Approved.")
        elif decision == "r":
            meta["approved_for_index"] = False
            meta["review_status"] = "rejected_by_user"
            print("Rejected; it will remain outside the index.")
        else:
            meta["review_status"] = "pending_manual_review"
            print("Left pending.")

    CANDIDATES_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    report = {
        "visual_records": len(records),
        "approved_for_index": sum(1 for r in records if r.get("metadata", {}).get("approved_for_index", False)),
        "pending_manual_review": sum(1 for r in records if r.get("metadata", {}).get("review_status") == "pending_manual_review"),
        "rejected": sum(1 for r in records if r.get("metadata", {}).get("review_status") == "rejected_by_user"),
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nUpdated review report:")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
