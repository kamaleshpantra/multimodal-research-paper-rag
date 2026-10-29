import json
import re
from pathlib import Path

processed = Path("data/processed")

for path in sorted(processed.glob("*.json")):
    if path.name == "ingestion_report.json":
        continue

    data = json.loads(path.read_text(encoding="utf-8"))

    print("\n" + "=" * 78)
    print("PAPER:", path.name)
    print("TITLE:", data.get("document", {}).get("title"))
    print("REFERENCE START:", data.get("reference_section", {}).get("start_page"))

    print("\nREFERENCE-RELATED LINES:")
    found = False

    for page in data.get("pages", []):
        for line in page.get("text", "").splitlines():
            if re.search(r"references?|bibliography|works cited", line, re.I):
                print(f"  Page {page['page_number']}: {line.strip()[:180]}")
                found = True

    if not found:
        print("  No reference-related lines detected.")

    print("\nFIRST-PAGE TEXT SAMPLE:")
    first_page = data.get("pages", [{}])[0]
    print(first_page.get("text", "")[:900])

    print("\nLAST THREE PAGES:")
    for page in data.get("pages", [])[-3:]:
        print(f"\n--- Page {page['page_number']} ---")
        print(page.get("text", "")[:800])
