from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.hybrid_search import HybridSearch  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def retrieval_metrics(ranked_ids: list[str], relevant_ids: list[str], k: int) -> dict[str, float | None]:
    if not relevant_ids:
        return {f"hit@{k}": None, f"recall@{k}": None, f"mrr@{k}": None, f"ndcg@{k}": None}
    relevant = set(relevant_ids)
    top = ranked_ids[:k]
    found_positions = [i + 1 for i, cid in enumerate(top) if cid in relevant]
    hit = 1.0 if found_positions else 0.0
    recall = len(set(top) & relevant) / len(relevant)
    mrr = 1.0 / found_positions[0] if found_positions else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(top, start=1) if cid in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return {f"hit@{k}": hit, f"recall@{k}": recall, f"mrr@{k}": mrr, f"ndcg@{k}": ndcg}


def compact_source(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": record.get("chunk_id"),
        "paper_id": record.get("paper_id"),
        "source_type": record.get("source_type"),
        "page_number": record.get("page_number"),
        "section": record.get("section"),
        "source_label": record.get("source_label"),
        "retrieval": record.get("retrieval", {}),
        "text_excerpt": str(record.get("text", ""))[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and optionally generate answers against a labeled seed set.")
    parser.add_argument("--eval-set", default="data/evaluation/eval_set.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--generate", action="store_true", help="Also call the answer LLM and create a human-review file.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 = all).")
    args = parser.parse_args()

    eval_path = (ROOT / args.eval_set).resolve()
    if not eval_path.exists():
        raise SystemExit(f"Evaluation set not found: {eval_path}")
    cases = read_jsonl(eval_path)
    if args.limit > 0:
        cases = cases[:args.limit]

    engine = HybridSearch()
    retrieval_rows = []
    per_case_scores = []
    grouped_scores: dict[str, list[dict[str, float | None]]] = defaultdict(list)

    print(f"Evaluation cases: {len(cases)}")
    print(f"Top K: {args.top_k}")
    print(f"Embedding model: {os.getenv('EMBEDDING_MODEL', 'bge-m3:latest')}")

    answer_fn = None
    if args.generate:
        from src.generation.answer_generator import answer_question
        answer_fn = answer_question

    generation_rows = []
    manual_review_rows = []

    for pos, case in enumerate(cases, start=1):
        question = case["question"]
        paper_id = case.get("paper_id")
        results = engine.search(question, top_k=args.top_k, paper_id=paper_id)
        result_ids = [str(record.get("chunk_id", "")) for record in results]
        relevant_ids = [str(item) for item in case.get("relevant_chunk_ids", [])]
        is_answerable = bool(case.get("answerable", True))
        metrics = retrieval_metrics(result_ids, relevant_ids, args.top_k) if is_answerable else {
            f"hit@{args.top_k}": None,
            f"recall@{args.top_k}": None,
            f"mrr@{args.top_k}": None,
            f"ndcg@{args.top_k}": None,
        }
        metrics.update({
            "question_id": case.get("id", f"case_{pos}"),
            "category": case.get("category", "uncategorized"),
            "answerable": is_answerable,
        })
        per_case_scores.append(metrics)
        if is_answerable:
            grouped_scores[case.get("category", "uncategorized")].append({k: v for k, v in metrics.items() if isinstance(v, float)})

        retrieval_rows.append({
            "id": case.get("id"),
            "question": question,
            "paper_id_filter": paper_id,
            "category": case.get("category"),
            "answerable": is_answerable,
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": result_ids,
            "metrics": {k: v for k, v in metrics.items() if k.startswith(("hit@", "recall@", "mrr@", "ndcg@"))},
            "results": [compact_source(record) for record in results],
            "label_status": case.get("label_status"),
        })
        hit = metrics.get(f"hit@{args.top_k}")
        hit_label = f"Hit@{args.top_k}={hit:.0%}" if isinstance(hit, float) else "retrieval metrics skipped"
        print(f"[{pos}/{len(cases)}] {case.get('id')}: {hit_label}")

        if answer_fn:
            generated = answer_fn(
                question,
                top_k=args.top_k,
                paper_id=paper_id,
                search_engine=engine,
            )
            generation_record = {
                "id": case.get("id"),
                "question": question,
                "category": case.get("category"),
                "answerable": is_answerable,
                "expected_answer_points": case.get("expected_answer_points", []),
                **generated,
            }
            generation_rows.append(generation_record)
            manual_review_rows.append({
                "id": case.get("id"),
                "category": case.get("category"),
                "question": question,
                "answerable_expected": is_answerable,
                "expected_answer_points": " | ".join(case.get("expected_answer_points", [])),
                "answer": generated.get("answer", ""),
                "source_labels": " | ".join(s.get("source_label", "") for s in generated.get("sources", [])),
                "warnings": " | ".join(generated.get("citation_warnings", [])),
                "correctness_0_2": "",
                "faithfulness_0_2": "",
                "citation_support_0_2": "",
                "relevance_0_2": "",
                "abstention_correct_yes_no": "",
                "notes": "",
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "data" / "evaluation" / "runs" / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "retrieval_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in retrieval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metric_keys = [f"hit@{args.top_k}", f"recall@{args.top_k}", f"mrr@{args.top_k}", f"ndcg@{args.top_k}"]
    valid_answerable = [row for row in per_case_scores if row["answerable"]]
    summary = {
        "timestamp": timestamp,
        "eval_set": str(eval_path.relative_to(ROOT)),
        "case_count": len(cases),
        "answerable_cases_scored_for_retrieval": len(valid_answerable),
        "unanswerable_cases_excluded_from_retrieval_ranking_metrics": sum(not row["answerable"] for row in per_case_scores),
        "top_k": args.top_k,
        "overall_retrieval_metrics": {
            metric: mean([float(row[metric]) for row in valid_answerable if row.get(metric) is not None])
            for metric in metric_keys
        },
        "by_category": {
            category: {
                metric: mean([float(row[metric]) for row in rows if row.get(metric) is not None])
                for metric in metric_keys
            }
            for category, rows in grouped_scores.items()
        },
        "note": "Seed labels are drafts based on observed records. Verify relevant chunk IDs and add held-out manually labeled questions before presenting these scores as a final benchmark.",
        "per_case": per_case_scores,
    }
    (outdir / "evaluation_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "evaluation_report.md").write_text(render_markdown(summary), encoding="utf-8")

    if args.generate:
        with (outdir / "generation_results.jsonl").open("w", encoding="utf-8") as handle:
            for row in generation_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        review_path = outdir / "manual_review.csv"
        with review_path.open("w", newline="", encoding="utf-8-sig") as handle:
            fields = list(manual_review_rows[0].keys()) if manual_review_rows else []
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manual_review_rows)

    print("\nOverall retrieval metrics:")
    for metric in metric_keys:
        value = summary["overall_retrieval_metrics"][metric]
        print(f"  {metric}: {'N/A' if value is None else f'{value:.4f}'}")
    print(f"\nSaved report: {outdir / 'evaluation_report.md'}")
    if args.generate:
        print(f"Saved manual answer review: {outdir / 'manual_review.csv'}")


def render_markdown(summary: dict[str, Any]) -> str:
    top_k = summary["top_k"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Run: `{summary['timestamp']}`",
        f"- Evaluation cases: {summary['case_count']}",
        f"- Answerable cases used for retrieval metrics: {summary['answerable_cases_scored_for_retrieval']}",
        f"- Unanswerable cases excluded from ranking metrics: {summary['unanswerable_cases_excluded_from_retrieval_ranking_metrics']}",
        f"- Top K: {top_k}",
        "",
        "## Overall retrieval metrics",
        "",
        f"| Metric | Score |",
        f"|---|---:|",
    ]
    for metric, value in summary["overall_retrieval_metrics"].items():
        lines.append(f"| {metric} | {'N/A' if value is None else f'{value:.4f}'} |")
    lines += ["", "## By category", "", "| Category | Hit@K | Recall@K | MRR@K | nDCG@K |", "|---|---:|---:|---:|---:|"]
    for category, metrics in summary["by_category"].items():
        vals = [metrics.get(f"hit@{top_k}"), metrics.get(f"recall@{top_k}"), metrics.get(f"mrr@{top_k}"), metrics.get(f"ndcg@{top_k}")]
        lines.append("| " + category + " | " + " | ".join("N/A" if v is None else f"{v:.4f}" for v in vals) + " |")
    lines += ["", "> **Important:** These are seed-evaluation results, not a final unbiased benchmark. Verify each relevant chunk ID against the source, then add held-out questions with human-confirmed evidence labels.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
