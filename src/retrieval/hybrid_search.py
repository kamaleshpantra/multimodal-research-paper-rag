from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "data" / "index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
RECORDS_FILE = INDEX_DIR / "records.jsonl"
MANIFEST_FILE = INDEX_DIR / "index_manifest.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
RRF_K = 60


def tokenize(text: str) -> list[str]:
    # Simple, dependency-light BM25 tokenizer. Keep hyphenated terms as a whole and as parts.
    words = re.findall(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", text.casefold())
    tokens: list[str] = []
    for word in words:
        tokens.append(word)
        tokens.extend(part for part in re.split(r"[-./]", word) if part and part != word)
    return tokens


def load_records() -> list[dict[str, Any]]:
    if not RECORDS_FILE.exists() or not EMBEDDINGS_FILE.exists() or not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            "The retrieval index is missing. Run: python src\\retrieval\\build_index.py"
        )
    records = []
    with RECORDS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def embed_query(query: str) -> np.ndarray:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": [query], "truncate": True},
        timeout=(15, 120),
    )
    response.raise_for_status()
    vectors = response.json().get("embeddings", [])
    if len(vectors) != 1:
        raise RuntimeError(f"Expected one query embedding, received {len(vectors)}")
    vector = np.asarray(vectors[0], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise RuntimeError("Query embedding has zero norm.")
    return vector / norm


class HybridSearch:
    def __init__(self) -> None:
        self.records = load_records()
        self.embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False)
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

        if manifest.get("embedding_model") != EMBEDDING_MODEL:
            raise RuntimeError(
                "Embedding model mismatch: index was built with "
                f"'{manifest.get('embedding_model')}', query config uses '{EMBEDDING_MODEL}'. "
                "Set EMBEDDING_MODEL to the same model and rebuild the index if needed."
            )
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.records):
            raise RuntimeError(
                f"Index mismatch: embeddings shape {self.embeddings.shape}, "
                f"records count {len(self.records)}. Rebuild the index."
            )
        if self.embeddings.shape[1] != int(manifest.get("embedding_dimensions", -1)):
            raise RuntimeError("Embedding dimension does not match index manifest.")

        self.bm25_tokens = [tokenize(record.get("text", "")) for record in self.records]
        self.bm25 = BM25Okapi(self.bm25_tokens)

    def search(self, query: str, top_k: int = 8, paper_id: str | None = None) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = embed_query(query)
        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise RuntimeError(
                f"Query vector dimension {query_vector.shape[0]} does not match "
                f"index dimension {self.embeddings.shape[1]}"
            )
        dense_scores = self.embeddings @ query_vector
        lexical_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

        allowed = [
            index for index, record in enumerate(self.records)
            if paper_id is None or record.get("paper_id") == paper_id
        ]
        if not allowed:
            return []

        candidate_n = min(len(allowed), max(top_k * 5, 30))
        dense_ranked = sorted(allowed, key=lambda index: float(dense_scores[index]), reverse=True)[:candidate_n]
        lexical_ranked = sorted(allowed, key=lambda index: float(lexical_scores[index]), reverse=True)[:candidate_n]

        rrf_scores: dict[int, float] = {}
        for ranking in (dense_ranked, lexical_ranked):
            for rank, index in enumerate(ranking, start=1):
                rrf_scores[index] = rrf_scores.get(index, 0.0) + 1.0 / (RRF_K + rank)

        # When the user names an explicit visual number (e.g., "Figure 1"),
        # directly include matching visual/caption records even if dense/BM25
        # retrieval omitted them from the candidate pools. This is a metadata
        # match, not a semantic relevance guarantee; the answer should still cite
        # and display the matched evidence for user verification.
        requested_visual = re.search(
            r"\b(figure|fig\.?|table|chart|graph)\s*(\d+(?:\.\d+)*)\b",
            query.casefold(),
        )
        exact_visual_matches: list[int] = []
        if requested_visual:
            requested_kind = requested_visual.group(1).replace(".", "")
            requested_num = requested_visual.group(2)
            if requested_kind == "fig":
                requested_kind = "figure"
            marker_pattern = re.compile(
                rf"^\s*{re.escape(requested_kind)}\s*{re.escape(requested_num)}\b",
                re.IGNORECASE,
            )
            for index in allowed:
                record = self.records[index]
                source_type = record.get("source_type", "")
                if source_type not in {"visual_description", "figure_caption", "table"}:
                    continue
                metadata = record.get("metadata") or {}
                candidate_caption = str(metadata.get("caption") or "")
                if not candidate_caption:
                    candidate_caption = str(record.get("text") or "")
                if marker_pattern.search(candidate_caption):
                    exact_visual_matches.append(index)
            if exact_visual_matches:
                current_best = max(rrf_scores.values(), default=(1.0 / (RRF_K + 1)))
                # Promote exact metadata matches into the rankable pool and put
                # them ahead of generic passages for explicitly numbered visuals.
                for index in exact_visual_matches:
                    rrf_scores[index] = max(
                        rrf_scores.get(index, 0.0), current_best * 1.10
                    )

        # Source-aware weighting: conceptual questions should favor article text,
        # while explicit bibliography/citation and visual/table questions should
        # favor their corresponding record types. This is a ranking heuristic,
        # not a guarantee of relevance; evaluate it against your question set.
        query_lower = query.casefold()
        citation_intent = (
            any(term in query_lower for term in (
                "reference", "bibliography", "doi", "who wrote", "authors of",
                "publication year", "which study", "which studies are cited",
                "cited by", "citation marker", "why does", "why is", "cites",
                "cited in", "citation context", "reference list", "bibliographic",
            ))
            or bool(re.search(
                r"\b(?:which|what|name|list|identify|find)\b.{0,60}"
                r"\b(?:earlier\s+)?(?:stud(?:y|ies)|papers?|works|authors|references)\b.{0,60}"
                r"\b(?:are\s+)?(?:cited|referenced|mentioned)\b",
                query_lower,
            ))
        )
        visual_intent = any(term in query_lower for term in (
            "figure", "fig.", "diagram", "graph", "plot", "chart",
            "architecture diagram", "illustrated", "shown in",
        ))
        table_intent = any(term in query_lower for term in (
            "table", "tabulated", "metric", "metrics", "results", "performance",
            "highest", "lowest", "best performing", "compare scores",
        ))

        def type_weight(record: dict[str, Any]) -> float:
            source_type = record.get("source_type", "unknown")
            metadata = record.get("metadata") or {}
            vision = metadata.get("vision") or {}
            vision_type = str(vision.get("figure_type") or "").casefold()

            # If a specific Figure/Table number was requested, distinguish a
            # matching visual record from unrelated visual records in the same
            # paper. Textual explanation remains eligible for normal ranking.
            if requested_visual and source_type in {
                "visual_description", "figure_caption", "table"
            }:
                requested_kind = requested_visual.group(1).replace(".", "")
                if requested_kind == "fig":
                    requested_kind = "figure"
                requested_num = requested_visual.group(2)
                candidate_caption = str(metadata.get("caption") or record.get("text") or "")
                caption_match = re.match(
                    r"\s*(figure|fig\.?|table|chart|graph)\s*([0-9]+(?:\.[0-9]+)*)\b",
                    candidate_caption,
                    re.IGNORECASE,
                )
                if caption_match:
                    candidate_kind = caption_match.group(1).replace(".", "").casefold()
                    if candidate_kind == "fig":
                        candidate_kind = "figure"
                    is_exact = (
                        candidate_kind == requested_kind
                        and caption_match.group(2) == requested_num
                    )
                    if not is_exact:
                        return 0.45 if source_type == "visual_description" else 0.65
                elif source_type == "visual_description":
                    return 0.60

            if citation_intent:
                weights = {
                    "citation_context": 1.35,
                    "bibliography_reference": 1.25,
                    "section_text": 1.10,
                    "page_text": 0.95,
                    "abstract": 0.95,
                    "figure_caption": 0.65,
                    "table": 0.75,
                    "visual_description": 0.35,
                }
            elif visual_intent:
                weights = {
                    # Visual descriptions were previously omitted here and fell
                    # back to 0.80, causing the exact Figure 1 record to rank below
                    # generic passages. Prefer reviewed visual evidence on visual asks.
                    "visual_description": 1.90,
                    "figure_caption": 1.20,
                    "table": 0.85,
                    "section_text": 1.05,
                    "page_text": 0.95,
                    "abstract": 0.75,
                    "citation_context": 0.55,
                    "bibliography_reference": 0.30,
                }
            elif table_intent:
                weights = {
                    "table": 1.35,
                    "section_text": 1.20,
                    "page_text": 1.10,
                    "figure_caption": 0.90,
                    "abstract": 0.85,
                    "citation_context": 0.65,
                    "bibliography_reference": 0.40,
                    "visual_description": 1.50 if vision_type == "table" else 0.75,
                }
            else:
                weights = {
                    "section_text": 1.40,
                    "abstract": 1.20,
                    "page_text": 1.00,
                    "citation_context": 0.95,
                    "figure_caption": 0.85,
                    "table": 0.90,
                    "bibliography_reference": 0.50,
                    "visual_description": 0.65,
                }
            return weights.get(source_type, 0.80)

        adjusted_scores = {
            index: float(score) * type_weight(self.records[index])
            for index, score in rrf_scores.items()
        }
        final_ranked = sorted(adjusted_scores, key=adjusted_scores.get, reverse=True)[:top_k]
        results = []
        for index in final_ranked:
            record = dict(self.records[index])
            metadata = record.get("metadata") or {}
            record["retrieval"] = {
                "rrf_score": round(float(rrf_scores[index]), 6),
                "source_weight": round(float(type_weight(record)), 3),
                "adjusted_score": round(float(adjusted_scores[index]), 6),
                "dense_cosine": round(float(dense_scores[index]), 6),
                "bm25_score": round(float(lexical_scores[index]), 6),
            }
            record["source_label"] = self.format_source(record, metadata)
            results.append(record)
        return results

    @staticmethod
    def format_source(record: dict[str, Any], metadata: dict[str, Any]) -> str:
        paper = record.get("paper_id", "unknown paper")
        source_type = record.get("source_type", "source")
        page = record.get("page_number")
        section = record.get("section")
        ref_title = metadata.get("title")
        ref_id = metadata.get("reference_id")
        parts = [paper, source_type]
        if page is not None:
            parts.append(f"page {page}")
        if section:
            parts.append(f"section {section}")
        if ref_id:
            parts.append(f"reference {ref_id}")
        if ref_title:
            parts.append(str(ref_title))
        if source_type == "visual_description":
            caption = metadata.get("caption")
            vision = metadata.get("vision") or {}
            visual_type = vision.get("figure_type")
            if visual_type:
                parts.append(f"visual type {visual_type}")
            if caption:
                caption_match = re.match(r"\s*(?:figure|fig\.?|table|chart|graph)\s*[A-Z]?\d+(?:\.\d+)*", str(caption), re.IGNORECASE)
                if caption_match:
                    parts.append(caption_match.group(0).strip())
        return " | ".join(str(part) for part in parts)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python src\\retrieval\\hybrid_search.py "your question here" [top_k] [paper_id]')
        raise SystemExit(2)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) >= 3 else 8
    paper_id = sys.argv[3] if len(sys.argv) >= 4 else None
    engine = HybridSearch()
    results = engine.search(query, top_k=top_k, paper_id=paper_id)

    print(f"\nQUERY: {query}\nRESULTS: {len(results)}\n")
    for position, record in enumerate(results, start=1):
        print(f"[{position}] {record['source_label']}")
        print(f"    chunk_id: {record.get('chunk_id')}")
        print(f"    scores: {record['retrieval']}")
        text = re.sub(r"\s+", " ", record.get("text", ""))
        print(f"    text: {text[:900]}{'...' if len(text) > 900 else ''}\n")


if __name__ == "__main__":
    main()
