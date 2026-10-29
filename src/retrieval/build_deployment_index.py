from __future__ import annotations

"""Build a Streamlit-friendly deployment index from the existing corpus.jsonl.

Uses a lightweight local FastEmbed model so the hosted app never calls localhost
or requires paid embedding API calls. This creates data/deployment_index separately;
it does not overwrite the current Ollama/BGE-M3 experiment index in data/index.
"""

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
GROBID_DIR = ROOT / "data" / "processed" / "grobid"
OUT_DIR = ROOT / "data" / "deployment_index"
MODEL_NAME = os.getenv("DEPLOY_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
BATCH_SIZE = max(4, int(os.getenv("DEPLOY_EMBED_BATCH_SIZE", "32")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id or not str(item.get("text", "")).strip():
                raise ValueError(f"Invalid corpus record at line {line_no}: chunk_id/text is missing")
            if chunk_id in seen:
                raise ValueError(f"Duplicate chunk_id at line {line_no}: {chunk_id}")
            seen.add(chunk_id)
            records.append(item)
    return records


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(matrix).all():
        raise ValueError("Cannot normalize zero or non-finite embeddings.")
    return matrix / norms


def approved_record(record: dict[str, Any]) -> bool:
    # Preserve captions and extracted tables, but never index unreviewed vision-model descriptions.
    if record.get("source_type") != "visual_description":
        return True
    meta = record.get("metadata") or {}
    status = str(meta.get("review_status", "")).casefold()
    # Require explicit machine-readable human approval; status text alone is not enough.
    return meta.get("approved_for_index") is True


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug[:100] or "visual_asset"


def stage_asset(record: dict[str, Any]) -> None:
    meta = record.setdefault("metadata", {})
    if record.get("source_type") != "visual_description":
        return
    original = meta.get("image_path")
    if not original:
        return
    original_path = (ROOT / str(original)).resolve()
    try:
        original_path.relative_to(ROOT.resolve())
    except ValueError:
        print(f"WARN: ignoring image outside project root for {record.get('chunk_id')}")
        return
    if not original_path.exists() or not original_path.is_file():
        print(f"WARN: missing reviewed visual crop for {record.get('chunk_id')}: {original_path}")
        return
    asset_dir = OUT_DIR / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    target = asset_dir / f"{safe_slug(str(record.get('chunk_id')))}.jpg"
    try:
        from PIL import Image
        with Image.open(original_path) as image:
            image = image.convert("RGB")
            image.thumbnail((1400, 1400))
            image.save(target, format="JPEG", quality=82, optimize=True)
    except Exception:
        # Keep the actual file format if Pillow conversion isn't available.
        target = asset_dir / f"{safe_slug(str(record.get('chunk_id')))}{original_path.suffix.lower()}"
        shutil.copy2(original_path, target)
    meta["deployment_asset_path"] = target.relative_to(ROOT).as_posix()


def build_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    if not GROBID_DIR.exists():
        return registry
    for path in sorted(GROBID_DIR.glob("*.grobid.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        paper_id = str(data.get("paper_id") or path.name.removesuffix(".grobid.json"))
        doc = data.get("document") or {}
        registry[paper_id] = {
            "title": str(doc.get("title") or paper_id),
            "authors": doc.get("authors") or [],
            "year": doc.get("year"),
            "source_metadata": path.name,
        }
    return registry


def main() -> int:
    if not CORPUS_PATH.exists():
        raise SystemExit(f"Corpus missing: {CORPUS_PATH}. Run python src\\retrieval\\build_corpus.py and merge approved visual records first.")
    all_records = read_jsonl(CORPUS_PATH)
    records = [dict(r) for r in all_records if approved_record(r)]
    if not records:
        raise SystemExit("No approved searchable records found in corpus.jsonl")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Corpus records: {len(all_records)}")
    print(f"Records retained (unreviewed/rejected visual descriptions excluded): {len(records)}")
    print(f"Embedding model: {MODEL_NAME}")
    print("Loading FastEmbed model. The first run downloads the small ONNX model once...")
    from fastembed import TextEmbedding
    embedder = TextEmbedding(model_name=MODEL_NAME)

    texts = [str(r["text"])[:6000] for r in records]
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        vector_batch = np.asarray(list(embedder.embed(batch)), dtype=np.float32)
        if vector_batch.ndim != 2 or vector_batch.shape[0] != len(batch):
            raise RuntimeError(f"Unexpected FastEmbed output shape {vector_batch.shape} for batch size {len(batch)}")
        batches.append(vector_batch)
        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} records")
    matrix = normalize_rows(np.vstack(batches))

    # Asset paths are staged after embeddings, but before the records JSON is saved.
    for record in records:
        stage_asset(record)

    tmp_vectors = OUT_DIR / "embeddings.tmp.npy"
    tmp_records = OUT_DIR / "records.tmp.jsonl"
    np.save(tmp_vectors, matrix)
    with tmp_records.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_vectors.replace(OUT_DIR / "embeddings.npy")
    tmp_records.replace(OUT_DIR / "records.jsonl")

    counts = Counter(str(r.get("source_type") or "unknown") for r in records)
    paper_counts = Counter(str(r.get("paper_id") or "unknown") for r in records)
    manifest = {
        "schema_version": "1.1",
        "embedding_provider": "fastembed-local-onnx",
        "embedding_model": MODEL_NAME,
        "embedding_dimensions": int(matrix.shape[1]),
        "record_count": len(records),
        "dtype": str(matrix.dtype),
        "normalized_for_cosine_similarity": True,
        "document_embedding_instruction": "none",
        "query_embedding_instruction": "Represent this sentence for searching relevant passages: ",
        "source_type_counts": dict(sorted(counts.items())),
        "paper_record_counts": dict(sorted(paper_counts.items())),
        "records_file": "records.jsonl",
        "embeddings_file": "embeddings.npy",
        "notes": [
            "This deployment index is separate from data/index and must be queried with the same FastEmbed model.",
            "Only reviewed visual-description records are retained; caption-only records remain explicitly caption-only.",
            "The text corpus and image crops are sourced from the local research-paper corpus; check redistribution rights before publishing them in a public repository.",
        ],
    }
    (OUT_DIR / "index_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "paper_registry.json").write_text(json.dumps(build_registry(), indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDEPLOYMENT INDEX REPORT")
    print(json.dumps({
        "record_count": len(records), "embedding_dimensions": int(matrix.shape[1]),
        "embedding_model": MODEL_NAME, "papers": dict(sorted(paper_counts.items())),
        "source_types": dict(sorted(counts.items())), "output": str(OUT_DIR),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
