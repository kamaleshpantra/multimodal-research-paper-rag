from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
CORPUS_FILE = ROOT / "data" / "processed" / "corpus.jsonl"
INDEX_DIR = ROOT / "data" / "index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
RECORDS_FILE = INDEX_DIR / "records.jsonl"
MANIFEST_FILE = INDEX_DIR / "index_manifest.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "24")))
REQUEST_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "180"))


def load_corpus(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {path}. Run src\\retrieval\\build_corpus.py first."
        )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = record.get("chunk_id")
            text = str(record.get("text", "")).strip()
            if not chunk_id or not text:
                raise ValueError(f"Record on line {line_number} has no chunk_id or text.")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in corpus: {chunk_id}")
            seen_ids.add(chunk_id)
            records.append(record)
    if not records:
        raise ValueError(f"No searchable records found in {path}")
    return records


def request_embeddings(texts: list[str]) -> np.ndarray:
    endpoint = f"{OLLAMA_BASE_URL}/api/embed"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(
                endpoint,
                json={"model": EMBEDDING_MODEL, "input": texts, "truncate": True},
                timeout=(15, REQUEST_TIMEOUT),
            )
            response.raise_for_status()
            payload = response.json()
            vectors = payload.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(texts):
                raise ValueError(
                    f"Embedding API returned {len(vectors) if isinstance(vectors, list) else 'invalid'} "
                    f"vectors for {len(texts)} inputs."
                )
            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(texts):
                raise ValueError(f"Unexpected embedding matrix shape: {matrix.shape}")
            if not np.isfinite(matrix).all():
                raise ValueError("Embedding response contains NaN or infinite values.")
            return matrix
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(
        f"Embedding request failed after 3 attempts using model '{EMBEDDING_MODEL}' "
        f"at {endpoint}: {last_error}"
    )


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    records = load_corpus(CORPUS_FILE)
    all_vectors: list[np.ndarray] = []
    texts = [str(record["text"]) for record in records]

    print(f"Corpus records: {len(records)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_BASE_URL}")
    print(f"Batch size: {BATCH_SIZE}")

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors = request_embeddings(batch)
        all_vectors.append(vectors)
        completed = min(start + len(batch), len(texts))
        print(f"Embedded {completed}/{len(texts)} records")

    matrix = np.vstack(all_vectors).astype(np.float32, copy=False)
    matrix = normalize_rows(matrix).astype(np.float32, copy=False)

    # Write temporary files first so a partially completed build is not mistaken for a valid index.
    temp_embeddings = INDEX_DIR / "embeddings.tmp.npy"
    temp_records = INDEX_DIR / "records.tmp.jsonl"
    np.save(temp_embeddings, matrix)
    with temp_records.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    temp_embeddings.replace(EMBEDDINGS_FILE)
    temp_records.replace(RECORDS_FILE)

    source_counts = Counter(record.get("source_type", "unknown") for record in records)
    paper_counts = Counter(record.get("paper_id", "unknown") for record in records)
    manifest = {
        "schema_version": "1.0",
        "embedding_provider": "ollama",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": int(matrix.shape[1]),
        "record_count": len(records),
        "dtype": str(matrix.dtype),
        "normalized_for_cosine_similarity": True,
        "source_type_counts": dict(sorted(source_counts.items())),
        "paper_record_counts": dict(sorted(paper_counts.items())),
        "records_file": "records.jsonl",
        "embeddings_file": "embeddings.npy",
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nIndex build complete:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nSaved index to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
