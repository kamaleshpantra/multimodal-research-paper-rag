from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
RETRIEVAL_DIR = ROOT / "src" / "retrieval"
if str(RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_DIR))

from deployment_rag import (  # noqa: E402
    DEFAULT_CHAT_MODEL,
    INDEX_DIR,
    DeploymentRetriever,
    answer_question,
)

APP_TITLE = "Research Paper Intelligence"
DISCLAIMER = (
    "Evidence-grounded assistant for three scientific papers. Responses are generated from the indexed excerpts; "
    "citations and similarity-based audits are aids for review, not proof of entailment."
)

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
st.title("📚 Multimodal Research Paper Intelligence")
st.caption("Hybrid retrieval • inline citations • bibliography lookup • cautious numeric abstention • reviewed visual evidence")
st.write(DISCLAIMER)


def config_value(name: str, default: str = "") -> str:
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value or os.getenv(name, default)).strip()


@st.cache_resource(show_spinner="Loading the local deployment index and embedding model…")
def load_retriever() -> DeploymentRetriever:
    model = config_value("DEPLOY_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    return DeploymentRetriever.from_directory(INDEX_DIR, embedding_model=model)


def saveable_report(result: dict[str, Any]) -> str:
    lines = [
        "# Research Paper RAG — Answer Report",
        "",
        f"- Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- Question: {result.get('question', '')}",
        f"- Model: {result.get('model', '')}",
        f"- Embedding model: {result.get('embedding_model', '')}",
        "",
        "## Answer",
        "",
        str(result.get("answer", "")),
        "",
        "## Sources",
        "",
    ]
    for source in result.get("sources", []):
        lines += [
            f"### [{source.get('marker')}] {source.get('source_label')}",
            f"- Chunk ID: `{source.get('chunk_id')}`",
            f"- Paper ID: `{source.get('paper_id')}`",
            "",
            str(source.get("text", ""))[:3000],
            "",
        ]
    if result.get("paper_coverage"):
        lines += ["## Paper coverage", "", "```json", json.dumps(result["paper_coverage"], ensure_ascii=False, indent=2), "```", ""]
    if result.get("warnings"):
        lines += ["## Warnings", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    if result.get("citation_audit"):
        lines += ["## Citation audit (heuristic, not entailment)", "", "```json",
                  json.dumps(result["citation_audit"], ensure_ascii=False, indent=2), "```", ""]
    if result.get("overlap_flags"):
        lines += ["## Potential source phrase overlap (manual review required)", "", "```json",
                  json.dumps(result["overlap_flags"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


if not (INDEX_DIR / "index_manifest.json").exists():
    st.error("The deployment index is missing. Build it locally before deploying this app.")
    st.code("python src\\retrieval\\build_deployment_index.py", language="powershell")
    st.stop()

try:
    retriever = load_retriever()
except Exception as exc:
    st.error(f"Could not load the deployment index: {exc}")
    st.info("Check that data/deployment_index/ contains embeddings.npy, records.jsonl, index_manifest.json, and paper_registry.json, and that DEPLOY_EMBED_MODEL matches the build-time model.")
    st.stop()

api_key = config_value("OPENROUTER_API_KEY")
model = config_value("OPENROUTER_MODEL", DEFAULT_CHAT_MODEL)
registry = retriever.registry
paper_ids = sorted({str(r.get("paper_id")) for r in retriever.records if r.get("paper_id")})
paper_labels = {"All papers": None}
for paper_id in paper_ids:
    paper_labels[str((registry.get(paper_id) or {}).get("title") or paper_id)] = paper_id

approved_visuals = sum(1 for r in retriever.records if r.get("source_type") == "visual_description")
caption_records = sum(1 for r in retriever.records if r.get("source_type") == "figure_caption")
col1, col2, col3 = st.columns(3)
col1.metric("Indexed records", f"{len(retriever.records):,}")
col2.metric("Papers", str(len(paper_ids)))
col3.metric("Reviewed visual records", str(approved_visuals))
if caption_records > approved_visuals:
    st.warning(
        f"Visual coverage notice: {approved_visuals} approved visual description(s) are indexed while {caption_records} figure-caption record(s) exist. "
        "For figures without an approved visual description, the app may only have caption/text evidence; do not treat graph values, axes, or arrows as verified."
    )
st.caption("Academic integrity: answers are prompted to paraphrase and cite. Exact-source-overlap flags are review aids, not AI-writing detection or a guarantee of plagiarism-free output.")

with st.sidebar:
    st.header("Retrieval settings")
    selected_label = st.selectbox("Paper scope", list(paper_labels.keys()), index=0)
    top_k = st.slider("Evidence passages", min_value=4, max_value=12, value=8, step=1)
    st.caption(f"Embedding model: {config_value('DEPLOY_EMBED_MODEL', 'BAAI/bge-base-en-v1.5')}")
    st.caption(f"Generation model: {model}")
    st.divider()
    st.subheader("Privacy and free-tier notice")
    st.caption(
        "Questions, retrieved excerpts, and any approved figure crops selected for visual analysis are sent to the configured "
        "hosted inference provider. The free model/provider may log or use requests under its own policy. Do not submit personal, "
        "confidential, unpublished, or otherwise sensitive material. The demo is designed for the included public papers."
    )

st.info(
    "Use natural questions across one paper or all papers. The app supports text QA, cross-paper synthesis, exact bibliography-title "
    "lookups, figure/table questions when reviewed evidence is available, and cautious abstention when indexed evidence is insufficient. "
    "Unreviewed vision-model descriptions are excluded from the index."
)

with st.form("question_form", clear_on_submit=False):
    question = st.text_area(
        "Ask a research question",
        placeholder=(
            "Examples: Compare the central contributions of two papers; explain a figure's information flow; "
            "find an exact bibliography title; synthesize limitations reported across the selected papers."
        ),
        height=100,
        max_chars=1600,
    )
    submitted = st.form_submit_button("Retrieve evidence and answer", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    elif not api_key:
        st.error("OPENROUTER_API_KEY is not configured. Add it in Streamlit app Secrets or your local environment.")
    elif int(st.session_state.get("request_count", 0)) >= 10:
        st.warning("This browser session has reached its 10-question soft limit. It is a basic cost-control guard, not a global abuse-prevention mechanism.")
    else:
        # The counter is per browser session only; free-provider quotas remain shared across the app.
        st.session_state["request_count"] = int(st.session_state.get("request_count", 0)) + 1
        with st.spinner("Retrieving relevant passages and composing a cited answer…"):
            try:
                result = answer_question(
                    question.strip(),
                    retriever,
                    api_key=api_key,
                    model=model,
                    top_k=top_k,
                    paper_id=paper_labels[selected_label],
                )
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"The question could not be completed: {exc}")
                st.caption("Check the provider key, free-model quota, network connection, and deployment-index files.")

result = st.session_state.get("last_result")
if result:
    st.divider()
    st.subheader("Answer")
    st.markdown(result.get("answer", ""))

    coverage = result.get("paper_coverage") or {}
    if coverage.get("enabled"):
        st.caption("Cross-paper coverage enabled. Candidate pool by paper: " + ", ".join(f"{pid}: {count}" for pid, count in coverage.get("candidate_count_by_paper", {}).items()))

    warnings = result.get("warnings") or []
    if warnings:
        with st.expander(f"Reliability and review warnings ({len(warnings)})", expanded=True):
            for warning in warnings:
                st.warning(warning)

    dedup = result.get("deduplication") or {}
    st.caption(
        f"Evidence candidates: {dedup.get('raw_count', '—')} • Retained: {dedup.get('retained_count', len(result.get('sources', [])))} • "
        f"Duplicate extractions removed: {dedup.get('removed_count', 0)}"
    )

    sources = result.get("sources") or []
    st.subheader(f"Supporting sources ({len(sources)})")
    if not sources:
        st.info("No supporting passages were retrieved for this answer.")
    for source in sources:
        title = f"[{source.get('marker')}] {source.get('source_label') or source.get('chunk_id')}"
        with st.expander(title, expanded=False):
            st.caption(f"Chunk: {source.get('chunk_id')} • Paper: {source.get('paper_id')} • Type: {source.get('source_type')}")
            if source.get("retrieval"):
                st.json(source["retrieval"])
            meta = source.get("metadata") or {}
            if meta.get("reference_id") or meta.get("title"):
                st.write({k: meta.get(k) for k in ("reference_id", "title", "authors", "year", "venue") if meta.get(k) is not None})
            st.write(str(source.get("text", ""))[:5000])
            img = source.get("image_path")
            if img and Path(img).is_file():
                st.image(img, caption=f"Reviewed visual crop: {source.get('source_label')}", use_container_width=True)
            if source.get("source_type") == "figure_caption" and not img:
                st.caption("Caption evidence only; the full visual was not verified in this source record.")

    audit = result.get("citation_audit") or []
    if audit:
        with st.expander("Claim-level citation alignment audit (heuristic)", expanded=False):
            st.caption("Semantic and lexical overlap indicate topical alignment only; they do not prove that a source entails a claim.")
            st.dataframe([
                {
                    "Claim": item.get("sentence", ""),
                    "Status": item.get("status", ""),
                    "Cited source": (item.get("best_cited_source") or {}).get("source_marker"),
                    "Cited score": (item.get("best_cited_source") or {}).get("score"),
                    "Best match": (item.get("best_overall_source") or {}).get("source_marker"),
                    "Best score": (item.get("best_overall_source") or {}).get("score"),
                }
                for item in audit
            ], use_container_width=True, hide_index=True)

    overlap = result.get("overlap_flags") or []
    if overlap:
        with st.expander("Potential long-phrase overlap — manual review", expanded=True):
            st.warning("These are exact-overlap flags, not an AI detector and not a plagiarism verdict. Review quotations, technical language, and attribution manually.")
            st.json(overlap)

    st.download_button(
        "Download answer, source excerpts, and audit (.md)",
        data=saveable_report(result),
        file_name="research_rag_answer_report.md",
        mime="text/markdown",
    )

st.divider()
st.caption(
    "Research prototype · Hybrid dense + BM25/RRF retrieval · Matching local embedding model · Source markers validated · "
    "Visual evidence requires human review · No claim of guaranteed factuality or plagiarism detection."
)
