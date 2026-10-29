import importlib.util
import math
import sys
import types
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
script_path = project_root / "src" / "generation" / "answer_generator.py"

stub = types.ModuleType("hybrid_search")
class HybridSearch:
    def __init__(self, *args, **kwargs):
        pass
    def search(self, *args, **kwargs):
        return []
stub.HybridSearch = HybridSearch
sys.modules["hybrid_search"] = stub
spec = importlib.util.spec_from_file_location("answer_generator_under_test", script_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def unavailable(*args, **kwargs):
    raise mod.requests.ConnectionError("Ollama disabled in unit test")
mod.requests.post = unavailable

evidence = [
    {"paper_id": "foundational_rag", "source_type": "section_text", "chunk_id": "rag-methods",
     "text": "The retriever uses a query encoder and document index. Maximum Inner Product Search (MIPS) selects the top passages for the generator. The generator conditions on query, retrieved passages, and previous output tokens."},
    {"paper_id": "recommendation_xrec", "source_type": "section_text", "chunk_id": "xrec-methods",
     "text": "The recommendation system uses collaborative signals and a large language model to produce explanations for recommended items."},
]

answer = (
    "The retriever uses MIPS to select relevant passages from the document index [S1].\n"
    "XRec uses collaborative signals to explain recommended items [S2]."
)
audit = mod.audit_inline_citations(answer, evidence)
assert len(audit) == 2, f"Expected two claim audit rows, got {len(audit)}"
assert all(row["citation_markers"] for row in audit), "Existing citations were not parsed"
assert all(row["status"] == "plausible_match" for row in audit), audit

wrong = mod.audit_inline_citations(
    "The retriever uses MIPS to select relevant passages from the document index [S2].", evidence
)
assert wrong[0]["status"] == "weak_match_review", wrong

# Embedding borderline test: old threshold (0.40) accepted this; new threshold (0.48) flags it.
class FakeResponse:
    def raise_for_status(self):
        pass
    def json(self):
        sim = 0.58
        return {"embeddings": [[1.0, 0.0], [sim, math.sqrt(1.0 - sim * sim)]]}
def fake_embed(*args, **kwargs):
    return FakeResponse()
mod.requests.post = fake_embed
borderline = mod.audit_inline_citations(
    "This architecture improves quality of generated responses using retrieved external knowledge [S1].",
    [{"paper_id": "foundational_rag", "source_type": "section_text", "chunk_id": "weak-support",
      "text": "The architecture supplies external document context to the generator."}],
)
assert len(borderline) == 1, borderline
assert borderline[0]["status"] == "weak_match_review", borderline
assert borderline[0]["best_cited_source"]["score"] < 0.48, borderline

mod.requests.post = unavailable
cleaned, used, invalid = mod.validate_source_markers(
    "The retriever uses MIPS to find documents [S1, S2].", source_count=2
)
assert cleaned.endswith("[S1, S2]."), cleaned
assert used == ["S1", "S2"] and not invalid, (used, invalid)
cleaned, used, invalid = mod.validate_source_markers(
    "The retriever uses MIPS to find documents [S1, S9].", source_count=2
)
assert "[S1]" in cleaned and "S9" not in cleaned, cleaned
assert used == ["S1"] and invalid == [9], (used, invalid)
uncited = mod.audit_inline_citations(
    "The retriever uses MIPS to select passages from documents.", evidence
)
assert len(uncited) == 1 and uncited[0]["status"] == "uncited", uncited
print("PASS: existing citations, mismatch, borderline support, and marker validation tests passed.")

# Duplicate extraction records are removed, preferring cleaner section text over page text.
shared = (
    "The retriever encodes the input query and compares it with passages in the document index "
    "using maximum inner product search to select the top passages. The generator then conditions "
    "on the input query, the retrieved passages, and previous output tokens to generate the answer. "
    "These components form the retrieval augmented generation architecture described in the Methods section."
)
page = {
    "paper_id": "foundational_rag", "source_type": "page_text", "chunk_id": "page-copy",
    "text": shared + " Code to run experiments with RAG has been open-sourced as additional page footer noise."
}
section = {
    "paper_id": "foundational_rag", "source_type": "section_text", "chunk_id": "section-clean",
    "text": shared
}
deduped, removed = mod.deduplicate_retrieval_results([page, section])
assert len(deduped) == 1, (deduped, removed)
assert deduped[0]["chunk_id"] == "section-clean", deduped
assert len(removed) == 1 and removed[0]["removed_chunk_id"] == "page-copy", removed

# Regression test for a page extraction that repeats a structured section then adds footer noise.
actual_style_shared = (
    "We explore RAG models, which use the input sequence x to retrieve text documents z and use them "
    "as additional context when generating the target sequence y. As shown in Figure 1, our models "
    "leverage two components: a retriever with parameters that returns distributions over text passages "
    "given a query and a generator that produces a token based on the query, retrieved text, and previous tokens. "
    "This is a sufficiently long Methods passage to be eligible for cross-pipeline deduplication."
)
page_actual = {
    "paper_id": "foundational_rag", "source_type": "page_text", "chunk_id": "p2-page-copy",
    "text": actual_style_shared + " 1Code to run experiments with RAG has been open-sourced as footer noise."
}
section_actual = {
    "paper_id": "foundational_rag", "source_type": "section_text", "chunk_id": "methods-section-copy",
    "text": actual_style_shared
}
deduped_actual, removed_actual = mod.deduplicate_retrieval_results([page_actual, section_actual])
assert len(deduped_actual) == 1, (deduped_actual, removed_actual)
assert deduped_actual[0]["chunk_id"] == "methods-section-copy", deduped_actual
assert removed_actual and removed_actual[0]["removed_chunk_id"] == "p2-page-copy", removed_actual

# Related, but not near-duplicate, evidence should be retained.
related = {
    "paper_id": "foundational_rag", "source_type": "section_text", "chunk_id": "different-section",
    "text": "The evaluation studies open domain question answering and factual verification across tasks. "
            "It reports separate retrieval and generation experiments with multiple baselines and metrics."
}
deduped_related, removed_related = mod.deduplicate_retrieval_results([section, related])
assert len(deduped_related) == 2 and not removed_related, (deduped_related, removed_related)

# Keep distinct visual evidence even when related wording overlaps text extraction.
visual = {"paper_id": "foundational_rag", "source_type": "visual_description", "chunk_id": "figure",
          "text": shared}
deduped_visual, removed_visual = mod.deduplicate_retrieval_results([section, visual])
assert len(deduped_visual) == 2 and not removed_visual, (deduped_visual, removed_visual)
print("PASS: near-duplicate extraction is removed; related text and visual evidence stay distinct.")

assert mod.LLM_MODEL == "qwen2.5-coder:7b", mod.LLM_MODEL
# Scientific notation and a small footer can differ while retaining most 5-token shingles.
section_scientific = {"paper_id":"foundational_rag", "source_type":"section_text", "chunk_id":"methods-structured",
    "text":"We explore RAG models which use the input sequence x to retrieve text documents z and use them as additional context when generating the target sequence y. As shown in Figure 1 our models leverage two components a retriever with parameters that returns top k distributions over text passages given a query and a generator that generates a current token based on the previous output tokens the original input and a retrieved passage."}
page_scientific = {"paper_id":"foundational_rag", "source_type":"page_text", "chunk_id":"methods-page-copy",
    "text":"We explore RAG models which use the input sequence x to retrieve text documents z and use them as additional context when generating the target sequence y. As shown in Figure 1 our models leverage two components a retriever with parameters that returns top k distributions over text passages given a query and a generator that generates a current token based on the previous output tokens the original input and a retrieved passage. Code to run experiments with RAG has been open sourced as footer noise."}
deduped_scientific, removed_scientific = mod.deduplicate_retrieval_results([section_scientific, page_scientific])
assert len(deduped_scientific) == 1, (deduped_scientific, removed_scientific)
assert deduped_scientific[0]["chunk_id"] == "methods-structured", deduped_scientific
assert removed_scientific, removed_scientific
print("PASS: v5 model default and extraction-noise deduplication tests passed.")

visual_overlap_only = mod.exact_overlap_flags(
    "The retriever uses MIPS against the document index to select top-K text passages.",
    [{"paper_id":"foundational_rag", "source_type":"visual_description", "chunk_id":"reviewed-figure",
      "text":"The retriever uses MIPS against the document index to select top-K text passages and the generator produces output."}],
)
assert visual_overlap_only == [], visual_overlap_only
print("PASS: visual-description labels are not misreported as source-prose overlap.")


# Regression for the real diagnostic: 70-token section text and 98-token page text
# have 0.70 sequence containment, 0.7857 set containment and about 0.6818 shingle
# containment. Section text must win, while merely related passages stay separate.
section_tokens = [f"sectiontoken{i}" for i in range(70)]
page_tokens = (
    section_tokens[:49]
    + [f"pagetail{i}" for i in range(20)]
    + section_tokens[60:66]
    + [f"pagefooter{i}" for i in range(23)]
)
real_diagnostic_section = {
    "paper_id": "foundational_rag", "source_type": "section_text",
    "chunk_id": "foundational_rag:grobid_section:6:chunk1", "text": " ".join(section_tokens)
}
real_diagnostic_page = {
    "paper_id": "foundational_rag", "source_type": "page_text",
    "chunk_id": "foundational_rag:p2:text5", "text": " ".join(page_tokens)
}
deduped_diagnostic, removed_diagnostic = mod.deduplicate_retrieval_results(
    [real_diagnostic_section, real_diagnostic_page]
)
assert len(deduped_diagnostic) == 1, (deduped_diagnostic, removed_diagnostic)
assert deduped_diagnostic[0]["chunk_id"] == "foundational_rag:grobid_section:6:chunk1", deduped_diagnostic
assert len(removed_diagnostic) == 1, removed_diagnostic
assert removed_diagnostic[0]["removed_chunk_id"] == "foundational_rag:p2:text5", removed_diagnostic
print("PASS: the reported 70-vs-98-token cross-pipeline duplicate pattern is removed, preferring section_text.")


# Guard against generic performance claims when their cited evidence is weak.
weak_performance_answer = (
    "This architecture allows the model to leverage external knowledge from documents "
    "to improve the quality of generated responses [S3].\n\n"
    "The retriever selects passages for use by the generator [S1]."
)
weak_performance_audit = [{
    "sentence": "This architecture allows the model to leverage external knowledge from documents to improve the quality of generated responses [S3].",
    "status": "weak_match_review",
    "citation_markers": ["S3"],
    "best_cited_source": {"source_marker": "S3", "score": 0.452, "lexical_coverage": 0.2},
}]
clean_answer, removed_claims = mod.remove_weakly_supported_performance_claims(
    weak_performance_answer, weak_performance_audit
)
assert "improve the quality" not in clean_answer, clean_answer
assert "The retriever selects passages" in clean_answer, clean_answer
assert len(removed_claims) == 1, removed_claims
print("PASS: weakly supported performance/benefit claim is removed and unrelated sentence is retained.")

# A strongly aligned performance claim is left intact for citation and manual review.
strong_audit = [{
    "sentence": "The method improves accuracy in the reported evaluation [S1].",
    "status": "plausible_match",
    "citation_markers": ["S1"],
    "best_cited_source": {"source_marker": "S1", "score": 0.82, "lexical_coverage": 0.72},
}]
strong_answer = "The method improves accuracy in the reported evaluation [S1]."
strong_clean, strong_removed = mod.remove_weakly_supported_performance_claims(strong_answer, strong_audit)
assert strong_clean == strong_answer and not strong_removed, (strong_clean, strong_removed)
print("PASS: supported or non-weak performance claims are not removed.")

# Deterministic bibliography title lookup: exact title beats a similarly named entry.
reference_question = (
    "Which earlier work in the XRec bibliography is titled "
    "Personalized prompt learning for explainable recommendation?"
)
reference_evidence = [
    {"paper_id": "recommendation_xrec", "source_type": "bibliography_reference", "chunk_id": "ref-b11",
     "metadata": {"reference_id": "b11", "title": "Representation learning with large language models for recommendation"},
     "text": "Representation learning with large language models for recommendation"},
    {"paper_id": "recommendation_xrec", "source_type": "bibliography_reference", "chunk_id": "ref-b7",
     "metadata": {"reference_id": "b7", "title": "Personalized prompt learning for explainable recommendation"},
     "text": "Personalized prompt learning for explainable recommendation"},
    {"paper_id": "recommendation_xrec", "source_type": "bibliography_reference", "chunk_id": "ref-b6",
     "metadata": {"reference_id": "b6", "title": "Personalized transformer for explainable recommendation"},
     "text": "Personalized transformer for explainable recommendation"},
]
lookup = mod.deterministic_bibliography_title_lookup(reference_question, reference_evidence)
assert lookup is not None and lookup["mode"] == "exact_match", lookup
assert lookup["matched_title"] == "Personalized prompt learning for explainable recommendation", lookup
assert lookup["answer"].endswith("[S2]."), lookup
assert "Personalized transformer" not in lookup["answer"], lookup
assert mod.deterministic_bibliography_title_lookup(
    "How does XRec generate explainable recommendations?", reference_evidence
) is None
print("PASS: exact bibliography title lookup chooses the matched entry over a similar title.")

# Integration guard: exact bibliography title lookup must bypass LLM generation.
class FakeSearchEngine:
    def search(self, question, top_k=8, paper_id=None):
        return reference_evidence

original_call_ollama = mod.call_ollama
def unexpected_llm_call(prompt):
    raise AssertionError("LLM should not run for exact bibliography title lookup")
mod.call_ollama = unexpected_llm_call
integrated = mod.answer_question(
    reference_question, top_k=8, paper_id="recommendation_xrec", search_engine=FakeSearchEngine()
)
mod.call_ollama = original_call_ollama
assert integrated["answer"].endswith("[S2]."), integrated["answer"]
assert integrated["bibliography_lookup"]["mode"] == "exact_match", integrated.get("bibliography_lookup")
assert integrated["sources"][1]["chunk_id"] == "ref-b7", integrated["sources"]
print("PASS: end-to-end exact bibliography lookup skips LLM generation and cites the exact record.")

# Exact-title lookup with no match abstains rather than substituting a similar title.
not_found_question = "Which XRec bibliography entry is titled A Completely Unindexed Work Name?"
not_found = mod.deterministic_bibliography_title_lookup(not_found_question, reference_evidence)
assert not_found is not None and not_found["mode"] == "no_exact_match_in_retrieved_references", not_found
assert "will not substitute a similarly named work" in not_found["answer"], not_found
print("PASS: missing exact bibliography title yields abstention rather than fuzzy substitution.")


# Full-index exact numeric-parameter scan: don't mistake a generic discussion or a year for a coefficient.
sae_question = "What exact SAE sparsity coefficient was used in the experiments described by the papers in this library?"
no_value_records = [
    {"paper_id": "llm_interpretability", "source_type": "section_text", "chunk_id": "sae-context-1",
     "section": "Improve sparsity reconstruction trade-off",
     "text": "Sparse autoencoders (SAEs) balance reconstruction loss and sparsity loss. These objectives are balanced by the L1 coefficient lambda, but the extracted passage does not show a numeric setting."},
    {"paper_id": "llm_interpretability", "source_type": "page_text", "chunk_id": "sae-context-2",
     "section": "SAE review", "text": "In 2024, the review discussed SAE approaches and the sparsity-reconstruction trade-off."},
]
no_value_hits, no_value_scan = mod.scan_index_for_numeric_parameter_evidence(sae_question, no_value_records)
assert no_value_scan["available"] and no_value_scan["records_scanned"] == 2, no_value_scan
assert not no_value_hits and no_value_scan["candidate_count"] == 0, (no_value_hits, no_value_scan)
print("PASS: full-index parameter scan does not mistake discussion or publication year for an explicit numeric assignment.")

value_records = [
    {"paper_id": "llm_interpretability", "source_type": "section_text", "chunk_id": "sae-numeric-1",
     "section": "SAE training", "text": "SAE sparsity is tuned during the experiment. The L1 coefficient lambda is set to 0.005 during training."},
]
value_hits, value_scan = mod.scan_index_for_numeric_parameter_evidence(sae_question, value_records)
assert value_scan["candidate_count"] == 1 and len(value_hits) == 1, (value_hits, value_scan)
assert value_hits[0]["numeric_parameter_scan"]["matched_value"] == "0.005", value_hits
print("PASS: full-index scan finds an explicit numeric coefficient assignment and preserves its source record.")

# Integration: if a full text-index scan finds no explicit number, return one cited and scope-limited abstention without calling the LLM.
class FakeSaeSearchEngine:
    records = no_value_records
    def search(self, question, top_k=8, paper_id=None):
        return no_value_records[:1]
original_call_ollama = mod.call_ollama
def unexpected_numeric_llm_call(prompt):
    raise AssertionError("LLM should be bypassed for deterministic no-value abstention after corpus scan")
mod.call_ollama = unexpected_numeric_llm_call
numeric_abstention_result = mod.answer_question(sae_question, top_k=8, search_engine=FakeSaeSearchEngine())
mod.call_ollama = original_call_ollama
assert "did not find a verifiable numeric value" in numeric_abstention_result["answer"], numeric_abstention_result["answer"]
assert "[S1" in numeric_abstention_result["answer"], numeric_abstention_result["answer"]
assert numeric_abstention_result["numeric_parameter_scan"]["records_scanned"] == 2, numeric_abstention_result["numeric_parameter_scan"]
assert not any(row["status"] == "uncited" for row in numeric_abstention_result["citation_audit"]), numeric_abstention_result["citation_audit"]
print("PASS: no-value result becomes one cited, scoped abstention instead of an overbroad claim.")
