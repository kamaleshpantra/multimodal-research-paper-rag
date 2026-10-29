# Submission checklist — Multimodal Research Paper RAG

## What the package supports

- Three papers are discovered dynamically from the corpus/GROBID registry; production code does not special-case one paper ID or one figure number.
- Dense retrieval with FastEmbed + lexical BM25 reciprocal-rank fusion, source-type-aware ranking, near-duplicate text-extraction filtering, and dynamic evidence coverage for explicit cross-paper synthesis.
- Structured exact bibliography-title lookup, cautious index-wide numeric-parameter scanning, inline citation-marker validation, claim-level semantic/lexical alignment audit, and long exact phrase-overlap warnings for manual review.
- Figure/table evidence is used only when its visual description has an explicit human approval flag. The vision stage is round-robin across paper IDs and does not auto-approve.
- Deployment UI shows supporting source text, review warnings, citation-alignment audit, visual crops where available, and exact phrase-overlap flags.

## Required commands before you submit

1. Overlay this bundle into the existing project root (merge; do not delete your original raw corpus or local index).
2. Run the syntax and unit tests from `DEPLOYMENT_README.md`.
3. Run the round-robin visual candidate pass, review carefully, approve only accurate crops, and merge them if time permits. The currently audited data contains only one approved visual description; do not claim all figures/tables are fully understood unless you have reviewed more.
4. Build the deployment index using the exact same FastEmbed model configured for runtime.
5. Run deployment retrieval evaluation. Save the report as retrieval-only evaluation; manually review answer correctness and citations separately.
6. Run five smoke tests in the app: ordinary factual question, exact bibliography title, explicit cross-paper synthesis, figure/table query, and unanswerable parameter question.
7. Confirm no API key, `.env`, raw PDFs, private papers, `data/index/`, or Ollama model files are committed. Check redistribution rights before pushing corpus excerpts/visual crops to a public repository.

## Academic-integrity statement

The assistant is instructed to paraphrase explanatory prose, cite source-derived claims, preserve exact bibliographic titles when a title lookup is requested, and report long exact source-phrase overlap for manual review. This is not AI-writing detection, a plagiarism verdict, nor a guarantee of originality. You must review overlap alerts, quote/cite source material correctly, and apply your institution's integrity rules.

## Evaluation language that is safe to claim

- Unit tests passing demonstrates behavior on synthetic/regression fixtures, not scientific performance on the actual PDFs.
- Hit@K/Recall@K/MRR/nDCG measure retrieval rankings against the provided labels. The current label set is small and should be manually verified; it does not measure generation factuality, visual accuracy, citation entailment, abstention accuracy, or plagiarism.
- The deployment endpoint uses a hosted open-weight multimodal model because free Streamlit hosting cannot serve the local Ollama endpoint. This is remote inference, not fully self-hosted inference.
