# Multimodal Research Paper Intelligence

An evidence-grounded research assistant for asking questions about three scientific papers. It retrieves relevant passages from the indexed papers, generates an answer with source markers, and lets the reader inspect the evidence behind the answer.

> [**Live demo:** -](https://arxiv.org/abs/2005.11401)
>
> **GitHub repository:** Add the repository URL here after creating it.

## What this project does

The application is designed to help a reader explore the included papers without manually searching every page.

- **Ask questions in natural language.** Ask about a single paper or select all papers for a cross-paper question.
- **Retrieve evidence using hybrid search.** Dense embeddings find semantically similar passages; BM25 finds keyword matches. Reciprocal-rank fusion combines the ranked results.
- **Show source citations.** Answers use inline markers such as `[S1]`. Expand a source to inspect its paper, section, page or bibliography metadata, and retrieved text.
- **Compare and synthesize papers.** The retriever can reserve evidence from multiple papers for comparison and synthesis questions.
- **Look up bibliography entries.** Exact title questions are answered from structured reference metadata where a matching record is available.
- **Support figures and tables cautiously.** Captions, extracted table records, and human-reviewed visual descriptions can be retrieved. Reviewed visual crops may be sent to the multimodal model when relevant.
- **Abstain on unsupported numeric details.** For supported parameter-style questions, the code scans indexed text for explicit numeric assignments and reports when it cannot verify a value.
- **Review citation alignment and phrase overlap.** Similarity audits and exact-phrase overlap flags help a human inspect an answer; they do not prove entailment or determine plagiarism.
- **Export a report.** Download the answer, supporting excerpts, warnings, and audits as a Markdown file.

## Research papers used by this assistant

This demo is intentionally scoped to **these three papers**. The chatbot retrieves excerpts from their indexed text and approved visual descriptions; it is not a general-purpose web-search chatbot. Use the links below to read the original papers and understand what the demo is designed to answer.

### 1. Retrieval-Augmented Generation (RAG)

**Patrick Lewis et al. (2020). _Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks._ NeurIPS 2020.**

This foundational paper introduces a generator that combines a pre-trained language model with an external, non-parametric document memory. In this demo it supports questions about the retriever/generator workflow, RAG-Sequence versus RAG-Token, and the paper's experiments and findings.

- [Read the paper on arXiv](https://arxiv.org/abs/2005.11401)
- [Open the PDF](https://arxiv.org/pdf/2005.11401)
- [NeurIPS paper page](https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)

### 2. Mechanistic interpretability

**Daking Rai, Yilun Zhou, Shi Feng, Abulhair Saparov, and Ziyu Yao (2024). _A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models._ arXiv:2407.02646.**

This review surveys ways to study the internal computations of transformer language models, including interpretability objects, techniques, evaluation approaches, findings, limitations, and open directions. In this demo it supports questions about the review's taxonomy, methods, and stated limitations.

- [Read the paper on arXiv](https://arxiv.org/abs/2407.02646)
- [Open the PDF](https://arxiv.org/pdf/2407.02646)

### 3. Explainable recommendation

**Qiyao Ma, Xubin Ren, and Chao Huang (2024). _XRec: Large Language Models for Explainable Recommendation._ Findings of EMNLP 2024, pp. 391–402.**

This paper presents XRec, a framework that combines collaborative signals with large language models to produce explanations for recommendations. In this demo it supports questions about the collaborative adaptor, user–item signals, explanation generation, and the paper's methods and experiments.

- [Read the paper in the ACL Anthology](https://aclanthology.org/2024.findings-emnlp.22/)
- [Open the PDF](https://aclanthology.org/2024.findings-emnlp.22.pdf)
- [arXiv version](https://arxiv.org/abs/2406.02377)

### Why these papers?

Together, the collection provides three complementary research perspectives: **RAG methodology**, **understanding the internal computations of language models**, and **using language models for explainable recommendation**. The application can answer single-paper questions, retrieve bibliography entries, and attempt cross-paper synthesis when the indexed evidence supports it. If evidence is missing, the answer should say so rather than invent details.

**About distributing the PDFs:** this repository links to the original publisher or preprint pages instead of re-hosting the paper PDFs. Before committing the PDFs, extracted full text, or figure crops to a public repository, check the applicable license and redistribution terms. For local ingestion, place permitted PDF copies in `data/selected_papers/`; that local input folder is ignored by the recommended `.gitignore`.

The app builds its paper selector from the paper registry included with the deployment index. To add a paper, process it into the corpus, review any visual descriptions, rebuild the deployment index, and test retrieval and answers for the new paper.

## Architecture

```mermaid
flowchart TD
    A[Research paper PDFs] --> B[PDF text and layout extraction]
    B --> C[GROBID metadata, sections, references and citation links]
    B --> D[Corpus records with source metadata]
    C --> D
    D --> E[Optional visual candidate extraction]
    E --> F[Human review and approval]
    F --> D
    D --> G[Deployment index builder]
    G --> H[FastEmbed document vectors]
    G --> I[BM25 searchable text]
    H --> J[Streamlit research assistant]
    I --> J
    J --> K[Hybrid retrieval and duplicate filtering]
    K --> L[Retrieved excerpts and approved visual evidence]
    L --> M[OpenRouter-hosted language / vision model]
    M --> N[Answer with source markers]
    N --> O[Citation alignment audit and overlap flags]
    O --> P[Answer, source viewers and Markdown export]
```

### Request flow

1. The reader enters a question and selects a paper scope.
2. The app embeds the question with the **same FastEmbed model used to build the deployment index**, then combines dense retrieval with BM25 results.
3. Near-duplicate text records from separate extraction paths are filtered to reduce redundant evidence.
4. Exact bibliography-title lookup and selected numeric-parameter checks can use deterministic metadata/text scans rather than relying only on generation.
5. Retrieved text, metadata, and relevant approved visual crops are passed to the configured hosted model.
6. The app displays the answer, source records, citation-alignment audit, and any warnings.

## Technology stack

| Area | Technology | Purpose |
|---|---|---|
| User interface | Streamlit | Paper selector, question form, answers, sources, warnings, and report download |
| Language | Python 3.11 | Application, ingestion utilities, retrieval, evaluation, and tests |
| Deployment embeddings | FastEmbed with `BAAI/bge-base-en-v1.5` by default | Local ONNX document and query embeddings for the deployed app |
| Keyword retrieval | `rank-bm25` | Lexical retrieval for exact terms and technical phrases |
| Retrieval ranking | NumPy, cosine similarity, reciprocal-rank fusion (RRF) | Combine dense and keyword rankings |
| Answer generation | OpenRouter API; default model `nvidia/nemotron-3.5-super-vl:free` | Generate evidence-conditioned text and interpret selected images |
| PDF/layout extraction | PyMuPDF | Extract text, page information, and figure/table-related candidates |
| Structured scientific parsing | GROBID service (ingestion pipeline) | Extract paper metadata, sections, references, and in-text citation links |
| Local model utilities | Ollama; Qwen models and BGE-M3 where configured | Local answer experiments and optional visual-description generation before deployment |
| Image handling | Pillow | Stage and display reviewed image crops |
| Evaluation | Python scripts and labeled JSONL seed set | Measure retrieval ranking and save per-question reports |

**Important model distinction:** the deployment app uses FastEmbed and OpenRouter. It does not call the Ollama server on your computer. The separate local experiment path can continue to use Ollama and its BGE-M3 index. Embeddings from different models must not be mixed; rebuild the index whenever the deployment embedding model changes.

## Repository layout

The full working project may include additional local ingestion files. The main files relevant to this app are:

```text
.
├── app.py                              # Streamlit entry point
├── requirements.txt                    # Python dependencies
├── runtime.txt                         # Streamlit runtime Python version
├── .gitignore                          # Ignore secrets, environments, local data and logs
├── src/
│   ├── ingestion/
│   │   ├── vision_index.py             # Create visual candidates across papers
│   │   └── approve_vision_records.py   # Human review of visual candidates
│   ├── retrieval/
│   │   ├── build_corpus.py             # Local project: build searchable records from processed PDFs
│   │   ├── build_deployment_index.py   # Build the separate FastEmbed deployment index
│   │   ├── deployment_rag.py           # Deployment search, grounded prompting, citations and guardrails
│   │   └── hybrid_search.py             # Local BGE-M3/Ollama experiment retrieval
│   ├── generation/
│   │   └── answer_generator.py         # Local Ollama answer-generation workflow
│   └── evaluation/
│       ├── evaluate_rag.py              # Local experiment retrieval/generation evaluation
│       └── evaluate_deployment_index.py # Deployment-index retrieval evaluation
├── tests/
│   ├── test_answer_generator_citation_audit.py
│   └── test_deployment_rag.py
└── data/
    ├── processed/                      # Locally generated corpus and GROBID outputs
    ├── index/                          # Local BGE-M3/Ollama experiment index
    ├── deployment_index/               # Required runtime index for the deployed app
    └── evaluation/
        ├── eval_set.jsonl              # Draft labeled retrieval questions
        └── ...                          # Evaluation reports (usually ignored by Git)
```

Some ingestion scripts and local data are created during the earlier PDF-processing workflow and may not be included in a deployment-only overlay. Keep the files that already exist in your project when merging updated files.

## Run locally

### Requirements

- Python 3.11
- Git
- Internet access for the first FastEmbed model download and hosted generation requests
- An OpenRouter API key for answer generation
- The built `data/deployment_index/` directory

### 1. Clone the repository

Replace the placeholders with your actual GitHub username and repository name after publishing:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Build the deployment index if it is not included

The builder requires the locally prepared `data/processed/corpus.jsonl` and, where available, GROBID metadata files. The first run downloads the configured embedding model.

**Windows PowerShell:**

```powershell
python src\retrieval\build_deployment_index.py
```

**macOS / Linux:**

```bash
python src/retrieval/build_deployment_index.py
```

The builder writes these files to `data/deployment_index/`:

- `records.jsonl`
- `embeddings.npy`
- `index_manifest.json`
- `paper_registry.json`
- `assets/` for reviewed visual crops, when present

The **deployment index must exist at runtime**. If you clone only the source code and do not include this directory, the app will display a missing-index message. If your clone does not contain the prepared corpus, use the already-built deployment index or run the full local PDF-ingestion workflow before rebuilding it.

### 4. Configure the API key without putting it in code

For a quick test in Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"
$env:OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl:free"
$env:DEPLOY_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
```

These environment variables apply to the current PowerShell session. Do **not** paste the real key into this README, `app.py`, a screenshot, or a Git commit.

### 5. Start the app

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal. The application user does not need to enter an API key; the app reads it from its environment or Streamlit secrets.

## Deploy to Streamlit Community Cloud

1. Build and test `data/deployment_index/` locally.
2. Create a GitHub repository and push the application source, `requirements.txt`, `runtime.txt`, tests, and the required deployment index.
3. Before publishing any text excerpts or figure crops, check the papers' licenses and redistribution rights. Do not upload source PDFs unless you have permission to redistribute them.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from your repository.
5. Set the entry-point file to `app.py` and deploy.
6. In the app settings, open **Secrets** and add:

   ```toml
   OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"
   OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl:free"
   DEPLOY_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
   ```

7. After deployment, add the real app URL to the **Live demo** section at the top of this README.
8. Smoke-test a normal text question, a bibliography-title lookup, a cross-paper question, a reviewed-figure question, and an unanswerable question.

See the [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) and [Streamlit secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) for the current deployment workflow.

## Evaluation and tests

Run the local syntax and unit-test checks before pushing:

```powershell
python -m py_compile app.py src\retrieval\deployment_rag.py src\retrieval\build_deployment_index.py src\generation\answer_generator.py src\ingestion\vision_index.py src\ingestion\approve_vision_records.py src\evaluation\evaluate_deployment_index.py
python tests\test_answer_generator_citation_audit.py
python -m unittest tests.test_deployment_rag -v
```

Run deployment retrieval evaluation after building the index:

```powershell
python src\evaluation\evaluate_deployment_index.py --top-k 8
```

The evaluation script writes a per-run JSON report and a Markdown summary under `data/evaluation/deployment_runs/`.

**Evaluation limitation:** the included evaluation set is a small seed set with draft relevance labels. Verify labels manually and add held-out questions before reporting benchmark-quality metrics. Hit@K, Recall@K, MRR, and nDCG measure retrieval; they do not establish answer correctness, citation entailment, visual-reading accuracy, abstention quality, or plagiarism status. Review generated answers separately.

## Academic integrity and known limitations

- Answers are intended to paraphrase explanations and cite claims derived from the papers. Bibliographic titles, technical terminology, equations, and short technical phrases may need to remain exact.
- Inline source markers are checked against the retrieved source list. Similarity-based citation audits identify potentially relevant evidence but **do not prove that the source entails a claim**.
- Phrase-overlap flags support manual review. They are not an AI-writing detector, do not give a plagiarism score, and cannot guarantee plagiarism-free output.
- Visual descriptions must be checked against their original crop before approval. A valid model-generated JSON response is not proof that labels, arrows, values, axes, or table cells were read correctly.
- If a numeric assignment is not found in the indexed text, the app can report that it was not found in that text. It cannot prove that the value does not appear in an unparsed equation, image, table, or missing extraction.
- Model/provider availability, privacy practices, free quotas, and pricing can change. The default OpenRouter model ID ends in `:free`, but it remains subject to provider availability and rate limits. Check the [model page](https://openrouter.ai/nvidia/nemotron-3.5-super-vl%3Afree) before a live demo.
- Questions, retrieved excerpts, and selected reviewed images are sent to the configured hosted inference provider. Do not use the public demo for confidential or unpublished documents.

## Git and secret-safety checklist

Before committing, inspect the files Git plans to include:

```bash
git status --short
```

Never commit:

- API keys, `.env` files, or `.streamlit/secrets.toml`
- `.venv/`, Python caches, or local model weights
- Raw PDFs or derived text/image material unless redistribution is permitted
- Local-only `data/index/` experiment embeddings and temporary evaluation runs

The deployment app **does need `data/deployment_index/`** unless you change the design to fetch that index from a separate controlled storage location. Do not accidentally ignore it. Check the rights to redistribute its extracted paper passages and reviewed crops before making the repository public.

After reviewing `git status`, push the repository:

```bash
git add README.md .gitignore app.py requirements.txt runtime.txt src tests data/evaluation/eval_set.jsonl data/deployment_index
git commit -m "Add research paper RAG assistant"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
git push -u origin main
```

If a Git remote already exists, do not add it a second time; check it with `git remote -v` and push to the correct repository.

## Cost notes

The application does local embedding/search and uses a hosted model for answer generation. The default model is a free OpenRouter variant at the time this README was written, but its availability, quotas, logging policy, and pricing are controlled by the provider and can change. Keep API keys private, do not configure a paid fallback unintentionally, and do not enable automatic credit top-ups if you want to avoid paid usage.

## Acknowledgements and references

- [Streamlit](https://streamlit.io/) — application framework.
- [FastEmbed](https://github.com/qdrant/fastembed) — local embedding generation.
- [GROBID](https://github.com/kermitt2/grobid) — scientific document metadata and reference parsing in the ingestion workflow.
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF extraction utilities.
- [OpenRouter](https://openrouter.ai/) — hosted model API.

---

*This README was written for this project and describes the intended implementation and its known constraints. Generated answers should still be checked against the cited papers before being used in academic work.*
