# Enterprise Technical Documentation Assistant (RAG System)
> **SkillFlow AI Major Capstone Project**: End-to-End Retrieval-Augmented Generation Application for Enterprise API Specifications, Developer SDKs, Code Libraries & Troubleshooting SOPs.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/RAG-FAISS%20%7C%20SentenceTransformers%20%7C%20Gemini-orange.svg)]()
[![UI](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Title & Overview
The **Enterprise Technical Documentation Assistant** is a production-grade Generative AI application built on **Retrieval-Augmented Generation (RAG)** principles. It enables software developers, DevOps engineers, and technical support teams to query multi-page technical manuals, OpenAPI specs, and troubleshooting guides in natural language, receiving deterministic, context-grounded responses accompanied by clean code snippets and transparent page-level citations.

---

## 2. Business Problem
Modern software enterprises maintain hundreds of technical PDFs, payment manuals, and microservice SOPs. Technical teams face severe efficiency bottlenecks:
* **Manual Overhead:** Engineers waste hours manually searching through long documentation files for error codes, HTTP headers, or request payload parameters.
* **LLM Hallucinations:** Raw, ungrounded LLMs frequently generate invalid code syntax, fake API endpoints, or deprecated parameter names.
* **Context Fragmentation:** Information is scattered across disparate manuals, making multi-document troubleshooting slow and error-prone.

---

## 3. Selected Domain & Rationale
* **Selected Domain:** **Technical Documentation Assistant (Developer & API Intelligence)**.
* **Why Selected:** Technical documentation demands absolute precision, code syntax preservation, exact parameter matching, and zero tolerance for hallucinated functions. Demonstrating RAG on technical specifications highlights high engineering value, real-world utility, and clear quantitative evaluation metrics.

---

## 4. Document Sources & Usage Conditions
* **Document Sources:**
  1. `enterprise_api_guide.pdf`: Enterprise Payment Gateway & Authentication API Spec (OAuth 2.0 Bearer tokens, `/v2/payments/charge` JSON payload, HTTP status codes 401/422/500, HMAC-SHA256 Webhook signatures).
  2. `stripe_payments_api_v3.pdf`: Stripe Payments API v3 Manual (POST `/v3/refunds`, `charge_id`, `amount`, `reason` fields, idempotency keys, rate limits).
  3. `docker_microservices_troubleshooting.pdf`: Microservices & Docker Troubleshooting Guide (Kubernetes `CrashLoopBackOff`, Exit Code 137 OOMKilled, PostgreSQL connection pool tuning).
* **Usage Conditions:** All documents are synthetic or publicly accessible developer guides, strictly containing zero confidential corporate secrets, private keys, or personal identifiable information (PII).

---

## 5. System Architecture
```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            END-TO-END RAG PIPELINE FLOW                                 │
│                                                                                         │
│ [Phase 1: Ingestion] ──► [Phase 2: Chunking] ──► [Phase 3: Embeddings] ──► [Phase 4: DB] │
│                                                                                  │      │
│ [Phase 8: Sources]  ◄── [Phase 7: LLM]   ◄── [Phase 6: Prompting]  ◄── [Phase 5: Search]│
│        │                                                                                │
│        ├──► [Phase 9: Hallucination Guardrails]                                         │
│        ├──► [Phase 10: Multi-Turn Conversational Memory]                                │
│        ├──► [Phase 11: Quantitative RAG Evaluation]                                     │
│        └──► [Phase 12: Interactive Streamlit Web Application]                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Document Processing (`src/document_loader.py`)
* Extracts raw text page-by-page using `pypdf`.
* Cleans structural formatting noise while preserving code indentation, curly braces `{ }`, JSON schemas, and HTTP header strings.
* Attaches metadata payloads to every page: `{source, file_path, page, total_pages, char_count, file_type}`.

---

## 7. Chunking Strategy (`src/chunking.py`)
* **Code-Aware Technical Chunker:** Custom recursive splitter tailored for technical syntax.
* **Chunk Size:** `600 characters` (~100 words) — sufficient to encompass complete endpoint definitions, payload examples, or error resolution tables.
* **Chunk Overlap:** `100 characters` — ensures boundary continuity across JSON parameters and code blocks.
* **Separators:** Dynamic hierarchy `["\n\n", "\n", " ", ""]` to prevent splitting mid-code block or mid-JSON payload.

---

## 8. Embedding Model (`src/embeddings.py`)
* **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
* **Normalization:** L2-normalization applied to all chunk and query vectors to compute exact Cosine Similarity using Inner Product dot products.

---

## 9. Vector Database (`src/retriever.py`)
* **Engine:** **FAISS** (Facebook AI Similarity Search) using `faiss.IndexFlatIP`.
* **Persistence:** Serializes binary index to `vectorstore/faiss_index.bin` and metadata payloads to `vectorstore/metadata.pkl` for fast startup.

---

## 10. Retrieval Strategy (`src/retriever.py`)
* Converts user queries into 384-dimensional dense vectors.
* Performs Top-$K$ ($K=4$) semantic search against FAISS.
* Enforces a similarity score threshold (`score_threshold = 0.25`) to filter out weak or off-topic context snippets.

---

## 11. LLM Integration (`src/rag_pipeline.py`)
* **LLM Engine:** Google Gemini API (`gemini-3.6-flash` / `gemini-2.5-flash`) operating at `temperature=0.0` for deterministic, factual generation.
* **Offline Fallback:** Includes a zero-dependency deterministic context extractor when offline or when API limits are exceeded.

---

## 12. Prompt Strategy (`src/rag_pipeline.py`)
* **Strict Context Grounding:**
  > *"You are an Enterprise Technical Support AI Assistant. Answer the user's question using ONLY the retrieved technical documentation context provided below. If the context does not contain enough information, reply EXACTLY with: 'I could not find sufficient information in the technical documentation to answer this question.'"*

---

## 13. Hallucination Handling & Guardrails (`src/rag_pipeline.py`)
* **Active Refusal:** Out-of-domain queries (e.g. employee leave policies) return zero relevant context chunks, triggering an exact refusal response.
* **Source Attribution:** Appends transparent citations linking every generated answer back to document filenames and page numbers (e.g., `Sources: • stripe_payments_api_v3.pdf (Page 1)`).
* **Multi-Turn Memory:** Reformulates ambiguous follow-up questions (e.g. *"What are the parameters for that payload?"*) using recent chat history before vector search.

---

## 14. Evaluation Methodology (`src/evaluation.py`)
Quantitative evaluation suite testing system performance across benchmark in-domain technical Q&As and out-of-domain refusal cases:
1. **Retrieval Hit Rate @ K:** Checks whether top-$K$ vector retrieval fetched relevant document context for in-domain queries.
2. **Answer Groundedness Accuracy:** Verifies generated answers against ground-truth technical keywords, endpoints, and parameters.
3. **Hallucination Refusal Rate:** Measures accuracy of refusing unsupported out-of-domain questions.

---

## 15. Benchmark Results

| Evaluation Metric | Score | Performance Explanation |
| :--- | :---: | :--- |
| **Retrieval Hit Rate @ K** | **100.0%** | FAISS top-$K$ search successfully retrieved correct document pages for all technical queries. |
| **Answer Groundedness Accuracy** | **100.0%** | Generated answers matched exact HTTP verbs, endpoints, request schemas, and error fixes. |
| **Hallucination Refusal Rate** | **100.0%** | System safely triggered active refusal for non-existent and out-of-domain questions. |

---

## 16. Streamlit Deployment Instructions

### Option A: Local Run
1. **Clone Repository & Install Requirements:**
   ```bash
   git clone https://github.com/MohneeshPurohit/Enterprise_Technical_Documentation_Assistant.git
   cd Enterprise_Technical_Documentation_Assistant
   pip install -r requirements.txt
   ```

2. **Set API Key (Optional):**
   ```bash
   export GEMINI_API_KEY="AIzaSy..."
   ```

3. **Run Evaluation Benchmark:**
   ```bash
   PYTHONPATH=. python3 src/evaluation.py
   ```

4. **Launch Application:**
   ```bash
   streamlit run app.py
   ```

---

## 17. Limitations
* **PDF Layout Parsing:** Heavy multi-column tables with merged cells require specialized OCR / table vision models for layout reconstruction.
* **Vector Model Context:** `all-MiniLM-L6-v2` has a 256-token context limit per chunk; extremely long code blocks must be split cleanly across chunks.

---

## 18. Future Improvements
1. **Hybrid Retrieval:** Combine BM25 keyword matching (for exact API method names like `POST`) with FAISS dense vector search.
2. **Reranking Engine:** Integrate a Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`) to re-score top-10 candidate chunks before prompt synthesis.
3. **Streaming Responses:** Enable real-time token streaming in the Streamlit UI using Gemini streaming output endpoints.

---

## Repository Structure
```text
Enterprise_Technical_Documentation_Assistant/
├── .gitignore                 # Cache, binary index, and secret exclusions
├── README.md                  # Master project README
├── requirements.txt           # Dependency requirements
├── app.py                     # Streamlit web application
├── evaluation_results.json    # Quantitative benchmark evaluation results
├── Technical_Documentation_Assistant_RAG_Capstone_Guide.pdf  # Master 8-page PDF guide
│
├── data/
│   └── documents/             # Technical PDF documents
│       ├── enterprise_api_guide.pdf
│       ├── stripe_payments_api_v3.pdf
│       └── docker_microservices_troubleshooting.pdf
│
├── src/                       # Core Python package
│   ├── __init__.py
│   ├── document_loader.py     # Phase 1: Ingestion & Text Extraction
│   ├── chunking.py            # Phase 2: Technical Code-Aware Chunking
│   ├── embeddings.py          # Phase 3: SentenceTransformers Embeddings
│   ├── retriever.py           # Phase 4 & 5: FAISS Vector DB Search
│   ├── rag_pipeline.py        # Phase 6–10: Grounding, LLM, Sources & Memory
│   └── evaluation.py          # Phase 11: Benchmark Evaluator
│
└── notebooks/                 # Jupyter Exploration Notebooks
    ├── 01_document_exploration.ipynb
    ├── 02_chunking_and_embeddings.ipynb
    └── 03_rag_evaluation.ipynb
```
