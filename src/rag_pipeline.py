"""
Phases 6–10: End-to-End RAG Synthesis & Generation Engine
==========================================================
Coordinates Prompt Engineering (Phase 6), LLM Integration (Phase 7),
Source Attribution (Phase 8), Hallucination Guardrails (Phase 9),
and Multi-Turn Conversational Memory (Phase 10).
"""

import sys, os
from typing import List, Dict, Any, Optional
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retriever import TechnicalRetriever


class TechnicalRAGPipeline:
    """Enterprise RAG Assistant Pipeline with Guardrails & Source Attribution."""

    def __init__(self, vectorstore_dir: str = "vectorstore", retriever: TechnicalRetriever = None):
        if retriever:
            self.retriever = retriever
        else:
            try:
                self.retriever = TechnicalRetriever.load(vectorstore_dir)
            except Exception:
                print("Vectorstore not found. Initializing empty retriever.")
                self.retriever = TechnicalRetriever()

        # Initialize Gemini LLM Client if API key environment variable is present
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print("Gemini LLM Client successfully connected.")
            except Exception as e:
                print(f"Failed to initialize Gemini client: {e}")

    def reformulate_query_with_memory(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Phase 10: Multi-Turn Conversational Memory.
        Uses recent conversation turns to resolve pronouns and implicit follow-ups
        into a standalone vector search query.
        """
        if not chat_history:
            return query

        followup_indicators = ["it", "that", "this", "them", "how to fix", "what about", "why"]
        is_followup = any(re.search(rf"\b{word}\b", query.lower()) for word in followup_indicators) or len(query.split()) <= 4

        if is_followup:
            last_user_q = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
            if last_user_q:
                contextual_query = f"{last_user_q} {query}"
                print(f"Reformulated query with memory: '{query}' -> '{contextual_query}'")
                return contextual_query

        return query

    def build_grounded_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Phase 6: Grounded Prompt Engineering.
        Constructs a strict prompt instructing the model to rely ONLY on retrieved context
        and refuse unsupported queries.
        """
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            source_info = f"[DOC {idx} | File: {chunk['metadata']['source']} | Page: {chunk['metadata']['page']}]"
            context_blocks.append(f"{source_info}\n{chunk['content']}")

        formatted_context = "\n\n".join(context_blocks)

        prompt = f"""You are an Enterprise Technical Support AI Assistant.
Answer the user's question using ONLY the retrieved technical documentation context provided below.

STRICT GUARDRAILS:
1. Do NOT use outside knowledge. Rely ONLY on the context below.
2. If the context does not contain enough information to answer the question accurately, reply EXACTLY with:
   "I could not find sufficient information in the technical documentation to answer this question."
3. Do NOT invent fake API endpoints, invalid parameters, or unverified error codes.
4. Format code snippets cleanly using Markdown code blocks (e.g., ```json ... ``` or ```bash ... ```).

DOCUMENTATION CONTEXT:
---
{formatted_context}
---

USER QUESTION: {query}

GROUNDED ANSWER:"""
        return prompt

    def generate_llm_answer(self, prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Phase 7: LLM Integration & Generation.
        Calls Google Gemini API with temperature=0.0 for deterministic factual outputs,
        or falls back to context extraction if running offline without an API key.
        """
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.0}
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"LLM API Call error: {e}. Falling back to context extractor.")

        # Fallback Deterministic Extractor when offline without API key
        if not retrieved_chunks:
            return "I could not find sufficient information in the technical documentation to answer this question."

        primary_chunk = retrieved_chunks[0]["content"]
        answer_parts = [f"Based on the technical documentation:\n\n{primary_chunk}"]
        if len(retrieved_chunks) > 1:
            second_chunk = retrieved_chunks[1]["content"]
            if second_chunk[:100] not in primary_chunk:
                answer_parts.append(f"\nAdditional Details:\n{second_chunk}")

        return "\n".join(answer_parts)

    def format_source_attribution(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Phase 8: Mandatory Source Attribution.
        Generates transparent citations mapping answers back to document names and page numbers.
        """
        if not retrieved_chunks:
            return ""

        seen_sources = set()
        citations = []
        for chunk in retrieved_chunks:
            meta = chunk["metadata"]
            cite_key = f"{meta['source']} — Page {meta['page']}"
            if cite_key not in seen_sources:
                seen_sources.add(cite_key)
                citations.append(f"• **{meta['source']}** (Page {meta['page']})")

        return "\n".join(citations)

    def run(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None, top_k: int = 4) -> Dict[str, Any]:
        """
        Phase 9 & Execution Engine: Coordinates full RAG execution with hallucination guardrails.
        """
        chat_history = chat_history or []

        # 1. Reformulate Query with Memory (Phase 10)
        search_query = self.reformulate_query_with_memory(user_query, chat_history)

        # 2. Semantic Vector Retrieval (Phase 5)
        retrieved_chunks = self.retriever.search(search_query, top_k=top_k, score_threshold=0.25)

        # 3. Hallucination Guardrail Refusal Check (Phase 9)
        if not retrieved_chunks:
            fallback_msg = "I could not find sufficient information in the technical documentation to answer this question."
            return {
                "answer": fallback_msg,
                "sources": [],
                "formatted_sources": "",
                "retrieved_chunks": [],
                "is_fallback": True
            }

        # 4. Build Grounded System Prompt (Phase 6)
        prompt = self.build_grounded_prompt(user_query, retrieved_chunks)

        # 5. Synthesize LLM Response (Phase 7)
        raw_answer = self.generate_llm_answer(prompt, retrieved_chunks)

        # 6. Format Transparent Sources (Phase 8)
        formatted_sources = self.format_source_attribution(retrieved_chunks)

        is_refusal = "I could not find sufficient information" in raw_answer
        final_answer = raw_answer
        if not is_refusal and formatted_sources:
            final_answer += f"\n\n### Sources:\n{formatted_sources}"

        return {
            "answer": final_answer,
            "raw_answer": raw_answer,
            "sources": [f"{c['metadata']['source']} (P.{c['metadata']['page']})" for c in retrieved_chunks],
            "formatted_sources": formatted_sources,
            "retrieved_chunks": retrieved_chunks,
            "is_fallback": is_refusal
        }