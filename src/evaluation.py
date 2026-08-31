"""
Phase 11: RAG Evaluation Benchmark Module
==========================================
Quantitative and qualitative evaluation framework for the RAG pipeline.
Tests Retrieval Hit Rate, Answer Groundedness, and Hallucination Refusal Rate.
Outputs `evaluation_results.json`.
"""

import sys, os
import json
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag_pipeline import TechnicalRAGPipeline

class RAGEvaluator:
    """Evaluates RAG pipeline performance against a benchmark dataset."""

    def __init__(self, pipeline: TechnicalRAGPipeline = None):
        self.pipeline = pipeline or TechnicalRAGPipeline("vectorstore")

    def get_benchmark_dataset(self) -> List[Dict[str, Any]]:
        """Constructs benchmark test dataset with ground truth expectations."""
        return [
            {
                "id": "eval_01",
                "question": "What is the HTTP Authorization header format required for API requests?",
                "expected_source": "enterprise_api_guide.pdf",
                "expected_page": 1,
                "keywords": ["Authorization", "Bearer", "sec_live_"],
                "type": "in_domain"
            },
            {
                "id": "eval_02",
                "question": "What endpoint and HTTP method are used for charging payment intents?",
                "expected_source": "enterprise_api_guide.pdf",
                "expected_page": 1,
                "keywords": ["/v2/payments/charge", "POST"],
                "type": "in_domain"
            },
            {
                "id": "eval_03",
                "question": "What does HTTP status code 401 mean in this API?",
                "expected_source": "enterprise_api_guide.pdf",
                "expected_page": 1,
                "keywords": ["401", "Unauthorized", "API key"],
                "type": "in_domain"
            },
            {
                "id": "eval_04",
                "question": "How are Webhook signatures generated and verified?",
                "expected_source": "enterprise_api_guide.pdf",
                "expected_page": 2,
                "keywords": ["HMAC-SHA256", "X-Signature-SHA256", "Webhook Secret"],
                "type": "in_domain"
            },
            {
                "id": "eval_05",
                "question": "What is the annual leave policy for full-time employees?",
                "expected_source": None,
                "expected_page": None,
                "keywords": ["could not find sufficient information"],
                "type": "out_of_domain"
            }
        ]

    def evaluate(self) -> Dict[str, Any]:
        """Runs evaluation benchmark and computes system metrics."""
        dataset = self.get_benchmark_dataset()
        results = []

        total_tests = len(dataset)
        retrieval_hits = 0
        grounded_answers = 0
        hallucination_refusals = 0

        for item in dataset:
            q = item["question"]
            q_type = item["type"]
            output = self.pipeline.run(q)

            ans = output["answer"]
            retrieved = output["retrieved_chunks"]

            # 1. Check Retrieval Hit Rate
            hit = False
            if q_type == "in_domain" and retrieved:
                for chunk in retrieved:
                    meta = chunk["metadata"]
                    if meta["source"] == item["expected_source"] and meta["page"] == item["expected_page"]:
                        hit = True
                        break
                if hit:
                    retrieval_hits += 1

            # 2. Check Answer Groundedness
            grounded = any(kw.lower() in ans.lower() for kw in item["keywords"])
            if grounded:
                grounded_answers += 1

            # 3. Check Out-of-Domain Refusal
            refusal_correct = False
            if q_type == "out_of_domain":
                if "could not find sufficient information" in ans.lower():
                    refusal_correct = True
                    hallucination_refusals += 1

            results.append({
                "id": item["id"],
                "question": q,
                "type": q_type,
                "retrieval_hit": hit if q_type == "in_domain" else True,
                "grounded": grounded,
                "refusal_correct": refusal_correct if q_type == "out_of_domain" else True,
                "answer_preview": ans[:200] + "...",
                "sources": output["sources"]
            })

        in_domain_count = sum(1 for item in dataset if item["type"] == "in_domain")
        out_domain_count = sum(1 for item in dataset if item["type"] == "out_of_domain")

        summary = {
            "total_benchmark_tests": total_tests,
            "retrieval_hit_rate": round(retrieval_hits / in_domain_count * 100, 2) if in_domain_count else 100.0,
            "answer_accuracy_rate": round(grounded_answers / total_tests * 100, 2),
            "hallucination_refusal_rate": round(hallucination_refusals / out_domain_count * 100, 2) if out_domain_count else 100.0,
            "test_cases": results
        }

        # Export evaluation results JSON
        output_file = "evaluation_results.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.evaluate()