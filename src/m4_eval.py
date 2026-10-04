from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SET_PATH, GEMINI_API_KEY, GEMINI_MODEL, GEMINI_OPENAI_BASE_URL,
                    EMBEDDING_QUERY_PREFIX)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    zeros = {m: 0.0 for m in metric_names}
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from datasets import Dataset

        llm, embeddings = _get_ragas_models()
        # Gemini không hỗ trợ n>1 completions → answer_relevancy sinh từng câu hỏi một
        answer_relevancy.strictness = 1

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm, embeddings=embeddings,
            # Free tier có rate limit → ít worker, retry nhiều
            run_config=RunConfig(max_workers=4, max_retries=10, max_wait=60, timeout=180),
            raise_exceptions=False,
        )
        df = result.to_pandas()

        def _score(row, m):
            v = row.get(m)
            return float(v) if v is not None and v == v else float("nan")  # NaN = metric lỗi

        per_question = [EvalResult(
            question=row["question"], answer=row["answer"], contexts=list(row["contexts"]),
            ground_truth=row["ground_truth"],
            **{m: _score(row, m) for m in metric_names},
        ) for _, row in df.iterrows()]

        aggregate = {}
        for m in metric_names:
            vals = [getattr(r, m) for r in per_question if getattr(r, m) == getattr(r, m)]
            aggregate[m] = sum(vals) / len(vals) if vals else 0.0
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {**zeros, "per_question": []}


def _get_ragas_models():
    """RAGAS judge = Gemini (qua OpenAI-compatible endpoint), embeddings = encoder local của M2 (miễn phí)."""
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src.m2_search import get_encoder

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY chưa được cấu hình — RAGAS cần LLM judge")

    class LocalEmbeddings(Embeddings):
        """Tái sử dụng SentenceTransformer đã load ở M2 thay vì load thêm 1 bản."""
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            texts = [EMBEDDING_QUERY_PREFIX + t for t in texts]  # answer_relevancy so khớp câu hỏi ↔ câu hỏi
            return get_encoder().encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    llm = ChatOpenAI(model=GEMINI_MODEL, api_key=GEMINI_API_KEY, base_url=GEMINI_OPENAI_BASE_URL,
                     temperature=0, max_retries=6)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(LocalEmbeddings())

def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — câu trả lời có claim không có trong context",
                         "Tighten prompt (chỉ dựa trên context), lower temperature, trích dẫn nguồn"),
        "context_recall": ("Missing relevant chunks — retriever không lấy đủ thông tin cần thiết",
                           "Improve chunking (parent-child), thêm BM25/HyQA, tăng top_k cho multi-hop"),
        "context_precision": ("Too many irrelevant chunks — context nhiễu, chunk liên quan bị xếp thấp",
                              "Add reranking, metadata filter (version/category), giảm top_k"),
        "answer_relevancy": ("Answer doesn't match question — trả lời lan man hoặc 'Không tìm thấy'",
                             "Improve prompt template, query rewriting, kiểm tra retrieval"),
    }
    metric_names = list(diagnostic_tree)

    analyzed = []
    for r in eval_results:
        # NaN (metric lỗi khi chấm) → bỏ qua khi chọn worst metric
        scores = {m: getattr(r, m) for m in metric_names if getattr(r, m) == getattr(r, m)}
        if not scores:
            continue
        avg = sum(scores.values()) / len(scores)
        worst_metric = min(scores, key=scores.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": round(avg, 4),
            "scores": {m: round(v, 4) for m, v in scores.items()},
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    analyzed.sort(key=lambda x: x["avg_score"])
    return analyzed[:bottom_n]

def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json",
                extra: dict | None = None):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        **(extra or {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
