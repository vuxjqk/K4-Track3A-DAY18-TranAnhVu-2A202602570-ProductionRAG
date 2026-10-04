from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from src.llm import has_llm, generate_answer
from config import RERANK_TOP_K

# Latency breakdown (ms) — build-time + per-query
LATENCY: dict = {"build": {}, "queries": []}


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1) — hierarchical: index child, trả về parent làm context
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_text = {p.metadata["parent_id"]: p.text for p in parents}
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {
                **child.metadata, "parent_id": child.parent_id, "parent_text": parent_text[child.parent_id]}})
    LATENCY["build"]["chunking_s"] = round(time.time() - t0, 2)
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5) — contextual prepend + HyQA questions được index cùng chunk
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = []
        for e in enriched:
            questions = "\n".join(e.hypothesis_questions)
            index_text = f"{e.enriched_text}\n\n{questions}" if questions else e.enriched_text
            all_chunks.append({"text": index_text, "metadata": {**e.auto_metadata, "original_text": e.original_text}})
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)
    LATENCY["build"]["enrichment_s"] = round(time.time() - t0, 2)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    LATENCY["build"]["indexing_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    LATENCY["build"]["reranker_load_s"] = round(time.time() - t0, 2)
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def _expand_to_parents(results, top_k: int) -> list[str]:
    """Retrieve child → return parent: lấy parent của các child theo thứ tự rank, bỏ trùng."""
    contexts, seen = [], set()
    for r in results:
        pid = r.metadata.get("parent_id")
        text = r.metadata.get("parent_text") or r.metadata.get("original_text") or r.text
        key = pid or text
        if key in seen:
            continue
        seen.add(key)
        contexts.append(text)
        if len(contexts) == top_k:
            break
    return contexts


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    timing = {}
    t0 = time.perf_counter()
    results = search.search(query)
    timing["search_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    # Rerank toàn bộ candidates, sau đó gom về top-k parent khác nhau
    reranked = reranker.rerank(query, docs, top_k=len(docs))
    contexts = _expand_to_parents(reranked or results, RERANK_TOP_K)
    timing["rerank_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if has_llm() and contexts:
        try:
            answer = generate_answer(query, contexts)
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    timing["generation_ms"] = (time.perf_counter() - t0) * 1000

    LATENCY["queries"].append(timing)
    return answer, contexts


def latency_summary() -> dict:
    """Trung bình latency từng bước (ms/query) + build time."""
    q = LATENCY["queries"]
    steps = ["search_ms", "rerank_ms", "generation_ms"]
    avg = {s: round(sum(t[s] for t in q) / len(q), 1) for s in steps} if q else {}
    if avg:
        avg["total_ms"] = round(sum(avg.values()), 1)
    return {"build": LATENCY["build"], "per_query_avg": avg, "num_queries": len(q)}


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    LATENCY["build"]["ragas_eval_s"] = round(time.time() - t0, 2)
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    latency = latency_summary()
    print("\nLATENCY BREAKDOWN (avg/query)")
    for step, ms in latency["per_query_avg"].items():
        print(f"  {step:<15} {ms:>9.1f} ms")
    for step, sec in latency["build"].items():
        print(f"  {step:<15} {sec:>9.2f} s")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures, extra={"latency": latency})
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
