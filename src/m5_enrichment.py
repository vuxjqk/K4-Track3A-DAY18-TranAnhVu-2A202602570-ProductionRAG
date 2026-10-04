from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm import has_llm, chat, chat_json


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if has_llm():
        try:
            summary = chat(
                "Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn bằng tiếng Việt, giữ nguyên các con số quan trọng. "
                "Bản tóm tắt phải ngắn hơn đoạn gốc. Chỉ trả về bản tóm tắt.",
                text,
            )
            # Đoạn gốc đã ngắn thì summary dài hơn là vô nghĩa → giữ nguyên text
            if summary and len(summary) <= max(len(text), 200):
                return summary
            return text
        except Exception as e:
            print(f"  ⚠️  Gemini summarize failed: {e}")

    # Extractive fallback (không cần API): lấy 2 câu đầu
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if has_llm():
        try:
            data = chat_json(
                f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời "
                "(cách nhân viên thường hỏi, mỗi câu kết thúc bằng dấu '?'). "
                'Trả về JSON: {"questions": ["...", "..."]}',
                text,
            )
            questions = [q.strip().lstrip("0123456789.-) ") for q in data.get("questions", []) if q.strip()]
            if questions:
                return questions[:n_questions]
        except Exception as e:
            print(f"  ⚠️  Gemini HyQA failed: {e}")

    # Extractive fallback: biến câu thành câu hỏi
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if has_llm():
        try:
            context = chat(
                "Viết 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì "
                "(nêu tên chính sách/phiên bản nếu có). Chỉ trả về 1 câu.",
                f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
            )
            if context:
                return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  Gemini contextual failed: {e}")

    # Simple fallback: prepend tên tài liệu
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if has_llm():
        try:
            meta = chat_json(
                "Trích xuất metadata từ đoạn văn. Trả về JSON: "
                '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|safety|compliance", "language": "vi|en"}',
                text,
            )
            if isinstance(meta, dict):
                return meta
        except Exception as e:
            print(f"  ⚠️  Gemini metadata failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if has_llm():
        try:
            result = chat_json(
                """Bạn hỗ trợ xây dựng hệ thống tìm kiếm tài liệu nội bộ. Phân tích đoạn văn (trích từ tài liệu) và trả về JSON:
{
  "summary": "tóm tắt 1-2 câu, giữ các con số quan trọng",
  "questions": ["3 câu hỏi tiếng Việt mà đoạn văn trả lời được, kết thúc bằng '?'"],
  "context": "1 câu mô tả đoạn văn thuộc tài liệu/chính sách nào (kèm phiên bản, ngày hiệu lực nếu có) và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|safety|compliance", "language": "vi|en"}
}""",
                f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
            )
            if isinstance(result, dict):
                return result
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")
    return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
