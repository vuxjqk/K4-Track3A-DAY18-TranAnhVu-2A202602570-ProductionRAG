"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys / LLM (Google Gemini — free tier) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Giới hạn request/phút phía client (free tier bị 429 nếu gọi dồn dập)
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "12"))
# OpenAI-compatible endpoint của Gemini — dùng cho RAGAS (langchain_openai.ChatOpenAI)
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
# Mặc định theo đề bài; máy ít RAM có thể override qua .env bằng model nhỏ hơn,
# VD: EMBEDDING_MODEL=intfloat/multilingual-e5-small (dim lấy tự động từ model)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024
# Họ e5 cần prefix "query: " / "passage: " để embed đúng
_IS_E5 = "e5" in EMBEDDING_MODEL.lower()
EMBEDDING_QUERY_PREFIX = "query: " if _IS_E5 else ""
EMBEDDING_PASSAGE_PREFIX = "passage: " if _IS_E5 else ""

# --- Reranker ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
