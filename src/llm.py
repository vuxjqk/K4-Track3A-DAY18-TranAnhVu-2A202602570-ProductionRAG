from __future__ import annotations

"""Gemini LLM helper — dùng chung cho M5 (enrichment), pipeline và naive baseline.

- Gọi Gemini qua SDK `google-genai` (free tier, thay cho OpenAI).
- Retry + exponential backoff khi gặp 429 (rate limit) / 5xx (quá tải).
- Cache kết quả ra đĩa (.cache/llm_cache.json) → chạy lại pipeline không tốn quota.
"""

import hashlib, json, os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_RPM, LLM_CACHE_DIR

_CACHE_PATH = os.path.join(LLM_CACHE_DIR, "llm_cache.json")
_cache: dict | None = None
_lock = threading.Lock()
_client = None

RETRYABLE_CODES = {429, 500, 502, 503, 504}
_last_call = 0.0


def _throttle() -> None:
    """Giãn cách request để không vượt rate limit (RPM) của free tier."""
    global _last_call
    with _lock:
        wait = _last_call + 60.0 / GEMINI_RPM - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def has_llm() -> bool:
    return bool(GEMINI_API_KEY)


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    os.makedirs(LLM_CACHE_DIR, exist_ok=True)
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False)
    os.replace(tmp, _CACHE_PATH)


def _cache_key(model: str, system: str, user: str, json_mode: bool, temperature: float) -> str:
    return hashlib.sha256(json.dumps([model, system, user, json_mode, temperature],
                                     ensure_ascii=False).encode()).hexdigest()


def chat(system: str, user: str, json_mode: bool = False, temperature: float = 0.0,
         model: str = GEMINI_MODEL, use_cache: bool = True, max_retries: int = 8) -> str:
    """Gọi Gemini với system prompt + user message, trả về text.

    Raise exception nếu không có API key hoặc hết số lần retry — caller tự fallback.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY chưa được cấu hình")

    key = _cache_key(model, system, user, json_mode, temperature)
    if use_cache:
        with _lock:
            cached = _load_cache().get(key)
        if cached is not None:
            return cached

    from google.genai import types, errors
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    delay = 2.0
    for attempt in range(max_retries):
        _throttle()
        try:
            resp = _get_client().models.generate_content(model=model, contents=user, config=config)
            text = (resp.text or "").strip()
            break
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == max_retries - 1:
                raise
            print(f"  ⏳ Gemini {e.code}, retry sau {delay:.0f}s...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)

    if use_cache and text:
        with _lock:
            _load_cache()[key] = text
            _save_cache()
    return text


def chat_json(system: str, user: str, **kwargs) -> dict:
    """Gọi Gemini ở JSON mode và parse kết quả."""
    def _parse(text: str) -> dict:
        # Phòng trường hợp model bọc JSON trong ```json ... ```
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        return json.loads(text)

    try:
        return _parse(chat(system, user, json_mode=True, **kwargs))
    except json.JSONDecodeError:
        # JSON hỏng (có thể đã bị cache) → gọi lại bỏ qua cache, kết quả mới ghi đè cache
        kwargs["use_cache"] = False
        text = chat(system, user, json_mode=True, **kwargs)
        result = _parse(text)
        key = _cache_key(kwargs.get("model", GEMINI_MODEL), system, user, True, kwargs.get("temperature", 0.0))
        with _lock:
            _load_cache()[key] = text
            _save_cache()
        return result


# Prompt sinh câu trả lời — dùng chung cho naive baseline và production pipeline
ANSWER_SYSTEM_PROMPT = (
    "Bạn là trợ lý tra cứu chính sách nội bộ công ty. Trả lời câu hỏi CHỈ dựa trên context được cung cấp, "
    "bằng tiếng Việt, ngắn gọn, nêu đủ các con số/điều kiện liên quan. "
    "Nếu context có nhiều phiên bản chính sách, ưu tiên phiên bản hiện hành (mới nhất) và có thể nhắc phiên bản cũ đã bị thay thế. "
    "Nếu context không chứa thông tin → trả lời 'Không tìm thấy.'"
)


def generate_answer(query: str, contexts: list[str]) -> str:
    context_str = "\n\n---\n\n".join(contexts)
    return chat(ANSWER_SYSTEM_PROMPT, f"Context:\n{context_str}\n\nCâu hỏi: {query}")
