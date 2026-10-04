# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Trần Anh Vũ  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 (all-MiniLM-L6-v2) tạo **208 chunks** (avg 99 ký tự) vs basic **51 chunks** (avg 410). Với tiếng Việt, MiniLM (model tiếng Anh) cho similarity giữa các câu liền kề thấp → tách quá vụn (min 6 ký tự). Cần model đa ngữ hoặc threshold thấp hơn mới thực sự "nhóm theo chủ đề". |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` + `_expand_to_parents()` | 112 child (≤256 ký tự) để retrieve chính xác, nhưng trả về **parent** làm context. Đây là thay đổi tác động mạnh nhất: context_recall 0.758 → **0.925**, câu "Lương thử việc Junior" từ avg 0.25 → 0.94. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | 106 chunks theo header `#`–`###`, giữ nguyên bảng/list, có `section` trong metadata — phù hợp để làm đơn vị trả về thay cho parent dài (hướng fix failure #1). |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `reciprocal_rank_fusion()` | underthesea gộp "nghỉ_phép" → phải `replace("_", " ")` để token query/doc khớp. RRF (k=60) chỉ dùng **thứ hạng** nên không cần chuẩn hóa điểm BM25 (0–20+) và cosine (0–1). BM25 bắt tốt keyword chính xác ("MFA", "PVI", "tạm ứng"), dense bắt paraphrase; search chỉ tốn **33.8 ms/query**. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Rerank top-20 → top-3 parent: **1059 ms/query** trên CPU (mmarco-mMiniLMv2) — chiếm ~35% latency. context_precision 0.85 → **0.95**. Hạn chế: chấm theo cả câu nên câu multi-hop bị lệch về 1 intent (bang_luong xuống hạng 12, failure #2), và không phân biệt version (v2023 xếp trên v2024, failure #1). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production: F 0.926 / AR 0.942 / CP 0.950 / CR 0.925. Baseline thấp nhất ở **context_recall (0.758)** vì chunk 500 ký tự + dense-only bỏ sót tài liệu thứ 2 của câu multi-hop/version. Nhận ra LLM judge (flash-lite) có false negative: câu trả lời "Không. …" bị faithfulness 0.5, câu đúng hoàn toàn bị context_precision 0. |
| Contextual embeddings + HyQA | M5 | `_enrich_single_call()` (1 call/chunk) | Mỗi chunk được prepend 1 câu ngữ cảnh ("Đoạn văn thuộc Chính sách nghỉ phép năm phiên bản 2024…") + index thêm 3 câu hỏi giả định → bridge vocabulary gap giữa câu hỏi người dùng và văn bản chính sách. Combined mode: 112 calls thay vì 448 calls (4 kỹ thuật riêng lẻ). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

**1. Chuyển từ OpenAI sang Gemini (free tier)**
- Lỗi: `404 This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash`
- Debug: gọi `GET /v1beta/models` để liệt kê model khả dụng với key, test lần lượt → chọn `gemini-3.5-flash-lite` (nhanh ~1s, quota free tốt hơn).
- Thiết kế: tạo `src/llm.py` dùng SDK `google-genai` cho M5 + generation; RAGAS (0.1.x chỉ nhận LangChain LLM) gọi Gemini qua **OpenAI-compatible endpoint** (`ChatOpenAI(base_url=".../v1beta/openai/")`).

**2. RAGAS answer_relevancy lỗi với Gemini**
- Lỗi: `400 Multiple candidates is not enabled for this model`
- Nguyên nhân: đọc source `ragas.llms.base.is_multiple_completion_supported()` → `ChatOpenAI` được coi là hỗ trợ `n>1`, `answer_relevancy` dùng `n=strictness=3`.
- Fix: `answer_relevancy.strictness = 1`.

**3. Torch không load được trên Windows**
- Lỗi: `OSError: [WinError 126] The specified module could not be found. Error loading "...\torch\lib\c10.dll" or one of its dependencies.` + `Microsoft Visual C++ Redistributable is not installed`
- Debug: kiểm tra `C:\Windows\System32` → thiếu `msvcp140.dll`. Không có quyền admin để cài VC++ Redist → `pip install msvc-runtime` (DLL nằm trong `.venv\Scripts`), nhưng venv vẫn chạy bằng `python.exe` gốc nên Windows không tìm thấy DLL → thêm file `.pth` gọi `os.add_dll_directory(<venv>\Scripts)`.

**4. Hết RAM khi load model**
- Lỗi: `memory allocation of 67072065 bytes failed` (Rust/hf_xet khi tải) và `OSError: The paging file is too small for this operation to complete. (os error 1455)`
- Debug: máy 7.8GB RAM, commit 13.4/15.8GB, pagefile đã max. bge-m3 chỉ có `pytorch_model.bin` (pickle, không mmap được) → load fp32 2.2GB → segfault kể cả khi ép bf16.
- Fix: `HF_HUB_DISABLE_XET=1` để tải; cho phép override model qua `.env` (`EMBEDDING_MODEL=intfloat/multilingual-e5-small`, `RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`), tự lấy dimension từ model, thêm prefix `query:`/`passage:` cho e5; dùng chung 1 instance encoder cho Dense search và RAGAS embeddings.

**5. Rate limit + JSON hỏng từ LLM**
- Lỗi: `429 You exceeded your current quota` (RAGAS gọi song song làm cạn quota/phút) và `Expecting property name enclosed in double quotes: line 11 column 5` — model sinh `-entities: [` trong JSON.
- Fix: throttle phía client (`GEMINI_RPM`), retry exponential backoff cho 429/5xx, cache kết quả LLM ra `.cache/` (chạy lại không tốn quota); khi JSON hỏng thì gọi lại bỏ qua cache và ghi đè.

**Kiến thức còn thiếu & cách bổ sung:** chưa nắm cách RAGAS tính từng metric (statement decomposition, rank-weighted precision) → đọc source `ragas/metrics/_faithfulness.py`, `_context_precision.py` để hiểu vì sao câu trả lời ngắn "Không." bị phạt; cần tìm hiểu thêm về quantization/ONNX để chạy bge-m3 trên máy ít RAM.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Chatbot hỏi đáp tài liệu nội bộ (quy chế, chính sách, hướng dẫn) cho doanh nghiệp

#### 1. Hiện trạng
- **Pipeline hiện tại:** chunk cố định theo số ký tự → embedding → vector search top-k → LLM, chưa có evaluation tự động.
- **Vấn đề / Bottlenecks:** trả lời theo phiên bản chính sách cũ; câu hỏi chứa từ khóa/mã số cụ thể (mã biểu mẫu, tên hệ thống) retrieve sai; không đo được chất lượng khi thay đổi prompt/model; PDF scan bị bỏ qua.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Structure-aware (theo heading) làm parent + child ~256 ký tự để retrieve — tài liệu chính sách có cấu trúc heading rõ, lab cho thấy return-parent tăng recall mạnh nhất (+0.17).
2. **Search retrieval:** Hybrid BM25 (underthesea) + Dense + RRF — BM25 bắt mã số/từ khóa chính xác, dense bắt paraphrase; thêm metadata `version`/`effective_date` để boost bản hiện hành.
3. **Reranking:** Có — bge-reranker-v2-m3 trên server có GPU (hoặc mMiniLM trên CPU nếu cần latency thấp); kết hợp ngưỡng score để dynamic top-k và MMR theo `source` cho câu multi-hop.
4. **Evaluation:** RAGAS 4 metrics trên test set ~50 câu chia theo loại (lookup, version, negation, multi-hop, numeric) chạy trong CI mỗi lần đổi prompt/model; judge dùng model mạnh hơn model generation để giảm false negative.
5. **Enrichment:** Combined single-call (contextual prepend + HyQA + auto-metadata) — chi phí 1 call/chunk, có cache theo hash nội dung để chỉ enrich chunk mới/thay đổi.

#### 3. Timeline triển khai
- **Tuần 1:** Xây test set 50 câu + chạy RAGAS cho pipeline hiện tại làm baseline; OCR các PDF scan.
- **Tuần 2:** Structure-aware + parent-child chunking, hybrid BM25 + dense + RRF; đo lại RAGAS.
- **Tuần 3:** Reranker + metadata version filter + query decomposition cho câu multi-hop; prompt tính toán cho câu numeric.
- **Tuần 4:** Enrichment combined + cache, latency breakdown/monitoring, tích hợp RAGAS vào CI và chốt ngưỡng chất lượng (≥ 0.85 mỗi metric) trước khi release.
