# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Trần Anh Vũ  
**Khóa:** K4 - Track 3A  

> **Cấu hình chạy:** LLM (answer + enrichment + RAGAS judge) = `gemini-3.5-flash-lite` (free tier, thay cho OpenAI).
> Do máy chỉ có 7.8GB RAM (không load nổi bge-m3 + bge-reranker-v2-m3 ~2.2GB/model), embedding và reranker được
> override qua `.env`: `intfloat/multilingual-e5-small` + `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
> Default trong `config.py` vẫn là bge-m3 / bge-reranker-v2-m3 theo đề bài. Cả baseline và production dùng chung
> embedding model và prompt sinh câu trả lời → so sánh công bằng.

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.9137 | **0.9257** | +0.0120 |
| Answer Relevancy | 0.8956 | **0.9421** | +0.0464 |
| Context Precision | 0.8500 | **0.9500** | +0.1000 |
| Context Recall | 0.7583 | **0.9250** | +0.1667 |

- **Naive:** paragraph chunking (51 chunks) + dense-only top-3.
- **Production:** hierarchical chunking (112 child / parent ≤ 2048) → M5 enrichment (contextual prepend + HyQA, 1 call/chunk)
  → BM25 (underthesea) + dense → RRF top-20 → cross-encoder rerank → **retrieve child, return parent** (top-3 parent khác nhau) → Gemini.
- Cải thiện lớn nhất ở **Context Recall (+0.17)**: trả về parent (cả section/tài liệu) thay vì đoạn 500 ký tự nên câu hỏi
  version (v2023 vs v2024) và multi-hop nhận đủ thông tin; câu tệ nhất của baseline ("Lương thử việc Junior", avg 0.25 —
  dense-only không lấy được `bang_luong_2024.md`) lên 0.94 ở production.

### Latency breakdown (production, CPU, trung bình / query)

| Bước | Thời gian |
|------|-----------|
| Hybrid search (BM25 + dense + RRF) | 33.8 ms |
| Rerank top-20 (cross-encoder) | 1059.1 ms |
| Generation (Gemini) | 1964.4 ms |
| **Tổng / query** | **3057.3 ms** |

| Bước build (1 lần) | Thời gian |
|--------------------|-----------|
| Chunking | 0.8 s |
| Enrichment 112 chunks (bị rate-limit free tier ~12 RPM) | 428.9 s |
| Indexing BM25 + Qdrant | 11.2 s |
| Load reranker | 7.9 s |
| RAGAS eval (20 câu × 4 metrics) | 386.8 s |

→ Reranker chiếm ~35% latency/query trên CPU; generation chiếm ~64%.

---

## Bottom-5 Failures (production)

### #1
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** v2024: từ 3 năm trở lên, +1 ngày mỗi 3 năm. v2023 cũ yêu cầu 5 năm.
- **Got:** "Theo chính sách hiện hành (2024)… thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm… (phiên bản cũ 2023: 5 năm)" — **đúng hoàn toàn**.
- **Worst metric:** context_precision = 0.0 (faithfulness 1.0, recall 1.0, relevancy 0.91)
- **Error Tree:** Output sai? → **Không** (đúng + đủ) → Context đúng? → Có (recall = 1.0, chứa cả v2024 và v2023) → Query OK? → Có → **Lỗi ở bước đánh giá (judge)/ranking**.
- **Root cause:** Thứ tự context sau rerank: (1) `nghi_phep_nam_v2023.md` — **bản cũ đã bị thay thế xếp hạng 1**, (2) `nghi_phep_nam_v2024.md`, (3) `nghi_phep_khong_luong.md` (không liên quan). Reranker chỉ nhìn độ khớp ngữ nghĩa nên không phân biệt phiên bản. Context precision tính có trọng số theo thứ hạng; judge `gemini-flash-lite` chấm "không hữu ích" cho cả 3 parent dài → precision 0 dù recall 1.0 — mâu thuẫn, nên một phần là false negative của LLM judge với context dài, phần còn lại do bản cũ đứng đầu + context nhiễu ở hạng 3.
- **Suggested fix:** Metadata filter/boost theo `version`/`effective_date` (ưu tiên bản hiện hành) trước khi chọn top-k; ngưỡng rerank score để bỏ context hạng 3 khi điểm thấp (dynamic top-k); trả về section (structure-aware) thay vì cả parent để context ngắn hơn; dùng judge mạnh hơn (`gemini-3.8-flash`) cho RAGAS.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày phép; lương Senior (P3-P4) 20–35 triệu/tháng.
- **Got:** Tính đúng 18 ngày phép, nhưng **không trả lời phần lương**.
- **Worst metric:** context_recall = 0.5 (faithfulness 0.75)
- **Error Tree:** Output sai? → Thiếu ý → Context đúng? → **Không đủ** — top-3 parent là `nghi_phep_nam_v2024`, `nghi_phep_nam_v2023`, `nghi_phep_khong_luong`; thiếu `bang_luong_2024.md` → **Lỗi ở Retrieval/Rerank (multi-hop)**.
- **Root cause:** Câu hỏi multi-hop 2 chủ đề trong 1 query. Hybrid search **có** lấy được chunk của `bang_luong_2024.md` trong top-20, nhưng cross-encoder chấm theo cả câu, vế "nghỉ phép + thâm niên" áp đảo nên chunk lương bị đẩy xuống hạng 12 → không vào top-3 parent.
- **Suggested fix:** Query decomposition (LLM tách thành 2 sub-query "ngày phép 9 năm thâm niên" và "lương Senior"), retrieve riêng rồi merge; hoặc đảm bảo đa dạng nguồn (MMR theo `source`) khi chọn top-k parent.

### #3
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo v2.0 bắt buộc MFA cho email, VPN, hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** "Có, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và các hệ thống nội bộ…" — đúng nhưng không nhắc v1.0.
- **Worst metric:** context_recall = 0.5
- **Error Tree:** Output sai? → Thiếu ý phụ (so sánh version) → Context đúng? → Chỉ có `mat_khau_v2.md`; claim "v1.0 không yêu cầu MFA" là suy ra từ việc v1 **không nhắc** MFA → không attributable được → **Lỗi do ground truth chứa thông tin dạng phủ định/ngầm định**.
- **Root cause:** Ground truth có claim "chính sách cũ không yêu cầu MFA" — muốn chứng minh cần context `mat_khau_v1.md`, nhưng v1 không chứa từ "MFA" nên cả BM25 lẫn dense không kéo v1 lên.
- **Suggested fix:** Metadata `version`/`supersedes` khi index (M5 auto-metadata) → khi retrieve một tài liệu versioned, tự kèm phiên bản liền trước để LLM so sánh; prompt yêu cầu nêu thay đổi so với bản cũ nếu có.

### #4
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Hạn 15 ngày → quá hạn 5 ngày; phí 2%/tháng × 15tr = 300.000đ/tháng (~50.000đ pro-rata cho 5 ngày).
- **Got:** Chỉ nhắc lại quy định "quá 15 ngày bị tính 2%/tháng, khấu trừ vào lương" — **không tính ra con số**.
- **Worst metric:** context_recall = 0.5
- **Error Tree:** Output sai? → Thiếu phép tính → Context đúng? → **Có** (`tam_ung.md` đủ thông tin 15 ngày, 2%/tháng) → Query OK? → Có → **Lỗi ở Generation (reasoning)**. (Recall 0.5 vì judge không attribute được các con số tính toán 300.000đ/50.000đ vào context.)
- **Root cause:** Prompt "trả lời ngắn gọn, chỉ dựa trên context" khiến model flash-lite chỉ trích dẫn quy định, không áp dụng số liệu trong câu hỏi.
- **Suggested fix:** Bổ sung vào system prompt: "Nếu câu hỏi có số liệu cụ thể, hãy áp dụng quy định và tính ra kết quả từng bước"; hoặc few-shot cho câu hỏi numeric.

### #5
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Chưa được hưởng gói PVI, chỉ tham gia BHXH bắt buộc.
- **Got:** "Không. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI." — đúng, có trong `thu_viec.md`.
- **Worst metric:** faithfulness = 0.5
- **Error Tree:** Output sai? → **Không** → Context đúng? → Có (recall 1.0, precision 1.0) → **Lỗi ở judge**.
- **Root cause:** RAGAS tách câu trả lời thành 2 statement ("Không." và "chưa được hưởng gói PVI"); statement "Không." đứng riêng thiếu chủ ngữ nên judge chấm không suy ra được từ context → 1/2 = 0.5. Câu trả lời ngắn kiểu Yes/No dễ bị phạt faithfulness.
- **Suggested fix:** Prompt yêu cầu trả lời thành câu đầy đủ (không mở đầu bằng "Có."/"Không." đứng riêng); bổ sung ý BHXH để đầy đủ hơn.

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?" (multi-hop, failure #2)

**Error Tree walkthrough:**
1. Output đúng? → **Một nửa**: 18 ngày phép đúng (15 + 9÷3), thiếu khoảng lương 20–35 triệu.
2. Context đúng? → **Không đủ**: top-3 parent sau rerank đều là tài liệu nghỉ phép (v2024, v2023, không lương), không có `bang_luong_2024.md` → context_recall 0.5. Chunk lương **có** trong top-20 của hybrid search nhưng bị reranker xếp hạng 12.
3. Query rewrite OK? → **Không có bước rewrite**: 1 query chứa 2 intent, intent "nghỉ phép" lấn át intent "lương"; reranker chấm theo cả câu nên ưu tiên chunk nghỉ phép.
4. Fix ở bước: **Query understanding / retrieval** — decomposition thành sub-queries + đảm bảo đa dạng `source` trong top-k. Generation không lỗi (đã tính đúng phần có context, không bịa phần lương).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu multi-hop + MMR theo `source` khi chọn top-3 parent (fix #2, #3).
- Prompt cho câu numeric: bắt buộc áp dụng số liệu và tính từng bước (fix #4).
- Dynamic top-k theo rerank score + trả về section thay vì parent dài (fix #1), chạy lại RAGAS với judge `gemini-3.8-flash` để giảm false negative của judge.
- OCR cho 2 PDF scan (`BCTC.pdf`, `Nghi_dinh_so_13-2023...pdf`) hiện bị bỏ qua.
