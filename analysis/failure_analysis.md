# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** 47/3
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.85
    - Relevancy: 0.9
- **Điểm LLM-Judge trung bình:** 4.73/ 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Knowledge Gap / Retrieval | 2 | Retriever không tìm thấy context hoặc AI không trích xuất được số liệu cụ thể |
| Incomplete / Extraction | 1 | AI trả lời chung chung theo danh mục, không liệt kê đúng các sản phẩm cụ thể |
| Hallucination | 0 | Không có trường hợp bịa đặt thông tin |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)
 
 ### Case #1: Thiếu hụt thông tin về số lượng công nhân vận tải năm 1969
 1. **Symptom:** Agent trả lời có các ý liên quan nhưng lại bỏ qua đáp án đúng là 175.500.
 2. **Why 1:** LLM không nhận được dữ liệu này trong context được truy xuất.
 3. **Why 2:** Retriever không tìm thấy đoạn văn bản chứa thông tin cụ thể này (Hit rate = 0).
 4. **Why 3:** Thông tin có thể nằm sâu trong bảng biểu hoặc file tài liệu lớn bị cắt đoạn (chunking) không hợp lý.
 5. **Root Cause:** Chiến lược truy xuất dữ liệu (Retrieval strategy) chưa hiệu quả với các câu hỏi về số liệu thống kê chi tiết.
 
 ### Case #2: Trả lời không đầy đủ chi tiết về các sản phẩm nông nghiệp
 1. **Symptom:** Agent liệt kê các lĩnh vực chung chung (lương thực, thực phẩm, thủy lợi) thay vì 3 sản phẩm cụ thể.
 2. **Why 1:** LLM tóm tắt các nhóm ngành thay vì trích xuất đúng thực thể (entity extraction) rời rạc.
 3. **Why 2:** Prompt không nhấn mạnh việc liệt kê tên sản phẩm cụ thể dẫn đến Agent ưu tiên tính tổng quát.
 4. **Root Cause:** Prompt Engineering chưa đủ độ cưỡng chế (constraint) để đảm bảo Agent trả lời đúng định dạng danh sách thực thể.
 
 ### Case #3: Không truy xuất được diễn biến giải phóng Trà Vinh ngày 30/4/1975
 1. **Symptom:** Agent trả lời "chưa có thông tin" và dẫn sang các sự kiện không liên quan.
 2. **Why 1:** Một lần nữa, context được trả về cho LLM hoàn toàn không chứa đoạn văn cần thiết.
 3. **Why 2:** Retriever bị sai lệch từ khóa (keyword mismatch) hoặc thông tin trong source data quá mỏng.
 4. **Root Cause:** Độ phủ của dữ liệu nguồn (Knowledge Source) hoặc khả năng tìm kiếm ngữ nghĩa (Semantic Search) chưa bao quát được toàn bộ các cột mốc lịch sử cụ thể.

## 4. Kế hoạch cải tiến (Action Plan)

Dựa trên các nguyên nhân gốc rễ đã xác định, dưới đây là kế hoạch hành động để tối ưu hóa hệ thống:

### 4.1. Tối ưu hóa xử lý dữ liệu (Data Pre-processing)
- **Semantic Chunking**: Thay đổi từ Fixed-size sang Semantic Chunking. 
    - *Lý do*: Đảm bảo các đoạn văn bản có cùng chủ đề hoặc các bảng số liệu không bị chia cắt giữa chừng, giúp Retriever dễ dàng lấy được toàn bộ ngữ cảnh liên quan (Khắc phục Case #1 và #3).
- **Metadata Enhancement**: Thêm các tag về thực thể (Entity Tags) và mốc thời gian vào metadata của từng chunk.

### 4.2. Nâng cấp quy trình Truy xuất (Retrieval)
- **Hybrid Search (BM25 + Vector)**: Điều chỉnh trọng số (Alpha) giữa tìm kiếm từ khóa và tìm kiếm ngữ nghĩa.
- **Reranking Step**: Thêm bước Reranking (sử dụng Cohere hoặc LLM-as-a-reranker) sau khi có kết quả thô.
    - *Lý do*: Chọn lọc chính xác hơn 3-5 đoạn văn bản thực sự chứa câu trả lời trong số 20 kết quả trả về ban đầu, giảm nhiễu cho bước tổng hợp.

### 4.3. Cải thiện mô hình Agent (Prompt Engineering)
- **Constraint-Based Prompting**: Cập nhật System Prompt với các ràng buộc cứng:
    - Yêu cầu trích xuất danh sách thực thể rời rạc nếu câu hỏi yêu cầu "Kể tên", "Liệt kê".
    - Nhấn mạnh việc ưu tiên số liệu tuyệt đối và không được phép bỏ qua các chi tiết nhỏ trong context.
    - *Lý do*: Khắc phục tình trạng trả lời quá tổng quát (Khắc phục Case #2).

### 4.4. Lộ trình triển khai (Timeline)
- **Giai đoạn 1 (Tuần 1)**: Cập nhật Prompt và triển khai Hybrid Search.
- **Giai đoạn 2 (Tuần 2)**: Re-index dữ liệu bằng Semantic Chunking và tích hợp Reranking.
- **Giai đoạn 3 (Tuần 3)**: Chạy lại toàn bộ Benchmark để xác nhận sự cải thiện của các chỉ số.
