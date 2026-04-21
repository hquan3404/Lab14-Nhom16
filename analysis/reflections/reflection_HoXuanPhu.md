# 📝 Báo Cáo Thu Hoạch Cá Nhân - Lab 14

**Họ và tên:** Hồ Xuân Phú  
**Nhóm:** 16  
**Vai trò dự án:** AI Evaluation & Analysis

---

## 1. Công việc đã thực hiện
Trong dự án này, tôi chịu trách nhiệm chính về việc vận hành hệ thống đánh giá và phân tích chất lượng của AI Agent. Các công việc cụ thể tôi đã hoàn thành bao gồm:

- **Thực thi Benchmark hệ thống:** Trực tiếp vận hành script `main.py` để chạy đánh giá trên toàn bộ 50 test cases của bộ dữ liệu lịch sử. Đảm bảo quy trình diễn ra ổn định, kiểm soát các phản hồi từ mô hình Gemini và giám sát hiệu năng hệ thống.
- **Tính toán và Phân tích số liệu:** 
    - Triển khai script `calculate_metrics.py` để tự động hóa việc tính toán các chỉ số RAGAS trung bình (Faithfulness: **0.85**, Relevancy: **0.90**).
    - Trích xuất và phân tích các chỉ số từ `reports/summary.json`, ghi nhận kết quả **LLM-Judge trung bình đạt 4.73/5.0**, **Hit Rate đạt 0.7** và **Agreement Rate đạt 0.938**.
- **Xây dựng cơ chế Auto-Gate (`engine/auto_gate.py`):** Thiết lập hệ thống tự động kiểm soát chất lượng (Gating). Hệ thống này tự động phân tích các chỉ số từ benchmark để đưa ra quyết định "RELEASE" hoặc "ROLLBACK" dựa trên các ngưỡng (thresholds) định sẵn về độ chính xác, độ truy xuất và hiệu năng (latency).
- **Hoàn thiện Báo cáo Phân tích Thất bại (`failure_analysis.md`):**
    - Thực hiện **Failure Clustering** để phân loại các lỗi truy xuất (Retrieval) và trích xuất (Extraction).
    - Áp dụng phương pháp **5 Whys** để đi sâu vào các case tệ nhất (như việc bỏ sót con số 175.500) nhằm tìm ra nguyên nhân gốc rễ từ Chunking và Prompting.
    - Đề xuất các giải pháp kỹ thuật trong **Kế hoạch cải tiến (Action Plan)** như Semantic Chunking và Reranking.

## 2. Bài học kinh nghiệm
- **Tư duy dựa trên dữ liệu (Data-driven):** Tôi nhận thấy rằng việc tối ưu Agent không thể chỉ dựa trên cảm tính. Các con số cụ thể từ RAGAS và Hit Rate cung cấp cái nhìn khách quan về việc hệ thống đang gặp vấn đề ở khâu tìm kiếm (Retrieval) hay khâu tạo câu trả lời (Generation).
- **Kỹ năng phân tích Root Cause:** Việc thực hiện phân tích 5 Whys giúp tôi rèn luyện tư duy phản biện, không chỉ dừng lại ở việc thấy lỗi mà phải hiểu rõ tại sao lỗi đó xảy ra (do cách cắt đoạn văn bản hay do giới hạn của mô hình).
- **Tầm quan trọng của Benchmark tự động:** Một quy trình đánh giá tự động và nhanh chóng giúp tiết kiệm rất nhiều thời gian so với kiểm thử thủ công, đồng thời cho phép thử nghiệm và so sánh giữa nhiều phiên bản Agent một cách chính xác.

## 3. Hướng phát triển & Cải tiến
- **Nâng cấp công cụ đo lường:** Tích hợp trực tiếp việc tính toán metrics vào pipeline `main.py` để không cần chạy script rời.
- **Tối ưu hóa Retrieval:** Thực hiện triển khai Semantic Chunking để khắc phục triệt để các lỗi về Knowledge Gap mà tôi đã phát hiện trong báo cáo phân tích.
- **Mở rộng Judge Model:** Thử nghiệm sử dụng thêm các model judge khác để tăng tính khách quan và đối chiếu độ chính xác của quá trình chấm điểm.
