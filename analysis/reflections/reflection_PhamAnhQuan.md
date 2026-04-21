# 📝 Báo Cáo Thu Hoạch Cá Nhân - Lab 14

**Họ và tên:** Phạm Anh Quân  
**Nhóm:** 16  
**Vai trò dự án:** AI Evaluation

---

## 1. Công việc đã thực hiện
Trong dự án này, tôi chịu trách nhiệm thiết kế toàn bộ quy trình và engine đánh giá hệ thống Chatbot Lịch sử. Các đầu việc cụ thể bao gồm:
- **Thiết kế Eval Engine thực tế:** Thay thế logic giả lập ban đầu bằng các thuật toán đo lường thực tế trong `main.py`. Tôi đã xây dựng bộ logic tính toán **Hit Rate** và **MRR** để kiểm chứng hiệu quả của giai đoạn Retrieval (Tìm kiếm).
- **Phát triển AI Judging:** Thiết lập và cấu hình lớp `MultiModelJudge` sử dụng model **Gemini 3.1 Flash Lite Preview**. Tôi đã thiết kế các tiêu chí chấm điểm (Prompt) để AI đóng vai trò như một chuyên gia lịch sử, thẩm định câu trả lời dựa trên độ chính xác và tính logic so với đáp án chuẩn.
- **Tối ưu hóa quy trình Benchmark:** Cấu hình Async Runner giúp việc đánh giá đồng thời 50 câu hỏi diễn ra nhanh chóng, đảm bảo tính ổn định và xử lý lỗi ngoai lệ (Exception handling) khi gọi các API từ xa.
- **Phân tích Regression & Metrics:** Trực tiếp thực hiện các lượt chạy Benchmark để so sánh hiệu năng giữa phiên bản gốc và phiên bản nâng cấp model, từ đó đưa ra phán quyết "Approve" dựa trên chỉ số Delta (+0.28).

## 2. Bài học kinh nghiệm
- **Tư duy đo lường (Measurement Mindset):** Tôi hiểu ra rằng một chatbot "trông có vẻ hay" là chưa đủ. Chỉ khi có các chỉ số định lượng như Hit Rate đạt mức 1.0 và Avg Score trên 4.5 như kết quả vừa rồi, chúng ta mới thực sự tin tưởng được vào chất lượng của Agent.
- **Sự khác biệt của các thế hệ Model:** Qua việc thử nghiệm, tôi nhận thấy Gemini 3.1 Flash Lite Preview không chỉ nhanh hơn mà còn có khả năng chấm điểm tinh tế hơn, nhận diện được những sai sót nhỏ về mặt lịch sử mà các model cũ thường bỏ qua.
- **Tầm quan trọng của Ground Truth:** Việc có một bộ dữ liệu Golden Set chất lượng (với context rõ ràng) là xương sống của mọi quy trình đánh giá. Nếu context sai, toàn bộ các chỉ số Hit Rate hay Faithfulness đều vô nghĩa.

## 3. Hướng phát triển & Cải tiến
- **Tự động hóa báo cáo sâu:** Phát triển script tự động phân loại các câu hỏi bị điểm thấp theo từng danh mục lỗi (ví dụ: lỗi trích dẫn, lỗi diễn đạt) để giảm bớt thời gian phân tích thủ công.
- **Mở rộng Judge Model:** Thử nghiệm cơ chế chấm điểm chéo (Cross-grading) giữa hai model khác nhau để loại bỏ hoàn toàn sự thiên kiến (bias) của một model duy nhất trong quá trình đánh giá.
- **Kiểm soát Cost hiệu quả:** Đề xuất các giải pháp sử dụng Local LLM cho một số chỉ số đơn giản để giảm thiểu 30-50% chi phí API khi quy mô đánh giá tăng lên hàng ngàn câu hỏi.
