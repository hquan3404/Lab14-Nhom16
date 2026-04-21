# 📝 Báo Cáo Thu Hoạch Cá Nhân - Lab 14

**Họ và tên:** Đào Danh Đăng Phụng  
**Mã học viên:** 2A202600358  
**Nhóm:** 16  
**Vai trò dự án:** Data & System Integration

---

## 1. Công việc đã thực hiện
Trong bài Lab 14, tôi phụ trách hai phần việc chính là xây dựng bộ dữ liệu `golden_set` và kết nối lại hệ thống core từ lab cũ sang bài lab mới. Cụ thể:
- **Tạo golden_set phục vụ benchmark:** Tôi tham gia chuẩn bị và sinh bộ dữ liệu `data/golden_set.jsonl` để hệ thống có đầu vào chuẩn cho bước đánh giá. Đây là nền tảng quan trọng để `main.py` có thể chạy benchmark và tính các chỉ số như Retrieval/QA metrics.
- **Kết nối hệ thống từ lab cũ sang lab mới:** Tôi hỗ trợ chuyển phần core Day3 sang cấu trúc của Lab 14, đảm bảo `agent/main_agent.py` có thể đóng vai trò entrypoint để benchmark gọi được agent cũ mà không cần viết lại toàn bộ logic từ đầu.
- **Kiểm tra tính tương thích dữ liệu:** Tôi rà soát `data/data.md` và các file liên quan để bảo đảm dữ liệu nguồn dùng cho retrieval/chunking phù hợp với pipeline hiện tại, tránh lỗi lệch format giữa bộ dữ liệu cũ và framework benchmark mới.
- **Hỗ trợ chạy end-to-end workflow:** Sau khi nối hệ thống, tôi phối hợp chạy các bước sinh dữ liệu, benchmark và kiểm tra cấu trúc đầu ra để đảm bảo repo có thể tạo ra `reports/summary.json` và `reports/benchmark_results.json` đúng yêu cầu nộp bài.

## 2. Bài học kinh nghiệm
- **Dữ liệu quyết định chất lượng đánh giá:** Tôi nhận ra rằng một bộ `golden_set` tốt không chỉ là có đủ số lượng câu hỏi, mà còn phải có context rõ ràng, đúng nguồn và phản ánh được các tình huống dễ sai của hệ thống.
- **Tích hợp hệ thống cần ưu tiên tính ổn định:** Khi chuyển core từ lab cũ sang lab mới, điều quan trọng nhất là giữ được contract đầu vào/đầu ra ổn định, đặc biệt là với entrypoint `MainAgent.query()` để benchmark không bị đứt gãy.
- **Tách rõ phần dữ liệu và phần logic đánh giá:** Việc chuẩn hóa pipeline giúp tôi hiểu rõ hơn vai trò của từng lớp: data generation, retrieval, agent core và benchmark engine. Khi mỗi phần tách bạch, việc debug và mở rộng sau này sẽ dễ hơn rất nhiều.

## 3. Hướng phát triển & Cải tiến
- **Nâng chất lượng golden_set:** Trong các lần làm tiếp theo, tôi muốn mở rộng thêm các câu hỏi khó, câu hỏi nhiễu và các case biên để bộ dữ liệu phản ánh tốt hơn hành vi thực tế của agent.
- **Tự động hóa kiểm tra tích hợp:** Có thể bổ sung thêm script kiểm tra nhanh sự tồn tại của file dữ liệu, report và các biến môi trường để giảm lỗi khi chạy benchmark.
- **Chuẩn hóa quy trình chuyển lab:** Tôi muốn xây dựng một checklist rõ ràng hơn cho việc chuyển core từ bài cũ sang bài mới, যাতে mỗi lần triển khai sẽ ít phát sinh lỗi import, config hoặc format dữ liệu hơn.
