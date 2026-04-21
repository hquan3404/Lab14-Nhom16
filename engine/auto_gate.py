import json
import os
import sys
from typing import Dict, Any, List, Optional

# Cấu hình ngưỡng (Thresholds)
DEFAULT_THRESHOLDS = {
    "avg_score": 4.2,         # Điểm trung bình từ LLM-Judge (thang 5)
    "hit_rate": 0.65,         # Tỷ lệ truy xuất chính xác (Hit Rate)
    "agreement_rate": 0.85,    # Độ đồng thuận giữa các Judge
    "avg_latency": 25.0       # Độ trễ trung bình tối đa (giây)
}

class AutoGate:
    def __init__(
        self, 
        summary_path: str = "reports/summary.json",
        results_path: str = "reports/benchmark_results.json",
        baseline_path: str = "reports/baseline_summary.json"
    ):
        self.summary_path = summary_path
        self.results_path = results_path
        self.baseline_path = baseline_path
        self.thresholds = DEFAULT_THRESHOLDS

    def load_json(self, path: str) -> Optional[Dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def calculate_latency(self) -> float:
        results = self.load_json(self.results_path)
        if not results:
            return 0.0
        latencies = [r.get("latency", 0) for r in results]
        return sum(latencies) / len(latencies) if latencies else 0.0

    def run(self):
        print("🚀 [Auto-Gate] Đang kiểm tra điều kiện phát hành (Release Gating)...\n")
        
        summary = self.load_json(self.summary_path)
        if not summary:
            print("❌ Lỗi: Không tìm thấy file summary.json. Hãy chạy benchmark trước.")
            sys.exit(1)

        metrics = summary.get("metrics", {})
        avg_latency = self.calculate_latency()
        
        # 1. Kiểm tra các ngưỡng tuyệt đối
        checks = [
            ("Avg Score", metrics.get("avg_score", 0), self.thresholds["avg_score"], ">="),
            ("Hit Rate", metrics.get("hit_rate", 0), self.thresholds["hit_rate"], ">="),
            ("Agreement", metrics.get("agreement_rate", 0), self.thresholds["agreement_rate"], ">="),
            ("Avg Latency", avg_latency, self.thresholds["avg_latency"], "<=")
        ]

        all_passed = True
        print(f"{'Tiêu chí':<20} | {'Thực tế':<10} | {'Ngưỡng':<10} | {'Trạng thái'}")
        print("-" * 60)

        for name, value, threshold, op in checks:
            passed = False
            if op == ">=":
                passed = value >= threshold
            elif op == "<=":
                passed = value <= threshold
            
            status = "✅ PASS" if passed else "❌ FAIL"
            if not passed:
                all_passed = False
            
            print(f"{name:<20} | {value:<10.3f} | {threshold:<10.3f} | {status}")

        # 2. Phân tích Delta (nếu có baseline)
        baseline = self.load_json(self.baseline_path)
        if baseline:
            print("\n📊 [Delta Analysis] So sánh với phiên bản cũ (Baseline):")
            old_score = baseline.get("metrics", {}).get("avg_score", 0)
            new_score = metrics.get("avg_score", 0)
            delta = new_score - old_score
            
            delta_status = "✅ Cải thiện" if delta >= 0 else "⚠️ Suy giảm"
            print(f"Delta Score: {delta:+.3f} ({delta_status})")
            
            # Nếu giảm quá 5% điểm số, coi như fail gate
            if delta < -0.2: # Ngưỡng suy giảm tối đa cho phép
                print("❌ CẢNH BÁO: Hiệu năng suy giảm vượt mức cho phép!")
                all_passed = False

        print("\n" + "="*40)
        if all_passed:
            print("🏆 KẾT LUẬN: [ RELEASE ]")
            print("Hệ thống đạt đủ tiêu chuẩn chất lượng để triển khai.")
            sys.exit(0)
        else:
            print("🛑 KẾT LUẬN: [ ROLLBACK ]")
            print("Hệ thống KHÔNG đạt đủ tiêu chuẩn. Hãy kiểm tra các tiêu chí bị FAIL.")
            sys.exit(1)

if __name__ == "__main__":
    gate = AutoGate()
    gate.run()
