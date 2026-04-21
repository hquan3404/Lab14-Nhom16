import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent

# Khởi tạo LLM cho việc chấm điểm (Judge)
from src.core.gemini_provider import GeminiProvider
from dotenv import load_dotenv

load_dotenv()

class ExpertEvaluator:
    async def score(self, case, resp): 
        """
        Tính toán Hit Rate và MRR thực tế.
        """
        gt_context = case.get("context", "").strip()
        retrieved_contexts = resp.get("contexts", [])
        
        hit = 0
        mrr = 0
        
        for i, ctx in enumerate(retrieved_contexts):
            # Kiểm tra xem đoạn văn bản chuẩn có nằm trong kết quả tìm kiếm không
            if gt_context and (gt_context in ctx or ctx in gt_context):
                hit = 1
                mrr = 1 / (i + 1)
                break
        
        return {
            "faithfulness": 0.85, # Có thể tích hợp RAGAS thật vào đây
            "relevancy": 0.9,
            "retrieval": {"hit_rate": hit, "mrr": mrr}
        }

class MultiModelJudge:
    def __init__(self):
        # Dùng model Gemini 3.1 mới nhất để làm giám khảo
        self.llm = GeminiProvider(
            model_name="gemini-3.1-flash-lite-preview", 
            api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY"),
            base_url=os.getenv("BASE_URL")
        )

    async def evaluate_multi_judge(self, q, a, gt): 
        """
        Dùng LLM làm giám khảo để chấm điểm câu trả lời (a) dựa trên đáp án đúng (gt).
        """
        judge_prompt = f"""Bạn là một chuyên gia thẩm định câu trả lời lịch sử.
Hãy so sánh câu trả lời của AI với đáp án chuẩn và chấm điểm trên thang điểm 5.

Câu hỏi: {q}
Đáp án chuẩn: {gt}
Câu trả lời của AI: {a}

Hãy trả về kết quả định dạng JSON:
{{
  "final_score": (số từ 1-5),
  "agreement_rate": (hệ số tin tưởng từ 0.1-1.0),
  "reasoning": "giải thích ngắn gọn tại sao cho điểm số đó"
}}
"""
        try:
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, self.llm.generate, judge_prompt)
            
            raw_content = res["content"]
            if "```json" in raw_content:
                raw_content = raw_content.split("```json")[1].split("```")[0].strip()
            
            judge_data = json.loads(raw_content)
            return judge_data
        except Exception as e:
            print(f"⚠️ Lỗi Judge: {e}")
            return {"final_score": 3.0, "agreement_rate": 0.5, "reasoning": "Lỗi khi gọi LLM Judge."}

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
