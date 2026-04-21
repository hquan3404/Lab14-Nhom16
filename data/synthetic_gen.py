import json
import asyncio
import os
import sys
import random
from typing import List, Dict
from dotenv import load_dotenv

# Thêm đường dẫn gốc để import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.gemini_provider import GeminiProvider

# Prompt để sinh câu hỏi
SDG_PROMPT = """Bạn là một chuyên gia tạo dữ liệu kiểm thử cho hệ thống AI Lịch sử.
Dựa trên đoạn văn bản lịch sử dưới đây, hãy tạo ra {num_per_chunk} cặp Câu hỏi và Câu trả lời mẫu.

Yêu cầu:
1. Câu hỏi phải đa dạng: câu hỏi sự kiện, câu hỏi nguyên nhân, câu hỏi ý nghĩa.
2. Câu trả lời (Expected Answer) phải chi tiết, chính xác dựa trên văn bản.
3. Phải bao gồm "context" là đoạn văn bản gốc dùng để đặt câu hỏi.
4. Định dạng trả về đúng JSON list.

Văn bản gốc:
{text}

Trả về định dạng JSON duy nhất như sau:
[
  {{"question": "...", "expected_answer": "...", "context": "...", "metadata": {{"type": "fact-check", "difficulty": "normal"}}}}
]
"""

async def generate_50_cases():
    load_dotenv()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Khởi tạo Gemini
    llm = GeminiProvider(
        model_name="gemini-3.1-flash-lite-preview", 
        api_key=os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY"),
        base_url=os.getenv("BASE_URL")
    )
    
    data_path = os.path.join(root_dir, "data", "data.md")
    if not os.path.exists(data_path):
        print(f"❌ Không tìm thấy file {data_path}")
        return

    print(f"📖 Đang đọc dữ liệu từ {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Chia nhỏ văn bản để sinh câu hỏi (mỗi đoạn khoảng 2000 chữ)
    chunks = [content[i:i+2000] for i in range(0, len(content), 10000)]
    random.shuffle(chunks) # Lấy ngẫu nhiên các đoạn
    chunks = chunks[:10]   # Lấy 10 đoạn tiêu biểu

    golden_set = []
    print(f"🚀 Bắt đầu sinh 50 câu hỏi (10 đoạn x 5 câu)...")

    for i, chunk in enumerate(chunks):
        print(f"  - Đang xử lý đoạn {i+1}/10...")
        prompt = SDG_PROMPT.format(num_per_chunk=5, text=chunk)
        
        try:
            response = llm.generate(prompt=prompt, system_prompt="Bạn là trợ lý chuyên về Lịch sử Việt Nam. Chỉ trả về JSON.")
            raw_json = response["content"]
            
            # Làm sạch JSON nếu LLM trả về kèm markdown
            if "```json" in raw_json:
                raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_json:
                raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
            cases = json.loads(raw_json)
            golden_set.extend(cases)
            
            if len(golden_set) >= 50:
                break
        except Exception as e:
            print(f"    ⚠️ Lỗi tại đoạn {i+1}: {e}")
            continue

    # Lưu kết quả
    output_path = os.path.join(root_dir, "data", "golden_set.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in golden_set[:50]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n✅ Đã hoàn thành! Đã tạo {len(golden_set[:50])} câu hỏi tại {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_50_cases())
