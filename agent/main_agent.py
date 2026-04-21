import asyncio
import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Ensure we can import from the 'src' directory in the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.agent.agent import HistoryAgent
    from src.tools.tool_registry import ToolRegistry
    from src.core.gemini_provider import GeminiProvider
    from src.core.openai_provider import OpenAIProvider
except ImportError:
    # Fallback for localized imports if needed
    print("Warning: Could not import HistoryAgent from src. Ensure 'src' directory exists in root.")

class MainAgent:
    """
    Hệ thống Chatbot Lịch sử được tích hợp từ Day 3.
    Sử dụng kiến trúc Planner Agent với các công cụ tra cứu chuyên sâu.
    """
    def __init__(self):
        # Nạp .env từ thư mục gốc của Lab14
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dotenv_path = os.path.join(root_dir, ".env")
        load_dotenv(dotenv_path)
        
        # Sửa lỗi 401: Nếu API_KEY trống, thử lấy từ GEMINI_API_KEY
        if not os.getenv("API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["API_KEY"] = os.getenv("GEMINI_API_KEY")

        self.name = "HistoryAgent-v2-Planner"
        
        # Khởi tạo LLM Provider
        provider_type = os.getenv("DEFAULT_PROVIDER", "google")
        model_name = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
        
        if provider_type == "google":
            self.llm = GeminiProvider(
                model_name=model_name,
                api_key=os.getenv("API_KEY"),
                base_url=os.getenv("BASE_URL")
            )
        else:
            self.llm = OpenAIProvider(
                model_name=model_name,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
        # Khởi tạo Tool Registry và Agent
        # file data.md đã được copy vào thư mục data/ của Lab14
        self.tool_registry = ToolRegistry(data_path="data/data.md")
        self.agent = HistoryAgent(llm=self.llm, tool_registry=self.tool_registry)

    async def query(self, question: str) -> Dict:
        """
        Thực hiện xử lý câu hỏi qua hệ thống Agent và trả về định dạng chuẩn cho Eval.
        """
        # Chạy Agent trong thread riêng để không block event loop (do agent.run là sync)
        loop = asyncio.get_event_loop()
        try:
            # history_agent.run returns {"answer": str, "contexts": list}
            result = await loop.run_in_executor(None, self.agent.run, question)
            
            # Trả về format yêu cầu bởi BenchmarkRunner
            return {
                "answer": result["answer"],
                "contexts": result["contexts"],
                "metadata": {
                    "model": self.llm.model_name,
                    "provider": os.getenv("DEFAULT_PROVIDER", "google"),
                    "type": "planner_agent"
                }
            }
        except Exception as e:
            return {
                "answer": f"Lỗi xử lý: {str(e)}",
                "contexts": [],
                "metadata": {"error": True}
            }

if __name__ == "__main__":
    agent = MainAgent()
    async def test():
        resp = await agent.query("Trận Điện Biên Phủ diễn ra khi nào?")
        print(f"Agent Response: {resp['answer']}")
    asyncio.run(test())
