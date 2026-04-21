"""
History Agent - AI Agent with Planning, Tool Selection, and Synthesis.

Pipeline:
1. Planner phân tích ý định câu hỏi
2. Planner chia thành sub-questions nếu cần
3. Planner chọn tool phù hợp cho từng sub-question
4. Nhận kết quả từ tool
5. Tổng hợp thành câu trả lời cuối
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.llm_provider import LLMProvider
from src.tools.tool_registry import ToolRegistry
from src.telemetry.logger import logger


@dataclass
class SubTask:
    """A single sub-task in the execution plan."""
    step: int
    tool: str
    query: str
    purpose: str
    result: str = ""


PLANNER_SYSTEM_PROMPT = """Bạn là một Planner thông minh cho hệ thống hỏi đáp lịch sử Việt Nam.

Nhiệm vụ của bạn: Phân tích câu hỏi của người dùng, chia thành các bước nhỏ (sub-tasks), và chọn tool phù hợp cho mỗi bước.

## Các tool có sẵn:
{tool_descriptions}

## Quy tắc chọn tool:
- **search_docs**: Khi cần tìm thông tin tổng quát (vì sao, như thế nào, ý nghĩa, hạn chế, mục tiêu, vai trò)
- **build_timeline**: Khi cần diễn biến theo thời gian (mốc quan trọng, trình tự, trước/sau)
- **lookup_entity**: Khi cần tra cứu thực thể cụ thể (X là gì, X đóng vai trò gì, chiến lược Y)

## Quy tắc chia sub-tasks:
- Câu hỏi đơn giản (1 ý): tạo 1 sub-task
- Câu hỏi phức tạp (nhiều ý hoặc cần nhiều loại thông tin): tạo 2-4 sub-tasks
- Mỗi sub-task phải có query CỤ THỂ bằng tiếng Việt, không lặp lại câu hỏi gốc

## Output format (JSON):
```json
{{
  "intent": "Tóm tắt ngắn ý định câu hỏi",
  "sub_tasks": [
    {{
      "step": 1,
      "tool": "tool_name",
      "query": "câu truy vấn cụ thể cho tool",
      "purpose": "mục đích của bước này"
    }}
  ]
}}
```

CHỈ trả về JSON, không thêm gì khác."""


SYNTHESIZER_SYSTEM_PROMPT = """Bạn là một chuyên gia lịch sử Việt Nam. Nhiệm vụ của bạn là tổng hợp các kết quả tìm kiếm thành câu trả lời hoàn chỉnh.

## Quy tắc:
1. CHỈ dựa trên thông tin đã thu thập được bên dưới.
2. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc.
3. Sử dụng số liệu, ngày tháng cụ thể nếu có.
4. Nếu thông tin không đủ, nói rõ phần nào chưa tìm thấy.
5. Tổ chức câu trả lời theo logic mạch lạc, dùng heading/bullet points.

## Thông tin đã thu thập:
{collected_info}
"""


class HistoryAgent:
    """
    Agent with Planning capabilities.

    Flow:
    1. PLAN: LLM analyzes the question → generates execution plan (sub-tasks + tools)
    2. EXECUTE: Run each sub-task's tool with its query
    3. SYNTHESIZE: LLM combines all results into a coherent final answer
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        max_sub_tasks: int = 5,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.max_sub_tasks = max_sub_tasks

    def run(self, question: str) -> Dict[str, Any]:
        """
        Full agent pipeline: Plan → Execute → Synthesize.

        Args:
            question: User's question about Vietnamese history.

        Returns:
            Dict containing 'answer' and 'contexts' (list of retrieved strings).
        """
        logger.log_event("AGENT_START", {"question": question[:100]})

        # Thêm bước kiểm tra ý định (Intent Check) để vượt rào (bypass) Planner nếu là chào hỏi
        intent_system_prompt = (
            "Bạn là một bộ lọc nội dung. Hãy phân loại câu hỏi của người dùng.\n"
            "1. 'GREETING': Câu chào hỏi (xin chào, hi, hello, chào buổi sáng).\n"
            "2. 'IRRELEVANT': Câu hỏi hoàn toàn không liên quan đến lịch sử Việt Nam (ví dụ về toán, lập trình).\n"
            "3. 'HISTORY': Câu hỏi cần tìm hiểu thông tin lịch sử.\n\n"
            "CHỈ trả về 1 từ duy nhất: GREETING, IRRELEVANT, hoặc HISTORY."
        )
        try:
            intent_res = self.llm.generate(prompt=question, system_prompt=intent_system_prompt)
            intent = intent_res["content"].strip().upper()
            
            if "GREETING" in intent or "IRRELEVANT" in intent:
                print(f"[Agent] Fast-path triggered (Intent: {intent})")
                direct_prompt = (
                    "Bạn là 'Sử Việt AI' - một chuyên gia am hiểu Lịch sử Việt Nam. "
                    "Người dùng vừa nhắn một tin nhắn, có thể là chào hỏi hoặc thắc mắc không liên quan đến lịch sử. "
                    "Hãy chào lại thân thiện, hoặc nhắc nhở nhẹ nhàng rằng bạn chuyên môn hỗ trợ các câu hỏi về Lịch sử Việt Nam."
                )
                direct_res = self.llm.generate(prompt=question, system_prompt=direct_prompt)
                answer = direct_res["content"]
                
                logger.log_event("AGENT_END", {
                    "sub_tasks": 0,
                    "answer_length": len(answer),
                    "type": "direct_answer"
                })
                return {"answer": answer, "contexts": []}
        except Exception as e:
            logger.error(f"[HistoryAgent] Intent check failed: {e}")
            # Fallback sang luồng lập kế hoạch

        # Step 1: PLAN
        print("\n[Agent] Step 1: Planning...")
        plan = self._plan(question)

        if not plan:
            return {"answer": "Không thể tạo kế hoạch cho câu hỏi này.", "contexts": []}

        print(f"[Agent] Intent: {plan.get('intent', 'N/A')}")
        print(f"[Agent] Sub-tasks: {len(plan.get('sub_tasks', []))}")

        # Step 2: EXECUTE
        print("[Agent] Step 2: Executing tools...")
        sub_tasks = self._execute(plan)

        # Step 3: SYNTHESIZE
        print("[Agent] Step 3: Synthesizing answer...")
        answer = self._synthesize(question, sub_tasks)

        # Trích xuất contexts từ kết quả của các tool
        contexts = [task.result for task in sub_tasks if task.result and "Lỗi" not in task.result]

        logger.log_event("AGENT_END", {
            "sub_tasks": len(sub_tasks),
            "answer_length": len(answer),
        })

        return {
            "answer": answer,
            "contexts": contexts
        }

    def _plan(self, question: str) -> Optional[Dict]:
        """
        Use LLM to analyze the question and create an execution plan.
        Returns a dict with 'intent' and 'sub_tasks'.
        """
        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            tool_descriptions=self.tools.get_tool_descriptions()
        )

        user_prompt = f"Câu hỏi cần phân tích:\n{question}"

        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            raw_output = response["content"].strip()

            # Extract JSON from response (handle markdown code blocks)
            json_str = raw_output
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)

            plan = json.loads(json_str)

            # Validate plan structure
            if "sub_tasks" not in plan:
                plan["sub_tasks"] = [{"step": 1, "tool": "search_docs", "query": question, "purpose": "Tìm kiếm chung"}]

            # Limit sub-tasks
            plan["sub_tasks"] = plan["sub_tasks"][:self.max_sub_tasks]

            # Validate tool names
            valid_tools = self.tools.get_tool_names()
            for task in plan["sub_tasks"]:
                if task.get("tool") not in valid_tools:
                    task["tool"] = "search_docs"  # Fallback

            logger.log_event("PLAN_CREATED", {
                "intent": plan.get("intent", ""),
                "num_tasks": len(plan["sub_tasks"]),
            })

            return plan

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[HistoryAgent] Planning failed: {e}")
            # Fallback: simple single-tool plan
            return {
                "intent": question[:100],
                "sub_tasks": [
                    {"step": 1, "tool": "search_docs", "query": question, "purpose": "Tìm kiếm chung"}
                ]
            }

    def _execute(self, plan: Dict) -> List[SubTask]:
        """Execute each sub-task in the plan."""
        sub_tasks = []

        for task_data in plan.get("sub_tasks", []):
            step = task_data.get("step", len(sub_tasks) + 1)
            tool_name = task_data.get("tool", "search_docs")
            query = task_data.get("query", "")
            purpose = task_data.get("purpose", "")

            sub_task = SubTask(
                step=step,
                tool=tool_name,
                query=query,
                purpose=purpose,
            )

            print(f"  [{step}] Tool: {tool_name} | Query: {query[:60]}...")

            try:
                result = self.tools.execute(tool_name, query)
                sub_task.result = result
                logger.log_event("TOOL_EXECUTED", {
                    "step": step,
                    "tool": tool_name,
                    "result_length": len(result),
                })
            except Exception as e:
                sub_task.result = f"Lỗi khi thực thi tool: {str(e)}"
                logger.error(f"[HistoryAgent] Tool execution failed: {e}")

            sub_tasks.append(sub_task)

        return sub_tasks

    def _synthesize(self, question: str, sub_tasks: List[SubTask]) -> str:
        """Use LLM to synthesize all tool results into a final answer."""

        # Build collected info string
        collected_parts = []
        for task in sub_tasks:
            collected_parts.append(
                f"--- Bước {task.step}: {task.purpose} ---\n"
                f"Tool sử dụng: {task.tool}\n"
                f"Truy vấn: {task.query}\n"
                f"Kết quả:\n{task.result}"
            )

        collected_info = "\n\n".join(collected_parts)

        system_prompt = SYNTHESIZER_SYSTEM_PROMPT.format(
            collected_info=collected_info
        )

        user_prompt = f"Câu hỏi gốc: {question}\n\nHãy tổng hợp thông tin trên thành câu trả lời hoàn chỉnh."

        try:
            response = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            return response["content"]
        except Exception as e:
            logger.error(f"[HistoryAgent] Synthesis failed: {e}")
            # Fallback: return raw results
            return f"Lỗi tổng hợp. Kết quả thô:\n\n{collected_info}"

    def _format_trace(self, sub_tasks: List[SubTask]) -> str:
        """Format the execution trace for transparency."""
        lines = ["--- Execution Trace ---"]
        for task in sub_tasks:
            status = "OK" if task.result and "Lỗi" not in task.result[:10] else "FAIL"
            lines.append(
                f"  Step {task.step}: [{task.tool}] {task.query[:50]}... -> {status}"
            )
        return "\n".join(lines)
