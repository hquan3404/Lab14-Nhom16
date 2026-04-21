"""
Tool: search_docs
Truy vấn văn bản tổng quát trong tài liệu lịch sử.

Dùng khi câu hỏi thuộc dạng:
- Vì sao...?
- Như thế nào...?
- Ý nghĩa gì...?
- Hạn chế gì...?
- Mục tiêu là gì...?
- Vai trò của X...?
"""

from typing import List
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.telemetry.logger import logger


class SearchDocsTool:
    """
    General document search tool.
    Returns the most relevant passages from the knowledge base.
    """

    description = (
        "Tìm kiếm các đoạn nội dung liên quan nhất trong tài liệu lịch sử. "
        "Dùng cho câu hỏi dạng: vì sao, như thế nào, ý nghĩa, hạn chế, "
        "mục tiêu, vai trò của một địa điểm/khía cạnh."
    )

    def __init__(self, retriever: HybridRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    def run(self, query: str) -> str:
        """
        Search for relevant document passages.

        Args:
            query: The search query.

        Returns:
            Formatted context string with relevant passages.
        """
        logger.log_event("TOOL_SEARCH_DOCS", {"query": query[:100]})

        results = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            mode="hybrid",
        )

        if not results:
            return f"Không tìm thấy thông tin liên quan đến: '{query}'"

        # Format results with metadata
        output_parts = []
        for i, r in enumerate(results, 1):
            location_parts = []
            if r.chunk.chapter:
                location_parts.append(r.chunk.chapter)
            if r.chunk.section:
                location_parts.append(r.chunk.section)
            if r.chunk.subsection:
                location_parts.append(r.chunk.subsection)
            location = " > ".join(location_parts) if location_parts else "N/A"

            output_parts.append(
                f"[Nguồn {i}] (Vị trí: {location})\n{r.chunk.content}"
            )

        return "\n\n---\n\n".join(output_parts)
