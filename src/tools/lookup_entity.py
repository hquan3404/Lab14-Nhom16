"""
Tool: lookup_entity
Tra cứu nhanh một thực thể trong tài liệu lịch sử.

Dùng khi câu hỏi thuộc dạng:
- X là gì?
- X đóng vai trò gì?
- X liên quan thế nào đến chiến dịch?
- Chiến lược Y nghĩa là gì?
"""

from typing import List
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.telemetry.logger import logger


class LookupEntityTool:
    """
    Entity lookup tool.
    Searches for specific entities (people, places, organizations,
    strategies, events) and returns focused information.
    """

    description = (
        "Tra cứu nhanh một thực thể cụ thể: nhân vật, địa danh, tổ chức, "
        "chiến lược, sự kiện. Dùng cho câu hỏi dạng: X là gì, X đóng vai trò gì, "
        "X liên quan thế nào, chiến lược Y nghĩa là gì."
    )

    def __init__(self, retriever: HybridRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    def _extract_entity_sentences(self, text: str, entity: str) -> List[str]:
        """
        Extract sentences that directly mention the entity.
        Returns a focused subset of the text.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        entity_lower = entity.lower()

        relevant = []
        for sentence in sentences:
            if entity_lower in sentence.lower():
                sentence = sentence.strip()
                if len(sentence) > 20:
                    relevant.append(sentence)

        return relevant

    def run(self, query: str) -> str:
        """
        Look up information about a specific entity.

        Args:
            query: The entity name or question about an entity.

        Returns:
            Focused information about the entity.
        """
        logger.log_event("TOOL_LOOKUP_ENTITY", {"query": query[:100]})

        # Retrieve chunks mentioning the entity
        results = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            mode="hybrid",
        )

        if not results:
            return f"Không tìm thấy thông tin về thực thể: '{query}'"

        # Extract entity-focused sentences
        entity_info = []
        seen = set()

        for r in results:
            sentences = self._extract_entity_sentences(r.chunk.content, query)

            if sentences:
                for s in sentences:
                    key = s[:80]
                    if key not in seen:
                        seen.add(key)
                        entity_info.append({
                            "sentence": s,
                            "section": r.chunk.section or r.chunk.chapter,
                        })
            else:
                # If no exact entity match, include the most relevant chunk
                preview = r.chunk.content[:400]
                key = preview[:80]
                if key not in seen:
                    seen.add(key)
                    entity_info.append({
                        "sentence": preview,
                        "section": r.chunk.section or r.chunk.chapter,
                    })

        if not entity_info:
            return f"Không tìm thấy thông tin chi tiết về: '{query}'"

        # Format output
        output_parts = [f"THÔNG TIN VỀ: {query.upper()}"]
        output_parts.append("=" * 50)

        current_section = None
        for info in entity_info[:10]:  # Limit to 10 entries
            if info["section"] != current_section:
                current_section = info["section"]
                output_parts.append(f"\n[Phần: {current_section}]")
            output_parts.append(f"  • {info['sentence']}")

        return "\n".join(output_parts)
