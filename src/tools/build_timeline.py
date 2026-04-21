"""
Tool: build_timeline
Trích xuất các mốc thời gian và dựng diễn biến theo thứ tự thời gian.

Dùng khi câu hỏi thuộc dạng:
- Diễn biến chính là gì?
- Các mốc quan trọng?
- Chiến dịch diễn ra theo trình tự nào?
- Đợt 1, đợt 2 khác nhau ra sao?
- Trước/sau một mốc nào đó?
"""

import re
from typing import List, Tuple
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.telemetry.logger import logger


class BuildTimelineTool:
    """
    Timeline builder tool.
    Searches for relevant chunks, extracts temporal markers,
    and returns events sorted chronologically.
    """

    description = (
        "Trích xuất và sắp xếp các mốc thời gian, diễn biến theo thứ tự. "
        "Dùng cho câu hỏi dạng: diễn biến chính, các mốc quan trọng, "
        "trình tự chiến dịch, trước/sau một mốc thời gian."
    )

    # Regex patterns for Vietnamese date formats
    DATE_PATTERNS = [
        # "ngày 30-4-1975", "ngày 30/4/1975"
        r'ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
        # "tháng 3-1965", "tháng 3/1965", "tháng 3 năm 1965"
        r'tháng\s+(\d{1,2})[/-](\d{4})',
        r'tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
        # "năm 1965", "năm 1968"
        r'năm\s+(\d{4})',
        # Standalone dates: "1965-1968", "(1965)", "1965"
        r'\b(\d{4})\b',
    ]

    def __init__(self, retriever: HybridRetriever, top_k: int = 8):
        self.retriever = retriever
        self.top_k = top_k

    def _extract_year(self, text: str) -> int:
        """Extract the primary year from a text snippet."""
        # Try specific patterns first
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                # The year is always the last or only 4-digit group
                for g in reversed(groups):
                    if len(g) == 4 and g.isdigit():
                        year = int(g)
                        if 1945 <= year <= 1980:
                            return year
        return 9999  # Unknown date, sort to end

    def _extract_date_sort_key(self, text: str) -> Tuple[int, int, int]:
        """Extract (year, month, day) tuple for sorting."""
        year, month, day = 9999, 0, 0

        # Try "ngày DD-MM-YYYY"
        m = re.search(r'ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})', text)
        if m:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (year, month, day)

        # Try "tháng MM-YYYY"
        m = re.search(r'tháng\s+(\d{1,2})[/-](\d{4})', text)
        if m:
            month, year = int(m.group(1)), int(m.group(2))
            return (year, month, 0)

        # Try "tháng MM năm YYYY"
        m = re.search(r'tháng\s+(\d{1,2})\s+năm\s+(\d{4})', text)
        if m:
            month, year = int(m.group(1)), int(m.group(2))
            return (year, month, 0)

        # Try standalone year
        m = re.search(r'\b(19\d{2})\b', text)
        if m:
            year = int(m.group(1))
            return (year, 0, 0)

        return (9999, 0, 0)

    def _extract_timeline_entries(self, text: str) -> List[dict]:
        """
        Extract individual timeline entries from a chunk.
        Splits by sentences containing date markers.
        """
        entries = []
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue

            # Check if sentence contains a date
            has_date = False
            for pattern in self.DATE_PATTERNS[:4]:  # Skip generic \b\d{4}\b
                if re.search(pattern, sentence, re.IGNORECASE):
                    has_date = True
                    break

            if has_date:
                sort_key = self._extract_date_sort_key(sentence)
                entries.append({
                    "text": sentence,
                    "sort_key": sort_key,
                    "year": sort_key[0],
                })

        return entries

    def run(self, query: str) -> str:
        """
        Build a chronological timeline for the query.

        Args:
            query: The search query about historical events.

        Returns:
            Chronologically sorted timeline of events.
        """
        logger.log_event("TOOL_BUILD_TIMELINE", {"query": query[:100]})

        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            mode="hybrid",
        )

        if not results:
            return f"Không tìm thấy thông tin thời gian liên quan đến: '{query}'"

        # Extract timeline entries from all chunks
        all_entries = []
        seen_texts = set()

        for r in results:
            entries = self._extract_timeline_entries(r.chunk.content)
            for entry in entries:
                # Deduplicate by first 80 chars
                key = entry["text"][:80]
                if key not in seen_texts:
                    seen_texts.add(key)
                    entry["source_section"] = r.chunk.section or r.chunk.chapter
                    all_entries.append(entry)

        if not all_entries:
            # Fallback: return chunks sorted by year extracted from content
            fallback_parts = []
            for i, r in enumerate(results[:5], 1):
                year = self._extract_year(r.chunk.content)
                year_str = str(year) if year != 9999 else "N/A"
                fallback_parts.append(
                    f"[{year_str}] {r.chunk.content[:300]}..."
                )
            return "Không trích xuất được mốc thời gian cụ thể. Các đoạn liên quan:\n\n" + "\n\n".join(fallback_parts)

        # Sort chronologically
        all_entries.sort(key=lambda x: x["sort_key"])

        # Format output
        output_lines = [f"DIỄN BIẾN THỜI GIAN ({len(all_entries)} mốc):"]
        output_lines.append("=" * 50)

        current_year = None
        for entry in all_entries:
            year = entry["year"]
            if year != 9999 and year != current_year:
                current_year = year
                output_lines.append(f"\n--- Năm {current_year} ---")

            output_lines.append(f"  • {entry['text']}")

        return "\n".join(output_lines)
