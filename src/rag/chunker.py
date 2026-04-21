"""
Chunker module for RAG Baseline.
Splits Vietnamese history markdown documents into semantically meaningful chunks
using a header-aware strategy with overlap for context preservation.
"""

import re
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from src.telemetry.logger import logger


@dataclass
class Chunk:
    """Represents a single document chunk with metadata."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    # Hierarchical section info
    chapter: str
    section: str
    subsection: str
    # Position info
    start_line: int
    end_line: int
    char_count: int
    word_count: int


class MarkdownChunker:
    """
    Header-aware Markdown chunker for Vietnamese history documents.
    
    Strategy:
    - Split by markdown headers (##, ###, ####) to preserve topical boundaries
    - Apply a max chunk size with overlap for long sections
    - Attach hierarchical metadata (chapter > section > subsection)
    """

    def __init__(
        self,
        max_chunk_size: int = 1500,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _generate_chunk_id(self, content: str, index: int) -> str:
        """Generate a deterministic chunk ID based on content hash."""
        hash_input = f"{content[:100]}_{index}"
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()[:12]

    def _count_words(self, text: str) -> int:
        """Count words in text (works for Vietnamese)."""
        return len(text.split())

    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Split document into sections based on markdown headers.
        Returns list of {level, title, content, start_line, end_line}.
        """
        lines = text.split("\n")
        sections = []
        current_section = {
            "level": 0,
            "title": "Mở đầu",
            "content_lines": [],
            "start_line": 1,
        }

        header_pattern = re.compile(r"^(#{1,4})\s+(.+)$")

        for i, line in enumerate(lines, start=1):
            match = header_pattern.match(line.strip())
            if match:
                # Save previous section
                if current_section["content_lines"]:
                    content = "\n".join(current_section["content_lines"]).strip()
                    if content:
                        sections.append({
                            "level": current_section["level"],
                            "title": current_section["title"],
                            "content": content,
                            "start_line": current_section["start_line"],
                            "end_line": i - 1,
                        })

                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = {
                    "level": level,
                    "title": title,
                    "content_lines": [],
                    "start_line": i,
                }
            else:
                current_section["content_lines"].append(line)

        # Don't forget the last section
        if current_section["content_lines"]:
            content = "\n".join(current_section["content_lines"]).strip()
            if content:
                sections.append({
                    "level": current_section["level"],
                    "title": current_section["title"],
                    "content": content,
                    "start_line": current_section["start_line"],
                    "end_line": len(lines),
                })

        return sections

    def _build_hierarchy(self, sections: List[Dict]) -> List[Dict]:
        """
        Assign hierarchical context (chapter, section, subsection) 
        to each section based on header levels.
        """
        current_chapter = ""
        current_section = ""
        current_subsection = ""

        for sec in sections:
            level = sec["level"]
            title = sec["title"]

            if level == 1:
                current_chapter = title
                current_section = ""
                current_subsection = ""
            elif level == 2:
                current_section = title
                current_subsection = ""
            elif level == 3:
                current_subsection = title
            elif level == 4:
                # Keep subsection as the #### title
                current_subsection = title

            sec["chapter"] = current_chapter
            sec["section"] = current_section
            sec["subsection"] = current_subsection

        return sections

    def _split_long_section(self, text: str, start_line: int) -> List[Dict[str, Any]]:
        """
        Split a long section into smaller chunks with overlap.
        Splits on paragraph boundaries (double newlines) when possible.
        """
        if len(text) <= self.max_chunk_size:
            return [{"content": text, "start_line": start_line, "end_line": start_line}]

        chunks = []
        paragraphs = re.split(r"\n\s*\n", text)
        
        current_chunk = ""
        current_start = start_line
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds max size, save current chunk
            if current_chunk and len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                chunks.append({
                    "content": current_chunk.strip(),
                    "start_line": current_start,
                    "end_line": current_start,
                })
                # Overlap: keep the last part of the current chunk
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
                current_start = current_start + 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "start_line": current_start,
                "end_line": current_start,
            })

        return chunks

    def chunk_file(self, file_path: str) -> List[Chunk]:
        """
        Main entry point: chunk a markdown file into semantically meaningful pieces.
        
        Args:
            file_path: Path to the markdown file.
            
        Returns:
            List of Chunk objects with content and metadata.
        """
        logger.log_event("CHUNKING_START", {"file": file_path})

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Step 1: Split by headers
        raw_sections = self._split_by_headers(text)
        logger.info(f"[Chunker] Found {len(raw_sections)} raw sections from headers")

        # Step 2: Build hierarchy
        sections_with_hierarchy = self._build_hierarchy(raw_sections)

        # Step 3: Split long sections and create Chunk objects
        chunks = []
        chunk_index = 0

        for sec in sections_with_hierarchy:
            sub_chunks = self._split_long_section(sec["content"], sec["start_line"])

            for i, sub in enumerate(sub_chunks):
                content = sub["content"]
                
                # Skip chunks that are too small
                if len(content) < self.min_chunk_size:
                    continue

                # Build context prefix for better retrieval
                context_prefix = ""
                if sec["chapter"]:
                    context_prefix += f"[Chương: {sec['chapter']}] "
                if sec["section"]:
                    context_prefix += f"[Phần: {sec['section']}] "
                if sec["subsection"]:
                    context_prefix += f"[Mục: {sec['subsection']}] "

                chunk = Chunk(
                    chunk_id=self._generate_chunk_id(content, chunk_index),
                    content=content,
                    metadata={
                        "source": os.path.basename(file_path),
                        "section_title": sec["title"],
                        "context_prefix": context_prefix.strip(),
                        "sub_chunk_index": i,
                        "total_sub_chunks": len(sub_chunks),
                    },
                    chapter=sec.get("chapter", ""),
                    section=sec.get("section", ""),
                    subsection=sec.get("subsection", ""),
                    start_line=sub["start_line"],
                    end_line=sub["end_line"],
                    char_count=len(content),
                    word_count=self._count_words(content),
                )
                chunks.append(chunk)
                chunk_index += 1

        logger.log_event("CHUNKING_COMPLETE", {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(c.char_count for c in chunks) // max(len(chunks), 1),
            "total_words": sum(c.word_count for c in chunks),
        })

        return chunks

    def save_chunks(self, chunks: List[Chunk], output_path: str):
        """Save chunks to a JSON file for inspection and caching."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [asdict(c) for c in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[Chunker] Saved {len(chunks)} chunks to {output_path}")

    def load_chunks(self, input_path: str) -> List[Chunk]:
        """Load chunks from a previously saved JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = []
        for d in data:
            chunks.append(Chunk(**d))
        logger.info(f"[Chunker] Loaded {len(chunks)} chunks from {input_path}")
        return chunks
