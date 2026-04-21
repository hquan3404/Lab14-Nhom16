"""
Tool Registry - Manages shared retriever and all tools.
Initializes the retriever once and passes it to all tools.
"""

import os
from typing import Dict, Any, List, Callable
from src.rag.chunker import MarkdownChunker, Chunk
from src.rag.retriever import HybridRetriever
from src.telemetry.logger import logger


class ToolRegistry:
    """
    Central registry that holds a shared retriever and all agent tools.
    This avoids re-building the BM25 index for each tool.
    """

    def __init__(
        self,
        data_path: str = "data/data.md",
        chunks_cache_path: str = "data/chunks.json",
    ):
        logger.info("[ToolRegistry] Initializing shared retriever...")

        # Build chunks
        chunker = MarkdownChunker(
            max_chunk_size=1500,
            chunk_overlap=200,
            min_chunk_size=100,
        )

        if os.path.exists(chunks_cache_path):
            self.chunks = chunker.load_chunks(chunks_cache_path)
        else:
            self.chunks = chunker.chunk_file(data_path)
            chunker.save_chunks(self.chunks, chunks_cache_path)

        # Build shared retriever (BM25 only)
        self.retriever = HybridRetriever(
            chunks=self.chunks,
            use_semantic=False,
        )

        # Register tools
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

        logger.info(f"[ToolRegistry] Ready with {len(self._tools)} tools, {len(self.chunks)} chunks")

    def _register_default_tools(self):
        """Register the 3 default tools."""
        from src.tools.search_docs import SearchDocsTool
        from src.tools.build_timeline import BuildTimelineTool
        from src.tools.lookup_entity import LookupEntityTool

        search_docs = SearchDocsTool(self.retriever)
        build_timeline = BuildTimelineTool(self.retriever)
        lookup_entity = LookupEntityTool(self.retriever)

        self.register("search_docs", search_docs.run, search_docs.description)
        self.register("build_timeline", build_timeline.run, build_timeline.description)
        self.register("lookup_entity", lookup_entity.run, lookup_entity.description)

    def register(self, name: str, func: Callable, description: str):
        """Register a tool."""
        self._tools[name] = {
            "name": name,
            "func": func,
            "description": description,
        }

    def get_tool(self, name: str) -> Dict[str, Any]:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def execute(self, tool_name: str, query: str) -> str:
        """Execute a tool by name with a query."""
        tool = self.get_tool(tool_name)
        return tool["func"](query)

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools for the Planner prompt."""
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool['description']}")
        return "\n".join(lines)

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())
