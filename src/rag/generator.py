"""
RAG Generator for Baseline Agent.
Combines retrieved context with LLM generation for grounded responses.
"""

from typing import List, Dict, Any, Optional
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.core.openai_provider import OpenAIProvider
from src.telemetry.logger import logger


# System prompt for the RAG baseline
RAG_SYSTEM_PROMPT = """Bạn là một chuyên gia lịch sử Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các nguồn tài liệu được cung cấp.

## Quy tắc:
1. CHỈ trả lời dựa trên thông tin có trong các nguồn tài liệu bên dưới.
2. Nếu thông tin không có trong tài liệu, hãy nói rõ "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
3. Trả lời bằng tiếng Việt, rõ ràng và có cấu trúc.
4. Nếu có nhiều nguồn liên quan, hãy tổng hợp thông tin một cách mạch lạc.
5. Đưa ra các số liệu, ngày tháng cụ thể nếu có trong tài liệu.

## Nguồn tài liệu:
{context}
"""

RAG_USER_PROMPT = """Câu hỏi: {question}

Hãy trả lời câu hỏi trên dựa trên các nguồn tài liệu đã cung cấp."""


class RAGGenerator:
    """
    RAG Generator: Retrieval-Augmented Generation pipeline.
    
    Pipeline:
    1. Query → Retriever → Top-K relevant chunks
    2. Chunks → Context formatting
    3. Context + Query → LLM → Grounded answer
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: OpenAIProvider,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode

    def generate(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline: Retrieve → Format → Generate.
        
        Args:
            question: User's question.
            top_k: Override default top_k.
            mode: Override default retrieval mode.
            
        Returns:
            Dict with answer, sources, and metadata.
        """
        k = top_k or self.top_k
        retrieval_mode = mode or self.retrieval_mode

        logger.log_event("RAG_GENERATE_START", {
            "question": question[:100],
            "top_k": k,
            "mode": retrieval_mode,
        })

        # Step 1: Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=question,
            top_k=k,
            mode=retrieval_mode,
        )

        if not results:
            return {
                "answer": "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.",
                "sources": [],
                "metadata": {"retrieval_count": 0},
            }

        # Step 2: Format context
        context = self.retriever.format_context(results)

        # Step 3: Build prompts
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        user_prompt = RAG_USER_PROMPT.format(question=question)

        # Step 4: Generate answer
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        answer = response["content"]

        # Build source metadata
        sources = []
        for r in results:
            sources.append({
                "chunk_id": r.chunk.chunk_id,
                "chapter": r.chunk.chapter,
                "section": r.chunk.section,
                "subsection": r.chunk.subsection,
                "score": r.score,
                "source_type": r.source,
                "preview": r.chunk.content[:200] + "...",
            })

        result = {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "retrieval_count": len(results),
                "retrieval_mode": retrieval_mode,
                "llm_usage": response.get("usage", {}),
                "llm_latency_ms": response.get("latency_ms", 0),
                "llm_provider": response.get("provider", ""),
            },
        }

        logger.log_event("RAG_GENERATE_COMPLETE", {
            "answer_length": len(answer),
            "sources_count": len(sources),
            "latency_ms": response.get("latency_ms", 0),
        })

        return result

    def generate_answer_only(self, question: str) -> str:
        """Convenience method that returns just the answer string."""
        result = self.generate(question)
        return result["answer"]
