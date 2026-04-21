"""
RAG Baseline Agent.
This is the simple chatbot that directly uses RAG to answer questions
without any reasoning or tool-use capabilities.
"""

from typing import Optional
from src.core.openai_provider import OpenAIProvider
from src.rag.chunker import MarkdownChunker
from src.rag.retriever import HybridRetriever
from src.rag.generator import RAGGenerator
from src.telemetry.logger import logger


class RAGBaselineAgent:
    """
    RAG Baseline Chatbot Agent.
    
    This agent retrieves relevant context from the history knowledge base
    and generates a grounded answer. No reasoning loop, no tool use.
    
    Usage:
        llm = OpenAIProvider(model_name="gpt-4o-mini")
        agent = RAGBaselineAgent(llm)
        answer = agent.run("Ai là lãnh đạo của Trận Bạch Đằng năm 1288?")
    """

    def __init__(
        self,
        llm: OpenAIProvider,
        data_path: str = "data/data.md",
        chunks_cache_path: str = "data/chunks.json",
        use_semantic: bool = True,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
    ):
        self.llm = llm
        self.data_path = data_path
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode

        logger.log_event("RAG_BASELINE_INIT", {
            "data_path": data_path,
            "use_semantic": use_semantic,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
        })

        # Step 1: Chunk the document
        chunker = MarkdownChunker(
            max_chunk_size=1500,
            chunk_overlap=200,
            min_chunk_size=100,
        )

        # Try to load cached chunks first
        import os
        if os.path.exists(chunks_cache_path):
            logger.info("[RAGBaselineAgent] Loading cached chunks...")
            chunks = chunker.load_chunks(chunks_cache_path)
        else:
            logger.info("[RAGBaselineAgent] Chunking document...")
            chunks = chunker.chunk_file(data_path)
            chunker.save_chunks(chunks, chunks_cache_path)

        logger.info(f"[RAGBaselineAgent] Total chunks: {len(chunks)}")

        # Step 2: Build retriever
        self.retriever = HybridRetriever(
            chunks=chunks,
            use_semantic=use_semantic,
            embedding_cache_dir="data/embeddings",
        )

        # Step 3: Build generator
        self.generator = RAGGenerator(
            retriever=self.retriever,
            llm=self.llm,
            top_k=self.top_k,
            retrieval_mode=self.retrieval_mode,
        )

        logger.info("[RAGBaselineAgent] Initialization complete!")

    def run(self, question: str) -> str:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User's question about Vietnamese history.
            
        Returns:
            Generated answer grounded in the knowledge base.
        """
        logger.log_event("RAG_BASELINE_RUN", {"question": question[:100]})
        
        # Thêm bước kiểm tra ý định (Intent Check) để tránh truy vấn không cần thiết
        intent_system_prompt = (
            "Bạn là một hệ thống phân loại câu hỏi.\n"
            "Vui lòng phân loại câu hỏi của người dùng vào 1 trong 3 nhóm sau:\n"
            "1. 'GREETING': Câu chào hỏi, cảm ơn, tạm biệt, hỏi thăm thông thường.\n"
            "2. 'IRRELEVANT': Câu hỏi hoàn toàn không liên quan đến chủ đề Lịch sử, Văn hóa, Địa lý Việt Nam (ví dụ: toán học, lập trình, nấu ăn).\n"
            "3. 'HISTORY': Câu hỏi hoặc từ khóa có liên quan đến Lịch sử Việt Nam, hoặc cần tra cứu thông tin.\n\n"
            "CHỈ trả về 1 từ duy nhất: GREETING, IRRELEVANT, hoặc HISTORY."
        )
        try:
            intent_res = self.llm.generate(prompt=question, system_prompt=intent_system_prompt)
            intent = intent_res["content"].strip().upper()
            
            if "GREETING" in intent or "IRRELEVANT" in intent:
                logger.info(f"[RAGBaselineAgent] Detected direct query intent: {intent}")
                direct_prompt = (
                    "Bạn là một trợ lý ảo thân thiện, chuyên môn về Lịch sử Việt Nam. "
                    "Hãy trả lời tin nhắn của người dùng một cách lịch sự, ngắn gọn và tự nhiên. "
                    "Nếu người dùng hỏi vấn đề ngoài lề (toán học, lập trình...), hãy khéo léo từ chối và nhắc họ rằng bạn chỉ chuyên về Lịch sử Việt Nam."
                )
                direct_res = self.llm.generate(prompt=question, system_prompt=direct_prompt)
                answer = direct_res["content"]
                
                logger.log_event("RAG_BASELINE_COMPLETE", {
                    "answer_length": len(answer),
                    "sources": 0,
                    "type": "direct_answer"
                })
                return answer
                
        except Exception as e:
            logger.error(f"[RAGBaselineAgent] Intent classification failed: {e}")
            # Fallback nếu lỗi phân loại thì vẫn tiếp tục RAG

        # RAG Pipeline thực thụ
        result = self.generator.generate(question)
        
        # Format output
        answer = result["answer"]

        logger.log_event("RAG_BASELINE_COMPLETE", {
            "answer_length": len(answer),
            "sources": len(result["sources"]),
            "latency_ms": result["metadata"].get("llm_latency_ms", 0),
        })

        return answer

