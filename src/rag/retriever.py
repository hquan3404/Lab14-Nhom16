"""
Hybrid Retriever for RAG Baseline.
Combines BM25 (keyword-based) search with Semantic (embedding-based) search
using Reciprocal Rank Fusion (RRF) for result merging.
"""

import os
import re
import json
import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.rag.chunker import Chunk
from src.telemetry.logger import logger


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""
    chunk: Chunk
    score: float
    source: str  # "bm25", "semantic", or "hybrid"


class BM25:
    """
    BM25 scoring algorithm for Vietnamese text retrieval.
    Lightweight, no external dependencies needed.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.corpus_size: int = 0
        self.tokenized_docs: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace + punctuation tokenizer for Vietnamese.
        Vietnamese is mostly whitespace-delimited at syllable level.
        """
        text = text.lower()
        # Remove special characters but keep Vietnamese diacritics
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]  # Filter single chars

    def fit(self, documents: List[str]):
        """Build the BM25 index from documents."""
        self.corpus_size = len(documents)
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / max(self.corpus_size, 1)

        # Calculate document frequencies
        self.doc_freqs = {}
        for tokenized_doc in self.tokenized_docs:
            unique_tokens = set(tokenized_doc)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def _idf(self, term: str) -> float:
        """Calculate IDF for a term."""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_index: int) -> float:
        """Calculate BM25 score for a query against a specific document."""
        query_tokens = self._tokenize(query)
        doc_tokens = self.tokenized_docs[doc_index]
        doc_len = self.doc_lengths[doc_index]
        
        tf_counter = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            tf = tf_counter.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search and return top-k document indices with scores."""
        scores = []
        for i in range(self.corpus_size):
            s = self.score(query, i)
            if s > 0:
                scores.append((i, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SemanticSearch:
    """
    Semantic search using OpenAI embeddings.
    Uses cosine similarity for matching.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.embeddings: Optional[np.ndarray] = None
        self._client = None

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts via OpenAI API."""
        client = self._get_client()
        
        # Process in batches (OpenAI limit: 2048 per request)
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate very long texts to avoid token limits
            batch = [t[:8000] for t in batch]
            
            response = client.embeddings.create(
                model=self.model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def fit(self, documents: List[str], cache_path: Optional[str] = None):
        """
        Build the embedding index.
        Supports caching to avoid re-computing embeddings.
        """
        if cache_path and os.path.exists(cache_path):
            logger.info(f"[SemanticSearch] Loading cached embeddings from {cache_path}")
            self.embeddings = np.load(cache_path)
            return

        logger.info(f"[SemanticSearch] Computing embeddings for {len(documents)} documents...")
        self.embeddings = self._get_embeddings(documents)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, self.embeddings)
            logger.info(f"[SemanticSearch] Cached embeddings to {cache_path}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for most similar documents by cosine similarity."""
        if self.embeddings is None:
            raise ValueError("SemanticSearch not fitted. Call fit() first.")

        query_embedding = self._get_embeddings([query])[0]

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        norms = np.where(norms == 0, 1e-10, norms)  # Avoid division by zero
        similarities = np.dot(self.embeddings, query_embedding) / norms

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results


class HybridRetriever:
    """
    Hybrid Retriever combining BM25 and Semantic search.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(
        self,
        chunks: List[Chunk],
        use_semantic: bool = True,
        embedding_model: str = "text-embedding-3-small",
        embedding_cache_dir: str = "data/embeddings",
        rrf_k: int = 60,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ):
        self.chunks = chunks
        self.use_semantic = use_semantic
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

        # Prepare documents with context prefix for better retrieval
        self.documents = []
        for chunk in chunks:
            prefix = chunk.metadata.get("context_prefix", "")
            doc = f"{prefix}\n{chunk.content}" if prefix else chunk.content
            self.documents.append(doc)

        # Initialize BM25
        logger.info("[HybridRetriever] Building BM25 index...")
        self.bm25 = BM25()
        self.bm25.fit(self.documents)
        logger.info(f"[HybridRetriever] BM25 index built with {len(self.documents)} documents")

        # Initialize Semantic Search
        self.semantic: Optional[SemanticSearch] = None
        if use_semantic:
            logger.info("[HybridRetriever] Building Semantic index...")
            self.semantic = SemanticSearch(model=embedding_model)
            cache_path = os.path.join(embedding_cache_dir, "chunk_embeddings.npy")
            self.semantic.fit(self.documents, cache_path=cache_path)
            logger.info("[HybridRetriever] Semantic index built")

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        semantic_results: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """
        Merge results from BM25 and Semantic search using RRF.
        
        RRF Score = sum(1 / (k + rank_i)) for each ranking list
        """
        fused_scores: Dict[int, float] = defaultdict(float)

        # BM25 contributions
        for rank, (doc_idx, _score) in enumerate(bm25_results):
            fused_scores[doc_idx] += self.bm25_weight * (1.0 / (self.rrf_k + rank + 1))

        # Semantic contributions
        for rank, (doc_idx, _score) in enumerate(semantic_results):
            fused_scores[doc_idx] += self.semantic_weight * (1.0 / (self.rrf_k + rank + 1))

        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            mode: "hybrid", "bm25", or "semantic".
            
        Returns:
            List of RetrievalResult objects.
        """
        logger.log_event("RETRIEVAL_START", {"query": query[:100], "mode": mode, "top_k": top_k})

        if mode == "bm25":
            results = self._retrieve_bm25(query, top_k)
        elif mode == "semantic":
            results = self._retrieve_semantic(query, top_k)
        else:  # hybrid
            results = self._retrieve_hybrid(query, top_k)

        logger.log_event("RETRIEVAL_COMPLETE", {
            "num_results": len(results),
            "top_score": results[0].score if results else 0,
        })

        return results

    def _retrieve_bm25(self, query: str, top_k: int) -> List[RetrievalResult]:
        """BM25-only retrieval."""
        bm25_results = self.bm25.search(query, top_k=top_k)
        return [
            RetrievalResult(
                chunk=self.chunks[idx],
                score=score,
                source="bm25",
            )
            for idx, score in bm25_results
        ]

    def _retrieve_semantic(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Semantic-only retrieval."""
        if self.semantic is None:
            raise ValueError("Semantic search not initialized. Set use_semantic=True.")
        
        semantic_results = self.semantic.search(query, top_k=top_k)
        return [
            RetrievalResult(
                chunk=self.chunks[idx],
                score=score,
                source="semantic",
            )
            for idx, score in semantic_results
        ]

    def _retrieve_hybrid(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Hybrid retrieval using RRF."""
        # Get more results from each method, then fuse
        search_k = min(top_k * 3, len(self.chunks))
        
        bm25_results = self.bm25.search(query, top_k=search_k)

        if self.semantic:
            semantic_results = self.semantic.search(query, top_k=search_k)
            fused = self._reciprocal_rank_fusion(bm25_results, semantic_results)
        else:
            # Fallback to BM25 only if semantic is not available
            fused = [(idx, score) for idx, score in bm25_results]

        results = []
        for idx, score in fused[:top_k]:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=score,
                    source="hybrid",
                )
            )

        return results

    def format_context(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        Includes metadata for attribution.
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            header = f"--- Nguồn {i} (Score: {result.score:.4f}) ---"
            
            # Add hierarchical context
            location = []
            if chunk.chapter:
                location.append(f"Chương: {chunk.chapter}")
            if chunk.section:
                location.append(f"Phần: {chunk.section}")
            if chunk.subsection:
                location.append(f"Mục: {chunk.subsection}")
            
            location_str = " > ".join(location) if location else "N/A"
            
            context_parts.append(
                f"{header}\n"
                f"[Vị trí: {location_str}]\n"
                f"{chunk.content}"
            )

        return "\n\n".join(context_parts)
