# backend/core/reranker.py
# BGE-Reranker-v2-m3 cross-encoder and the retrieve() pipeline: hybrid recall -> rerank.

import os
from dataclasses import dataclass
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from backend.config import get_settings, resolve_path
from backend.core.logger import get_logger

logger = get_logger(__name__)
RERANK_MAX_INPUT_CHARS = 1200   # Cut long documents to stay within the CrossEncoder's max_length=512


@dataclass
class RankedDocument:
    """One document after reranking."""
    content:        str    # Document text
    score:          float  # BGE-Reranker relevance probability, 0-1
    original_index: int    # Position in the recalled candidate list (from 0)
    metadata:       dict   # Source metadata (source_name / chunk_type / course_id ...)


class BGEReranker:
    """BGE-Reranker-v2-m3 singleton. Scores hybrid-recall candidates with a cross-encoder and
    returns 0-1 probabilities, so no extra normalization is needed.

        reranker = BGEReranker.get_instance()
        docs, confidence = reranker.rerank_with_confidence("What is a React hook?", candidates, top_k=3)
    """

    _instance: Optional["BGEReranker"] = None

    def __init__(self):
        os.environ["ACCELERATE_USE_META_DEVICE"] = "0"
        settings = get_settings()
        model_path = resolve_path(settings.reranker_model_path)

        use_local = (
            os.path.exists(model_path)
            and os.path.isdir(model_path)
            and any(f.endswith((".bin", ".safetensors", ".json")) for f in os.listdir(model_path))
        )
        model_id = model_path if use_local else "BAAI/bge-reranker-v2-m3"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("reranker.loading", model_id=model_id, device=device)
        self._model = CrossEncoder(model_id, device=device, max_length=512)
        logger.info("reranker.loaded", model_id=model_id)

    @classmethod
    def get_instance(cls) -> "BGEReranker":
        """Singleton; loads the model on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank_with_confidence(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> tuple[list[RankedDocument], float]:
        """Rerank documents (each with a "content" field) against the query.

        Returns (ranked_docs, confidence): up to top_k documents by descending relevance, and the
        top document's probability. The QA agent compares confidence with its high-confidence
        threshold to choose between a knowledge base answer and a direct one.
        """
        if not documents:
            return [], 0.0

        # CrossEncoder input: (query, document) pairs, long documents cut
        pairs = [
            (query, (doc.get("content") or "")[:RERANK_MAX_INPUT_CHARS])
            for doc in documents
        ]

        # The default sigmoid activation makes predict() return 0-1 probabilities
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            [
                RankedDocument(
                    content=documents[i].get("content", ""),
                    score=scores[i],
                    original_index=i,
                    metadata=documents[i].get("metadata", {}),
                )
                for i in range(len(documents))
            ],
            key=lambda x: x.score,
            reverse=True,
        )

        top_results = ranked[:top_k]
        confidence = top_results[0].score if top_results else 0.0

        logger.info(
            "reranker.done",
            candidates=len(documents),
            top_k=top_k,
            confidence=round(confidence, 4),
        )

        return top_results, confidence


def retrieve(
    query: str,
    tenant_id: str,
    course_id: Optional[str] = None,
    recall_top_k: int = 10,
    rerank_top_k: int = 3,
) -> tuple[list[RankedDocument], float]:
    """Hybrid recall -> BGE rerank. The caller passes text; encoding happens here.

    recall_top_k candidates come from Milvus (filtered by tenant and, optionally, course), and the
    best rerank_top_k go back as (ranked_docs, confidence of the top document).
    """
    from backend.core.knowledge_base import BGEMEmbedder, KnowledgeBaseClient  # Lazy import avoids a cycle

    # ── 1. Encode the query ──
    embedder = BGEMEmbedder.get_instance()
    dense_vec, sparse_vec = embedder.encode_query(query)

    # ── 2. Hybrid recall ──
    kb = KnowledgeBaseClient()
    filters = kb._build_filter(tenant_id, course_id)
    candidates = kb._hybrid_search(
        query_embedding=dense_vec,
        query_sparse=sparse_vec,
        top_k=recall_top_k,
        filters=filters,
    )

    if not candidates:
        logger.info("retrieve.empty", query_preview=query[:50])
        return [], 0.0

    # ── 3. Rerank ──
    reranker = BGEReranker.get_instance()
    return reranker.rerank_with_confidence(query, candidates, top_k=rerank_top_k)
