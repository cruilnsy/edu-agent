# backend/core/knowledge_base.py
# BGE-M3 embedder, the Milvus chunk record, and the Milvus client (write + hybrid search).

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from backend.config import get_settings, resolve_path
from backend.core.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = get_settings().milvus_collection


# ── BGE-M3 embedder (process-wide singleton; dense + sparse output) ──

class BGEMEmbedder:
    """BGE-M3 singleton. One pass returns both:
      - a dense vector (1024 floats) for semantic search
      - a sparse vector ({token_id: weight}) for keyword search
    The first get_instance() loads the model (5-15 s); later calls reuse it.

        dense, sparse = BGEMEmbedder.get_instance().encode_query("What is a React hook?")
    """

    _instance: Optional["BGEMEmbedder"] = None

    def __init__(self, model_path: str):
        # ── Compatibility patch 1: FlagEmbedding 1.3.x uses a transformers internal ──
        # transformers>=5.0 removed is_torch_fx_available. The pinned 4.51.0 still has it;
        # this guards against an ImportError after an upgrade.
        import importlib.util as _ilu
        from transformers.utils import import_utils as _tf_iu
        if not hasattr(_tf_iu, "is_torch_fx_available"):
            _tf_iu.is_torch_fx_available = (
                lambda: _ilu.find_spec("torch.fx") is not None
            )

        # ── Compatibility patch 2: XLMRobertaModel rejecting a dtype argument ──
        # Some transformers versions do not accept dtype in XLMRobertaModel.__init__(), but
        # FlagEmbedding passes it (TypeError). A subclass drops it before calling the parent.
        from transformers.models.xlm_roberta import modeling_xlm_roberta as _xlm
        _OriginalXLMRoberta = _xlm.XLMRobertaModel

        class _PatchedXLMRobertaModel(_OriginalXLMRoberta):
            def __init__(self, config, **kwargs):
                kwargs.pop("dtype", None)
                super().__init__(config, **kwargs)

        _xlm.XLMRobertaModel = _PatchedXLMRobertaModel

        import torch
        from FlagEmbedding import BGEM3FlagModel

        logger.info("bge_m3.loading", model_path=model_path)

        # ── fp16 on CUDA only, not on MPS (Apple silicon) ──
        # MPS hits an LLVM ERROR in BGE-M3's attention matmul; fp32 on CPU is slower but stable.
        _use_fp16 = torch.cuda.is_available()

        self._model = BGEM3FlagModel(
            model_name_or_path=model_path,
            use_fp16=_use_fp16,
        )
        logger.info("bge_m3.loaded", use_fp16=_use_fp16)

    @classmethod
    def get_instance(cls) -> "BGEMEmbedder":
        """Singleton; loads the model on first use."""
        if cls._instance is None:
            bge3_path = resolve_path(get_settings().bge_m3_model_path)
            cls._instance = BGEMEmbedder(bge3_path)
        return cls._instance

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
    ) -> tuple[list[list[float]], list[dict]]:
        """Encode texts; returns (dense_vecs, sparse_vecs), one of each per text.
        batch_size: larger is faster but uses more memory; 12 suits 16 GB of (unified) memory."""
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,            # BGE-M3's limit; covers any chunk
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # ColBERT multi-vectors are not used
        )

        dense_vecs = output["dense_vecs"].tolist()   # numpy → Python list

        # numpy.float16 -> float is required: the LangGraph checkpointer serializes State with
        # msgpack, which raises TypeError on numpy.float16.
        sparse_vecs = [
            {int(k): float(v) for k, v in d.items()}
            for d in output["lexical_weights"]
        ]

        return dense_vecs, sparse_vecs

    def encode_query(self, text: str) -> tuple[list[float], dict]:
        """Encode one query (batch_size=1, no padding); returns (dense_vec, sparse_vec)."""
        dense_list, sparse_list = self.encode([text], batch_size=1)
        return dense_list[0], sparse_list[0]


# ── DocumentChunk: one row written to Milvus ──

@dataclass
class DocumentChunk:
    """One chunk to write to Milvus; fields match the collection schema.

    id:               MD5 of document_id + chunk_index + content prefix
    content:          chunk text (with the contextual sentence prepended when enabled)
    embedding:        dense BGE-M3 vector (1024)
    sparse_embedding: sparse BGE-M3 lexical weights {token_id: weight}
    source_name:      shown with answers, e.g. "ts-generics > TypeScript Handbook: Generics > Generic Types"
    """
    id:               str
    content:          str
    embedding:        list[float]
    sparse_embedding: dict
    course_id:        str
    document_id:      str
    source_name:      str
    chunk_type:       str                  # "text" / "code" / "table"
    chunk_index:      int
    version:          str
    tenant_id:        str = "tenant_default"
    updated_at:       int = field(default_factory=lambda: int(time.time()))


# ── Helpers ──

def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
    """Chunk id: MD5 of document_id + chunk_index + content prefix. Chunks at different positions
    never collide, and an unchanged chunk keeps its id, so rebuilding does not duplicate rows."""
    raw = f"{document_id}_{chunk_index}_{content[:50]}"
    return hashlib.md5(raw.encode()).hexdigest()


class KnowledgeBaseClient:
    """Milvus knowledge base client. One collection; tenants are separated by filtering on
    tenant_id. _client is a class attribute, so the process opens one connection."""

    _client: Optional["MilvusClient"] = None
    _loaded: bool = False


    def __init__(self):
        if KnowledgeBaseClient._client is None:
            settings = get_settings()
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
            KnowledgeBaseClient._client = MilvusClient(uri=uri)
            logger.info("milvus.connected", uri=uri)

        if not KnowledgeBaseClient._loaded:
            try:
                KnowledgeBaseClient._client.load_collection(COLLECTION_NAME)
            except Exception:
                pass   # Collection not created yet (init_milvus.py not run)
            KnowledgeBaseClient._loaded = True

    # ── Write: batch upsert ──

    def upsert_chunks(self, chunks: list) -> int:
        """Upsert chunks (update when the primary key exists, insert otherwise), one dict per row."""
        if not chunks:
            return 0

        data = [
            {
                "id":               c.id,
                "embedding":        c.embedding,
                "sparse_embedding": c.sparse_embedding,
                "content":          c.content[:4096],
                "chunk_index":      c.chunk_index,
                "document_id":      c.document_id,
                "course_id":        c.course_id,
                "tenant_id":        c.tenant_id,
                "source_name":      c.source_name,
                "chunk_type":       c.chunk_type,
                "version":          c.version,
                "updated_at":       c.updated_at,
            }
            for c in chunks
        ]

        self._client.upsert(collection_name=COLLECTION_NAME, data=data)
        logger.info("knowledge_base.chunks_upserted", count=len(chunks))
        return len(chunks)

    # ── Write: delete a document's chunks ──

    def delete_document_chunks(self, document_id: str) -> None:
        """Delete every chunk of a document (before re-inserting it). document_id is escaped to
        prevent filter expression injection."""
        safe_id = document_id.replace('"', '\\"')
        self._client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'document_id == "{safe_id}"',
        )
        logger.info("knowledge_base.document_deleted", document_id=document_id)

    # ── Helpers ──

    @staticmethod
    def generate_chunk_id(content: str, document_id: str, chunk_index: int) -> str:
        """Same as the module-level generate_chunk_id."""
        raw = f"{document_id}_{chunk_index}_{content[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()



    # ── Search settings ──
    VECTOR_TOP_K = 10    # Hybrid recall candidates passed to the reranker
    ANN_EF       = 64    # HNSW search breadth (accuracy vs speed)

    def _hybrid_search(
        self,
        query_embedding: list[float],
        query_sparse: dict,
        top_k: int,
        filters: Optional[str] = None,
    ) -> list[dict]:
        """Hybrid search: dense and sparse requests run in parallel in Milvus and are fused by
        WeightedRanker(0.7, 0.3). top_k applies to each request and to the fused list; filters is a
        Milvus boolean expression such as 'tenant_id == "x"'.

        Returns candidates with "content", "score" and "metadata". The score is a ranking signal,
        not a probability; the reranker scores the candidates properly.
        """
        try:
            # Dense: COSINE matches BGE-M3's normalized vectors
            dense_req = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={
                    "metric_type": "COSINE",
                    "params": {"ef": self.ANN_EF},
                },
                limit=top_k,
                expr=filters,
            )

            # Sparse: inner product is the standard metric for BGE-M3 lexical weights
            sparse_req = AnnSearchRequest(
                data=[query_sparse],
                anns_field="sparse_embedding",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=filters,
            )

            output_fields = [
                "content", "source_name", "chunk_type",
                "course_id", "document_id", "chunk_index",
            ]

            # Weights follow the request order: 0.7 dense, 0.3 sparse
            results = self._client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[dense_req, sparse_req],
                ranker=WeightedRanker(0.7, 0.3),
                limit=top_k,
                output_fields=output_fields,
            )

            # The fused score is in "distance"; it goes to the reranker unchanged
            candidates = []
            for hit in results[0]:
                candidates.append({
                    "content": hit["entity"].get("content") or "",
                    "score":   hit.get("distance") or 0.0,
                    "metadata": {
                        "source_name": hit["entity"].get("source_name") or "",
                        "chunk_type":  hit["entity"].get("chunk_type")  or "text",
                        "course_id":   hit["entity"].get("course_id")   or "",
                        "document_id": hit["entity"].get("document_id") or "",
                        "chunk_index": hit["entity"].get("chunk_index") or 0,
                    },
                })

            logger.info(
                "knowledge_base.hybrid_search_done",
                candidates=len(candidates),
            )
            return candidates

        except Exception as e:
            logger.error("knowledge_base.hybrid_search_failed", error=str(e))
            return []

    @staticmethod
    def _build_filter(tenant_id: str, course_id: Optional[str] = None) -> str:
        """Milvus filter for the tenant (and course). Values are escaped to prevent expression injection."""
        safe_tenant = tenant_id.replace('"', '\\"')
        expr = f'tenant_id == "{safe_tenant}"'
        if course_id:
            safe_course = course_id.replace('"', '\\"')
            expr += f' and course_id == "{safe_course}"'
        return expr
