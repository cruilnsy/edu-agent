# scripts/init_milvus.py
# (Re)create the Milvus collection. Drops the existing one first.
# Usage: python scripts/init_milvus.py
import os
import sys
from pathlib import Path

# Run as a script, sys.path[0] is scripts/, so add the project root to import backend.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymilvus import MilvusClient, DataType
from backend.config import get_settings


MILVUS_URI = f"http://{get_settings().milvus_host}:{get_settings().milvus_port}"
VECTOR_DIM = 1024                  # BGE-M3 dense vector size
COLLECTION_NAME = get_settings().milvus_collection  # One collection; tenants are separated by tenant_id


def build_schema(client: MilvusClient):
    """Schema: dense + sparse vectors plus scalar fields (including tenant_id)."""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id",               DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("embedding",        DataType.FLOAT_VECTOR, dim=VECTOR_DIM)      # Dense
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)              # Sparse
    schema.add_field("content",          DataType.VARCHAR, max_length=4096)
    schema.add_field("tenant_id",        DataType.VARCHAR, max_length=64)           # Tenant isolation
    schema.add_field("chunk_index",      DataType.INT64)
    schema.add_field("document_id",      DataType.VARCHAR, max_length=64)
    schema.add_field("course_id",        DataType.VARCHAR, max_length=64)
    schema.add_field("source_name",      DataType.VARCHAR, max_length=256)
    schema.add_field("chunk_type",       DataType.VARCHAR, max_length=32)
    schema.add_field("version",          DataType.VARCHAR, max_length=32)
    schema.add_field("updated_at",       DataType.INT64)
    return schema


def build_index_params(client: MilvusClient):
    """Indexes: HNSW for dense, SPARSE_INVERTED_INDEX for sparse, INVERTED for filters."""
    ip = client.prepare_index_params()
    # Dense: HNSW + COSINE (semantic similarity)
    ip.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 256})
    # Sparse: SPARSE_INVERTED_INDEX + IP; drop_ratio_build drops the lowest-weight tokens to save space
    ip.add_index(field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX",
                 metric_type="IP", params={"drop_ratio_build": 0.2})
    # Scalar fields: INVERTED index for fast filtering
    ip.add_index(field_name="tenant_id", index_type="INVERTED")
    ip.add_index(field_name="course_id", index_type="INVERTED")
    return ip


def main():
    print(f"Connecting to Milvus: {MILVUS_URI}")
    client = MilvusClient(uri=MILVUS_URI)

    if client.has_collection(COLLECTION_NAME):
        print(f"Dropping the existing collection '{COLLECTION_NAME}'...")
        client.drop_collection(COLLECTION_NAME)

    # Passing index_params builds the indexes and loads the collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=build_schema(client),
        index_params=build_index_params(client),
    )
    print(f"Collection '{COLLECTION_NAME}' created, indexed and loaded")
    print("Collections:", client.list_collections())
    print("The collection is empty now; run python -m scripts.build_knowledge_base to load it.")


if __name__ == "__main__":
    main()
