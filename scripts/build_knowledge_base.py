# scripts/build_knowledge_base.py
# Build the knowledge base: load -> chunk -> contextual description (optional) -> BGE-M3 embed -> Milvus.
#
# Usage:
#   python -m scripts.knowledge.fetch_sources          # download the sources first
#   python -m scripts.build_knowledge_base             # every source in scripts/knowledge/sources.json
#   python -m scripts.build_knowledge_base --only ts-generics --no-context
#   python -m scripts.build_knowledge_base --file notes.md --course-id <uuid>

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from backend.config import get_settings
from backend.core.knowledge_base import COLLECTION_NAME, BGEMEmbedder, DocumentChunk, KnowledgeBaseClient, generate_chunk_id
from backend.core.llm_factory import get_llm
from scripts.knowledge.fetch_sources import OUT_DIR, load_manifest

# ── Constants ──

MAX_CONTEXT_CONCURRENCY = 5    # Concurrent LLM calls when generating chunk context
CONTEXT_DOC_CHARS = 8000       # Document text shown to the LLM for each chunk's context
MAX_SOURCE_NAME = 256          # Milvus source_name field length

CONTEXTUAL_CHUNK_PROMPT = """\
<document>
{document_text}
</document>

Here is a chunk from the document above:
<chunk>
{chunk_content}
</chunk>

In one concise sentence, say where this chunk sits in the document and what it covers, to improve
search retrieval of the chunk. Output only that sentence, with no prefix or label."""


# ── Step 1: load ──

def load_pdf(file_path: str) -> list[Document]:
    """One Document per page. Image-only pages come back empty and are dropped when chunking."""
    pages = PyPDFLoader(file_path).load()
    print(f"  [PDF] loaded {len(pages)} pages <- {Path(file_path).name}")
    return pages


def load_markdown(file_path: str) -> list[Document]:
    """The whole file as one Document; header splitting happens in the chunking step."""
    docs = TextLoader(file_path, encoding="utf-8").load()
    print(f"  [MD]  loaded {len(docs[0].page_content)} chars <- {Path(file_path).name}")
    return docs


def load_document(file_path: str) -> list[Document]:
    """Pick the loader by extension: .pdf or .md / .markdown."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    if ext in (".md", ".markdown"):
        return load_markdown(file_path)
    raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf / .md / .markdown "
                     "(convert Word or PowerPoint to .md with markitdown first)")


# ── Step 2: chunk ──

_MD_HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4")],
    strip_headers=False,
)

_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def split_pdf_documents(pages: list[Document]) -> list[Document]:
    """Drop empty pages, then split by characters."""
    non_empty_pages = [p for p in pages if len(p.page_content.strip()) > 20]
    if len(pages) > len(non_empty_pages):
        print(f"  skipped {len(pages) - len(non_empty_pages)} empty (image or scanned) pages")

    chunks = _CHAR_SPLITTER.split_documents(non_empty_pages)
    for chunk in chunks:
        filename = Path(chunk.metadata.get("source", "unknown")).stem
        chunk.metadata["source_name"] = f"{filename} p.{chunk.metadata.get('page', 0) + 1}"
    print(f"  [PDF] {len(non_empty_pages)} pages -> {len(chunks)} chunks")
    return chunks


def split_markdown_documents(
    docs: list[Document],
    chunk_size: int = 1200,      # Suits docs with code; 600-800 for prose only
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split by headers first, then by size; source_name records the heading path."""
    header_chunks: list[Document] = []
    for doc in docs:
        sections = _MD_HEADER_SPLITTER.split_text(doc.page_content)
        for section in sections:
            section.metadata["source"] = doc.metadata.get("source", "")
        header_chunks.extend(sections)

    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_chunks = splitter.split_documents(header_chunks)
    for chunk in final_chunks:
        source_path = chunk.metadata.get("source", "")
        filename = Path(source_path).stem if source_path else "unknown"
        headings = [chunk.metadata.get(h, "") for h in ("H1", "H2", "H3", "H4")]
        headings = [h for h in headings if h]
        name = f"{filename} > {' > '.join(headings)}" if headings else filename
        chunk.metadata["source_name"] = name[:MAX_SOURCE_NAME]

    print(f"  [MD]  {len(docs)} file(s) -> {len(final_chunks)} chunks")
    return final_chunks


def split_documents(docs: list[Document], file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return split_pdf_documents(docs)
    if ext in (".md", ".markdown"):
        return split_markdown_documents(docs)
    raise ValueError(f"Unsupported file type: {ext}")


# ── Step 2.5: contextual retrieval ──

async def generate_chunk_context(llm, document_text: str, chunk_content: str,
                                 semaphore: asyncio.Semaphore) -> str:
    """One sentence placing the chunk in its document. Returns "" on failure (chunk kept as is)."""
    async with semaphore:
        try:
            prompt = CONTEXTUAL_CHUNK_PROMPT.format(document_text=document_text, chunk_content=chunk_content)
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            return (
                resp.text if hasattr(resp, "text") and not callable(resp.text) else str(resp.content)
            ).strip()
        except Exception as e:
            print(f"  [warning] context generation failed, keeping the raw chunk: {e}")
            return ""


async def add_context(chunks: list[Document], docs: list[Document],
                      concurrency: int = MAX_CONTEXT_CONCURRENCY) -> list[Document]:
    """Prepend "<context sentence>\\n\\n" to each chunk, so the embedding encodes where it sits
    as well as what it says."""
    full_doc_text = "\n\n".join(d.page_content for d in docs)[:CONTEXT_DOC_CHARS]
    llm = get_llm("qa", temperature=0)
    semaphore = asyncio.Semaphore(concurrency)
    contexts = await asyncio.gather(*[
        generate_chunk_context(llm, full_doc_text, c.page_content, semaphore) for c in chunks
    ])

    enriched = 0
    for chunk, ctx in zip(chunks, contexts):
        if ctx:
            chunk.page_content = f"{ctx}\n\n{chunk.page_content}"
            enriched += 1
    print(f"  context added to {enriched}/{len(chunks)} chunks")
    return chunks


# ── Step 3: embed ──

BATCH_SIZE = 12   # BGE-M3 batch size: balances speed and memory


def embed_chunks(chunks: list[Document], course_id: str, document_id: str,
                 tenant_id: str, version: str = "1.0") -> list[DocumentChunk]:
    """Dense + sparse BGE-M3 vectors for each chunk, ready to write to Milvus."""
    embedder = BGEMEmbedder.get_instance()   # Singleton; loads the model on first use
    doc_chunks: list[DocumentChunk] = []
    total = len(chunks)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        dense_vecs, sparse_vecs = embedder.encode([c.page_content for c in batch], batch_size=BATCH_SIZE)
        for i, (chunk, dense, sparse) in enumerate(zip(batch, dense_vecs, sparse_vecs)):
            index = batch_start + i
            doc_chunks.append(DocumentChunk(
                id=generate_chunk_id(chunk.page_content, document_id, index),
                content=chunk.page_content,
                embedding=dense,
                sparse_embedding=sparse,
                course_id=course_id,
                document_id=document_id,
                source_name=chunk.metadata.get("source_name", ""),
                chunk_type=chunk.metadata.get("chunk_type", "text"),
                chunk_index=index,
                version=version,
                tenant_id=tenant_id,
            ))
        print(f"  embedded {min(batch_start + BATCH_SIZE, total)}/{total}")
    return doc_chunks


# ── Step 4: write to Milvus ──

def write_to_milvus(doc_chunks: list[DocumentChunk]) -> None:
    """Delete the document's old chunks, then upsert, so an update leaves nothing stale."""
    if not doc_chunks:
        print("  no chunks to write, skipped")
        return
    kb = KnowledgeBaseClient()
    document_id = doc_chunks[0].document_id
    kb.delete_document_chunks(document_id)
    written = kb.upsert_chunks(doc_chunks)
    print(f"  wrote {written} chunks -> {COLLECTION_NAME} (old chunks of {document_id[:8]} removed)")


# ── Pipeline ──

async def build_pipeline(file_path: str, course_id: str, document_id: str, tenant_id: str,
                         version: str = "1.0", use_context: bool = True) -> int:
    """Load, chunk, add context (optional), embed and write one file. Returns the chunk count."""
    print(f"\n== {Path(file_path).name}  (document {document_id[:8]}, course {course_id[:8]})")
    docs = load_document(file_path)
    chunks = split_documents(docs, file_path)
    if use_context and chunks:
        chunks = await add_context(chunks, docs)
    doc_chunks = embed_chunks(chunks, course_id, document_id, tenant_id, version)
    write_to_milvus(doc_chunks)
    return len(doc_chunks)


def source_ids(source: dict) -> tuple[str, str]:
    """Stable (course_id, document_id) for a manifest source: rebuilding replaces, never duplicates.
    Each topic is one course, so QA can filter by topic."""
    course_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"eduagent:topic:{source['topic']}"))
    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source["url"]))
    return course_id, document_id


async def build_from_manifest(only: list[str] | None, tenant_id: str, use_context: bool) -> None:
    total = 0
    for source in load_manifest()["sources"]:
        if only and source["id"] not in only:
            continue
        path = OUT_DIR / source["topic"] / f"{source['id']}.md"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run python -m scripts.knowledge.fetch_sources")
        course_id, document_id = source_ids(source)
        total += await build_pipeline(str(path), course_id, document_id, tenant_id, use_context=use_context)
    print(f"\nDone: {total} chunks in {COLLECTION_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the knowledge base in Milvus.")
    parser.add_argument("--only", nargs="*", help="Manifest source ids to build (default: all)")
    parser.add_argument("--file", help="Build one local file instead of the manifest")
    parser.add_argument("--course-id", help="Course id for --file")
    parser.add_argument("--document-id", help="Document id for --file; reuse it to replace the document")
    parser.add_argument("--no-context", action="store_true", help="Skip contextual descriptions (no LLM calls)")
    args = parser.parse_args()
    tenant_id = get_settings().default_tenant_id

    if args.file:
        if not args.course_id:
            parser.error("--file needs --course-id")
        document_id = args.document_id or str(uuid.uuid4())
        asyncio.run(build_pipeline(args.file, args.course_id, document_id, tenant_id,
                                   use_context=not args.no_context))
        print(f"document_id = {document_id} (pass it again to update this document)")
    else:
        asyncio.run(build_from_manifest(args.only, tenant_id, use_context=not args.no_context))


if __name__ == "__main__":
    main()
