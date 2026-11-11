# backend/main.py

import os
import sys

# Windows conda only: add the base env's Library/bin to the DLL search path so _lzma.pyd
# finds liblzma.dll (skipped on Linux and macOS)
if sys.platform == "win32":
    _conda_base_lib_bin = os.path.normpath(
        os.path.join(os.path.dirname(sys.executable), "..", "..", "Library", "bin")
    )
    if os.path.isdir(_conda_base_lib_bin):
        os.add_dll_directory(_conda_base_lib_bin)

from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.core.logger import configure_logging, get_logger
from backend.api.router import api_router

settings = get_settings()

# MCP sub-apps are created before the lifespan so it and the mounts below share one instance
from backend.mcp.knowledge_base_server import mcp as kb_mcp      # Knowledge base MCP server
from backend.mcp.web_search_server import mcp as ws_mcp          # Web search MCP server

_kb_app = kb_mcp.streamable_http_app()                          # MCP servers as ASGI sub-apps
_ws_app = ws_mcp.streamable_http_app()

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger = get_logger(__name__)
    logger.info("app.starting | env=%s port=%s", settings.app_env, settings.app_port)

    # 1. Idempotent schema patches, on every start
    try:
        from backend.db.migrations import run_migrations
        await run_migrations()
    except Exception as e:
        logger.warning("app.migrations_failed | error=%s", e)   # Logged; the app still starts

    # 2. Load the three local models now, so the first request is not slow. One after another
    #    in one worker thread: loading transformers models concurrently races on its global
    #    "meta device" setup and can leave a model without weights ("meta tensors" errors).
    import asyncio
    try:
        from backend.core.reranker import BGEReranker
        from backend.core.query_classifier import QueryClassifier
        from backend.core.knowledge_base import BGEMEmbedder

        def load_models() -> None:
            for get_instance in (BGEReranker.get_instance, QueryClassifier.get_instance, BGEMEmbedder.get_instance):
                get_instance()

        await asyncio.get_running_loop().run_in_executor(None, load_models)
        logger.info("app.local_models_warmed_up")
    except Exception as e:
        logger.warning("app.local_models_warmup_failed | error=%s", e)  # Logged; the app still starts

    async with AsyncExitStack() as stack:
        # 3. Keep interview state in Postgres, so sessions survive a restart. Without the
        #    database the interview graph keeps its in-memory checkpointer.
        try:
            from backend.api.v1 import interview
            from backend.core.checkpointer import postgres_checkpointer
            interview.use_checkpointer(await stack.enter_async_context(postgres_checkpointer()))
        except Exception as e:
            logger.warning("app.postgres_checkpointer_failed | error=%s", e)

        # 4. Run the MCP sub-apps' lifespans (mount does not run them)
        await stack.enter_async_context(_kb_app.router.lifespan_context(_kb_app))
        await stack.enter_async_context(_ws_app.router.lifespan_context(_ws_app))
        logger.info("app.started")
        yield                                                   # The app runs here

        # Shutdown
        logger.info("app.shutting_down")
        from backend.core.llm_factory import LLMFactory
        LLMFactory.clear_cache()
        logger.info("app.shutdown_complete")


app = FastAPI(
    title="EduAgent API",
    description="AI teaching assistant agents API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS: the frontend dev servers (ports 3000, 5173, 8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:5173", "http://localhost:8080",
        "http://127.0.0.1:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

app.mount("/mcp/kb",         _kb_app)
app.mount("/mcp/web-search", _ws_app)


@app.get("/health", tags=["System"])                     # Liveness check
async def health_check():
    return {"status": "ok", "env": settings.app_env}
