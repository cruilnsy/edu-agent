# Knowledge base sources

The QA agent answers from twelve public documentation pages in three topics. `sources.json`
pins each page to a commit, so the knowledge base and the QA eval labels stay reproducible.

The documents are **not** stored in this repo. `fetch_sources.py` downloads them into
`data/knowledge/` (git-ignored) and strips doc-site markup (front matter, MDX components,
MDN macros, react.dev exercises, CSS in sandboxes); code blocks are kept as they are.

```bash
python -m scripts.knowledge.fetch_sources     # download and clean
python scripts/init_milvus.py                 # only if the collection does not exist yet (drops it)
python -m scripts.build_knowledge_base        # chunk, add context, embed, write to Milvus
```

Each topic becomes one `course_id` and each page one `document_id`, both derived from
`sources.json`, so rebuilding replaces a document instead of duplicating it.

## Attribution

| Topic | Document | Source | License |
|---|---|---|---|
| Python, backend and full-stack | Dependencies; OAuth2 with Password, Bearer with JWT tokens | [FastAPI docs](https://fastapi.tiangolo.com/), © Sebastián Ramírez | MIT |
| | Coroutines and Tasks | [Python documentation](https://docs.python.org/3/library/asyncio-task.html), © Python Software Foundation | PSF-2.0 |
| | Cross-Origin Resource Sharing (CORS) | [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS), © Mozilla Contributors | CC BY-SA 2.5 |
| | Building best practices | [Docker docs](https://docs.docker.com/build/building/best-practices/), © Docker Inc. | Apache-2.0 |
| TypeScript and React | Everyday Types; Generics | [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/2/everyday-types.html), © Microsoft | CC BY 4.0 |
| | Thinking in React; Reusing Logic with Custom Hooks | [react.dev](https://react.dev/learn), © Meta Platforms | CC BY 4.0 |
| AI engineering | How do Transformers work? | [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/4), © Hugging Face | Apache-2.0 |
| | Retrieval Augmented Generation (RAG) | [Prompt Engineering Guide](https://www.promptingguide.ai/techniques/rag), © DAIR.AI | MIT |
| | Graph API overview | [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/graph-api), © LangChain | MIT |

The cleaned copies in `data/knowledge/` are modified versions of these pages (markup removed).
