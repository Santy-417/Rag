# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the application
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Enable debug logging (shows similarity scores, retrieved chunks, response times)
DEBUG_RAG=true streamlit run app.py
```

## Environment Setup

Requires a `.env` file with:
```
OPENAI_API_KEY=sk-...
DEBUG_RAG=true   # optional, defaults to true
```

## Architecture

Single-file application (`app.py`) using Streamlit for UI and LangChain for the RAG pipeline. All logic lives in one file — no separate modules.

**RAG pipeline flow:**
1. PDF upload → `PyPDFLoader` extracts pages → `RecursiveCharacterTextSplitter` creates chunks (900 chars, 200 overlap, semantic separators)
2. Chunks → OpenAI `text-embedding-3-small` → persisted in ChromaDB at `data/chroma/`
3. User query → `similarity_search_with_relevance_scores` checks threshold (0.3) before invoking LLM — returns "not found" message if below threshold
4. MMR retriever (k=6, fetch_k=20, lambda=0.5) → `ConversationalRetrievalChain` with `gpt-4o-mini` → response with page citations

**Key design decisions:**
- ChromaDB is loaded from disk on startup if it exists, so PDFs don't need reprocessing between sessions
- PDF deduplication via SHA-256 hash stored in `st.session_state.pdf_hash` — same file won't be re-embedded
- `ConversationBufferWindowMemory(k=6)` keeps last 6 turns; clearing chat recreates the chain object to reset internal memory
- `get_embeddings()` is wrapped in `@st.cache_resource` — embeddings model is only instantiated once per session
- PyMuPDF (`fitz`) is imported with a try/except — used only for image counting in PDF stats, falls back gracefully if unavailable

**Session state keys:** `chat_history`, `vector_store`, `chain`, `pdf_processed`, `pdf_hash`, `pdf_stats`

**Constants to know:**
- `SIMILARITY_THRESHOLD = 0.3` — increase to make retrieval stricter
- `MAX_PDF_SIZE_MB = 50` — upload limit
- `CHROMA_DIR = "data/chroma"` — vector store location
