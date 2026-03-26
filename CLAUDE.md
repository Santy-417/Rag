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

**UI structure:** Three tabs — `💬 Chat`, `🎤 Voz`, `🕸️ Grafo Semántico`.

**RAG pipeline flow (Chat tab):**
1. PDF upload → `PyPDFLoader` extracts pages → `RecursiveCharacterTextSplitter` creates chunks (900 chars, 200 overlap, semantic separators)
2. Chunks → OpenAI `text-embedding-3-small` → persisted in ChromaDB at `data/chroma/`
3. User query → `check_relevance()` checks similarity threshold (0.3) before invoking LLM — returns "not found" if below threshold
4. MMR retriever (k=6, fetch_k=20, lambda=0.5) → `ConversationalRetrievalChain` with `gpt-4o-mini` → response with page citations

**Voice tab (Fase 9):**
- `transcribe_audio()` → OpenAI Whisper API (`whisper-1`), language=es
- Chain is invoked **directly without `check_relevance()`** — voice transcriptions (Whisper) produce natural language phrasing that yields higher L2 distances than typed text, causing false negatives below the 0.3 threshold. The grounded system prompt handles irrelevant queries instead.
- `text_to_speech()` → gTTS generates MP3 played via `st.audio()`
- Dependencies: `streamlit-mic-recorder`, `gTTS`

**Semantic graph tab (Fase 10):**
- `extract_entities_llm()` → sends up to 15 chunks to `gpt-4o-mini`, asks for JSON with entities and relations
- `build_nx_graph()` → `nx.DiGraph` with color-coded node types (Technology, Concept, Person, Organization, Process, Tool)
- `render_pyvis_html()` → PyVis Network with ForceAtlas2 physics, rendered via `st.components.v1.html()`
- Entities/graph stored in `st.session_state.graph_entities`, `graph_relations`, `graph_html`
- Dependencies: `networkx`, `pyvis`

**Key design decisions:**
- ChromaDB loaded from disk on startup — PDFs don't need reprocessing between sessions
- PDF deduplication via SHA-256 hash in `st.session_state.pdf_hash`
- `ConversationBufferWindowMemory(k=6)` — shared between Chat and Voice tabs; clearing chat recreates the chain
- `get_embeddings()` cached with `@st.cache_resource` — instantiated once per session
- PyMuPDF (`fitz`) imported with try/except — used only for image counting, falls back gracefully

**Session state keys:** `chat_history`, `vector_store`, `chain`, `pdf_processed`, `pdf_hash`, `pdf_stats`, `voice_transcript`, `voice_response`, `voice_audio_bytes`, `graph_entities`, `graph_relations`, `graph_html`

**Constants:**
- `SIMILARITY_THRESHOLD = 0.3` — used only in Chat tab; Voice tab bypasses this
- `MAX_PDF_SIZE_MB = 50`
- `CHROMA_DIR = "data/chroma"`

## Deployment (Streamlit Community Cloud)

- `tiktoken` must use `>=` not `==` — strict pin fails on Python 3.14 (no prebuilt wheel, requires Rust compiler not available on Streamlit Cloud)
- `data/` and `.env` are excluded by `.gitignore` — secrets must be set in Streamlit Cloud dashboard
- ChromaDB storage is ephemeral on Streamlit Cloud — users re-upload PDF each session
