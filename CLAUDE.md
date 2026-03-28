# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the application
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Enable debug logging (shows retrieved chunks, response times)
DEBUG_RAG=true streamlit run app.py
```

## Environment Setup

Requires a `.env` file with:
```
OPENAI_API_KEY=sk-...
DEBUG_RAG=true   # optional, defaults to true
```

## Architecture

Single-file application (`app.py`) + `voice_avatar.py` module. Streamlit for UI, LangChain for embeddings/vector store only — RAG chain uses direct OpenAI API calls.

**UI structure:** Three tabs — `💬 Chat`, `🎤 Voz`, `🕸️ Grafo Semántico`. All tabs show a `_render_lock_screen()` until a PDF is processed in the current session.

**Theme:** `.streamlit/config.toml` sets base dark colors. Additional CSS is injected via `st.markdown()` at app startup (neutral dark palette: `#0D0D0D` bg, `#161616` cards, `#C0392B` red accent only on interactive elements).

**RAG pipeline flow (Chat + Voice tabs):**
1. PDF upload → `fitz.get_text()` extracts text page-by-page (PyPDF fallback) → `RecursiveCharacterTextSplitter` creates chunks (1500 chars, 300 overlap, separators: `["\n\n", "\n", ".", " "]` — no `""` to avoid mid-word splits)
2. Chunks → OpenAI `text-embedding-3-small` → persisted in ChromaDB at `data/chroma/`
3. User query → `_DirectRAGChain.invoke()` → MMR retriever (k=12, fetch_k=50, lambda_mult=0.5) fetches diverse chunks → `gpt-4o-mini` with grounded system prompt → response with page citations

**`_DirectRAGChain`** (direct OpenAI, no LangChain chains):
- Builds messages list from system prompt + conversation history (last 6 turns) + retrieved context
- `self._history` accumulates Q&A pairs; shared between Chat and Voice tabs via `st.session_state.chain`
- Clearing chat recreates the chain (resets history)

**System prompt key rules:**
- If context has partial info → present it as partial ("Según los fragmentos disponibles…"), don't refuse
- For list/count questions → list everything found even if incomplete
- Only say "No se encontró información" when context has zero relevant data

**Voice tab (Fase 9):**
- `voice_avatar.py` → `render_avatar(state)` renders an animated SVG robot via `st.components.v1.html()`; states: `idle`, `listening`, `processing`, `speaking`
- `transcribe_audio()` → OpenAI Whisper API (`whisper-1`), language=es
- Chain is invoked directly — no relevance gate (Whisper paraphrases queries; grounded prompt handles irrelevance)
- `text_to_speech()` → gTTS generates MP3 played via `st.audio()`

**Semantic graph tab (Fase 10):**
- `extract_entities_llm()` → sends up to 15 chunks to `gpt-4o-mini`, returns JSON with entities/relations
- `build_nx_graph()` → `nx.DiGraph` with color-coded node types (Technology, Concept, Person, Organization, Process, Tool)
- `render_pyvis_html()` → PyVis Network with ForceAtlas2, `bgcolor="#0D0D0D"`, rendered via `st.components.v1.html()`
- `export_graph_png()` → matplotlib render of the graph as PNG (dark background)
- Export buttons: PNG download + HTML download shown after graph is generated
- Entities/graph stored in `st.session_state.graph_entities`, `graph_relations`, `graph_html`

**Key design decisions:**
- ChromaDB auto-loaded from disk on startup (vector_store/chain initialized), but `pdf_processed = False` — tabs stay locked until user explicitly uploads and processes a PDF in the current session
- "Resetear vector store" button: releases ChromaDB connection (`vs._client = None`, `gc.collect()`) before calling `shutil.rmtree()` to avoid PermissionError on Windows; shows fallback warning if file still locked
- PDF deduplication via SHA-256 hash in `st.session_state.pdf_hash`
- `get_embeddings()` cached with `@st.cache_resource` — instantiated once per session
- MMR retriever (`lambda_mult=0.5`) preferred over pure similarity for aggregate queries ("list all countries") because diversity forces retrieval across different document sections
- `check_relevance()` function exists but is NOT called — kept for reference; was causing false negatives

**Session state keys:** `chat_history`, `vector_store`, `chain`, `pdf_processed`, `pdf_hash`, `pdf_stats`, `voice_transcript`, `voice_response`, `voice_audio_bytes`, `graph_entities`, `graph_relations`, `graph_html`

**Constants:**
- `SIMILARITY_THRESHOLD = 0.1` — defined but unused (check_relevance not called)
- `NO_INFO_PHRASE = "No se encontró información suficiente"` — used to suppress source citations when model returns no answer
- `MAX_PDF_SIZE_MB = 50`
- `CHROMA_DIR = "data/chroma"`

## Project files

```
rag-pdf-chat/
├── app.py              # Main application (all RAG + UI logic)
├── voice_avatar.py     # Animated SVG robot avatar for voice tab
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .streamlit/
│   └── config.toml     # Streamlit dark theme configuration
├── CLAUDE.md           # This file
└── README.md
```

## Deployment (Streamlit Community Cloud)

- `tiktoken` must use `>=` not `==` — strict pin fails on Python 3.14 (no prebuilt wheel)
- `data/` and `.env` are excluded by `.gitignore` — secrets must be set in Streamlit Cloud dashboard
- ChromaDB storage is ephemeral on Streamlit Cloud — users re-upload PDF each session
- `.streamlit/config.toml` must be committed for the theme to apply on Cloud
