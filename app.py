"""
RAG PDF Chat — Aplicación local para conversar con documentos PDF
usando Retrieval-Augmented Generation con OpenAI.

Versión optimizada:
  - Retriever MMR (k=6, fetch_k=20)
  - Chunking mejorado (900/200)
  - System prompt grounded anti-alucinación
  - Manejo de "no encontrado" con threshold de similitud
  - Fuentes visibles en la UI
  - Memoria con ventana deslizante (k=6)
  - Caché de PDF (no reprocesa el mismo archivo)
  - Logging de depuración configurable
  - [Fase 9] Voz: STT con OpenAI Whisper + TTS con gTTS
  - [Fase 10] Grafo semántico: extracción LLM + NetworkX + PyVis
"""

import os
import re
import io
import sys
import json
import time
import hashlib
import tempfile
import logging
from collections import Counter

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# --- Fase 9: Voz ---
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# --- Fase 10: Grafo semántico ---
try:
    import networkx as nx
    from pyvis.network import Network
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from voice_avatar import render_avatar

# ---------------------------------------------------------------------------
# Configuración de depuración
# ---------------------------------------------------------------------------
DEBUG_RAG = os.getenv("DEBUG_RAG", "true").lower() in ("true", "1", "yes")

logging.basicConfig(
    level=logging.DEBUG if DEBUG_RAG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords (español + inglés) para análisis de palabras repetidas
# ---------------------------------------------------------------------------
STOPWORDS = {
    # Artículos y determinantes ES
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Preposiciones ES
    "de", "del", "al", "a", "ante", "bajo", "con", "contra", "desde",
    "en", "entre", "hacia", "hasta", "para", "por", "según", "sin",
    "sobre", "tras", "durante", "mediante",
    # Conjunciones ES
    "y", "e", "ni", "o", "u", "pero", "sino", "aunque", "que", "si",
    "como", "cuando", "donde", "quien", "cual", "cuyo",
    # Pronombres ES
    "yo", "tu", "tú", "él", "ella", "nosotros", "vosotros", "ellos",
    "ellas", "me", "te", "le", "nos", "os", "les", "se", "lo",
    "mi", "mis", "tus", "sus", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "aquel", "aquella", "aquellos", "aquellas",
    # Verbos comunes ES
    "es", "son", "fue", "era", "han", "ha", "he", "ser", "estar",
    "haber", "tener", "hacer", "hay", "sido", "sido", "estar",
    # Adverbios ES
    "no", "sí", "muy", "más", "menos", "tan", "tanto", "también",
    "ya", "así", "aquí", "allí", "ahora", "antes", "después",
    # Inglés básico
    "the", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these", "those",
    "it", "its", "not", "nor", "all", "can", "just", "also", "than",
    "then", "when", "where", "which", "who", "how", "what", "if",
}

# ---------------------------------------------------------------------------
# 1. Configuración inicial
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.path.join("data", "chroma")
MAX_PDF_SIZE_MB = 50
SIMILARITY_THRESHOLD = 0.1  # Umbral mínimo de relevancia (0-1, mayor = más estricto)
NO_INFO_PHRASE = "No se encontró información suficiente"  # Frase que indica respuesta vacía

st.set_page_config(
    page_title="📄 RAG PDF Chat",
    page_icon="📄",
    layout="centered",
)

# ── UI — Dark Neutral Theme (red accent only) ───────────────────────────────
st.markdown("""
<style>
:root {
    --bg-deep:      #0D0D0D;
    --bg-card:      #161616;
    --bg-border:    #2A2A2A;
    --accent-red:   #C0392B;
    --accent-red-h: #E74C3C;
    --accent-glow:  rgba(192,57,43,0.20);
    --text-primary: #F0F0F0;
    --text-muted:   #888888;
}

/* App background */
.stApp { background-color: var(--bg-deep); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-card);
    border-right: 1px solid var(--bg-border);
}
[data-testid="stSidebarContent"] { background-color: var(--bg-card); }

/* Tabs — fondo neutro, rojo solo en activa */
[data-baseweb="tab-list"] {
    background-color: var(--bg-card) !important;
    border-bottom: 1px solid var(--bg-border) !important;
    padding: 0 0.5rem;
    border-radius: 8px 8px 0 0;
}
[data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-muted) !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 6px 6px 0 0 !important;
    font-weight: 500;
    transition: color 0.2s ease, background-color 0.2s ease;
}
[data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background-color: rgba(255,255,255,0.04) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--text-primary) !important;
    background-color: transparent !important;
}
/* Indicador rojo bajo la tab activa — usa el elemento nativo de Streamlit */
[data-baseweb="tab-highlight"] {
    background-color: var(--accent-red) !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0 !important;
}
[data-baseweb="tab-border"]    { background-color: var(--bg-border) !important; }

/* Buttons — dark card, borde gris, rojo en hover */
[data-testid^="baseButton-"],
.stButton > button,
[data-testid="stButton"] > button {
    background-color: #1C1C1C !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 8px !important;
    font-weight: 600;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
}
[data-testid^="baseButton-"]:hover,
.stButton > button:hover,
[data-testid="stButton"] > button:hover {
    border-color: var(--accent-red) !important;
    box-shadow: 0 0 12px var(--accent-glow) !important;
    transform: translateY(-1px);
}
[data-testid^="baseButton-"]:active,
.stButton > button:active { transform: translateY(0); }

/* Chat — user bubble — neutro sin rojo */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: rgba(255,255,255,0.03);
    border: 1px solid var(--bg-border);
    border-radius: 12px;
    margin-bottom: 0.75rem;
}

/* Chat — assistant bubble — borde izq rojo (único acento) */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-left: 3px solid var(--accent-red);
    border-radius: 12px;
    margin-bottom: 0.75rem;
}

/* Chat input bar */
[data-testid="stBottom"] {
    background-color: var(--bg-deep);
    border-top: 1px solid var(--bg-border);
    padding-top: 0.5rem;
}
[data-testid="stChatInput"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-red) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: var(--bg-card);
    border-radius: 10px;
    padding: 0.5rem;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg-card) !important;
    border: 2px dashed var(--bg-border) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent-red) !important;
}

/* Metrics — valores en blanco, sin rojo */
[data-testid="stMetric"] {
    background-color: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: var(--bg-card);
    border: 1px solid var(--bg-border) !important;
    border-radius: 10px;
}
[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    border-radius: 10px;
    font-weight: 500;
}
[data-testid="stExpander"] summary:hover {
    background-color: rgba(255,255,255,0.03) !important;
}
[data-testid="stExpanderDetails"] {
    background-color: var(--bg-card);
    border-top: 1px solid var(--bg-border);
}

/* Headings — blanco puro, sin gradiente de color */
h1, h2, h3 {
    color: var(--text-primary);
    font-weight: 700;
}

/* Dividers */
hr { border-color: var(--bg-border) !important; opacity: 1; }

/* DataFrames */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-radius: 8px;
    overflow: hidden;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3A3A3A, var(--accent-red));
    border-radius: 4px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track  { background: var(--bg-deep); }
::-webkit-scrollbar-thumb  { background: #3A3A3A; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-red); }

/* Audio player dark */
audio { filter: invert(1) hue-rotate(180deg); width: 100%; border-radius: 8px; }

/* Shimmer animation (title) */
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
</style>
""", unsafe_allow_html=True)

os.makedirs(CHROMA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Validación de API key
# ---------------------------------------------------------------------------
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-xxxx"):
    st.error(
        "❌ **OPENAI_API_KEY** no configurada. "
        "Crea un archivo `.env` con tu clave (ver `.env.example`)."
    )
    logger.error("OPENAI_API_KEY no encontrada o es el placeholder por defecto.")
    st.stop()

# ---------------------------------------------------------------------------
# System prompt grounded (anti-alucinación)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Eres un asistente experto en análisis de documentos.

REGLAS ESTRICTAS:

1. Responde únicamente con información contenida en el contexto proporcionado.
2. Si el contexto contiene información PARCIAL sobre la pregunta, preséntala indicando que
   puede no ser exhaustiva (ej: "Según los fragmentos disponibles, se mencionan al menos...").
   Solo di "No se encontró información suficiente en el documento." cuando el contexto
   NO contenga NINGÚN dato relevante sobre el tema preguntado.
3. No inventes información.
4. Para preguntas que piden listar o contar elementos (países, autores, conceptos, etc.),
   lista todos los que aparezcan en el contexto aunque puedas no tener la lista completa.
5. Cuando sea posible, menciona el criterio o dato clave del documento.
6. Mantén coherencia con el historial de conversación.
7. Prioriza exactitud sobre fluidez.

FORMATO:
- Respuesta clara en español
- Sin relleno innecesario

CONTEXTO DEL DOCUMENTO:
{context}
"""


# ---------------------------------------------------------------------------
# 2. Funciones auxiliares
# ---------------------------------------------------------------------------

def _compute_file_hash(file_bytes: bytes) -> str:
    """Calcula SHA-256 del contenido del archivo para detectar duplicados."""
    return hashlib.sha256(file_bytes).hexdigest()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Devuelve el modelo de embeddings (cacheado para toda la sesión)."""
    logger.info("Inicializando modelo de embeddings text-embedding-3-small")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )


def load_and_split_pdf(pdf_path: str, progress_callback=None):
    """Lee un PDF con extracción de texto estándar y lo divide en chunks."""
    logger.info("Cargando PDF: %s", pdf_path)

    pages = []

    if FITZ_AVAILABLE:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError("El PDF no contiene páginas legibles.")
        logger.info("Páginas encontradas: %d", total_pages)
        for i, fitz_page in enumerate(doc):
            if progress_callback:
                progress_callback(i, total_pages)
            text = fitz_page.get_text()
            pages.append(Document(
                page_content=text,
                metadata={"page": i, "source": pdf_path},
            ))
        doc.close()
    else:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError("El PDF no contiene páginas legibles.")
        logger.info("Páginas encontradas: %d", total_pages)
        for i, pdf_page in enumerate(reader.pages):
            if progress_callback:
                progress_callback(i, total_pages)
            text = pdf_page.extract_text() or ""
            pages.append(Document(
                page_content=text,
                metadata={"page": i, "source": pdf_path},
            ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    logger.info("Chunks generados: %d", len(chunks))

    if DEBUG_RAG:
        avg_len = sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)
        logger.debug("Tamaño promedio de chunk: %.0f caracteres", avg_len)

    return pages, chunks


def create_vector_store(chunks):
    """Crea (o sobreescribe) la base vectorial en Chroma."""
    logger.info("Generando embeddings y almacenando en ChromaDB…")
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    logger.info("Base vectorial creada exitosamente en %s", CHROMA_DIR)
    return vector_store


def load_existing_vector_store():
    """Carga una base vectorial existente de disco, si existe."""
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return None
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    # Verificar que realmente tiene documentos
    if vector_store._collection.count() == 0:
        return None
    logger.info(
        "Base vectorial cargada desde disco (%d documentos).",
        vector_store._collection.count(),
    )
    return vector_store


class _DirectRAGChain:
    """
    RAG conversacional sin dependencia de langchain.chains.Chain.
    Compatible con Python 3.14 — usa la API de OpenAI directamente.
    Misma interfaz: .invoke({"question": str}) → {"answer": str, "source_documents": list}
    """

    def __init__(self, retriever, k: int = 6):
        self.retriever = retriever
        self.k = k
        self._history: list[tuple[str, str]] = []

    def invoke(self, inputs: dict) -> dict:
        import openai
        question = inputs.get("question", "")
        source_docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in source_docs)
        system_content = SYSTEM_PROMPT.replace("{context}", context)

        messages: list[dict] = [{"role": "system", "content": system_content}]
        for q, a in self._history[-self.k:]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": question})

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or ""
        self._history.append((question, answer))
        logger.debug("RAG — pregunta: %s | respuesta: %s…", question[:60], answer[:60])
        return {"answer": answer, "source_documents": source_docs}


def get_conversational_chain(vector_store):
    """Construye la cadena RAG conversacional con MMR y prompt grounded."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 50, "lambda_mult": 0.5},
    )
    logger.info("Cadena RAG directa creada (MMR k=12, fetch_k=50, lambda=0.5).")
    return _DirectRAGChain(retriever=retriever, k=6)


def check_relevance(vector_store, query: str, threshold: float = SIMILARITY_THRESHOLD):
    """
    Verifica si los chunks recuperados superan el umbral de similitud.
    Retorna (es_relevante: bool, scores: list[float]).
    """
    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query, k=6
        )
        if not results:
            return False, []

        scores = [score for _, score in results]

        if DEBUG_RAG:
            logger.debug("Scores de similitud: %s", [f"{s:.3f}" for s in scores])
            logger.debug("Threshold: %.2f", threshold)

        # Al menos un chunk debe superar el threshold
        is_relevant = any(score >= threshold for score in scores)
        return is_relevant, scores

    except Exception as e:
        logger.warning("Error al verificar relevancia: %s", e)
        return True, []  # En caso de error, dejar pasar


def format_sources(source_documents) -> str:
    """Extrae páginas únicas de los source_documents y formatea la línea de fuentes."""
    if not source_documents:
        return ""

    pages = set()
    for doc in source_documents:
        page_num = doc.metadata.get("page")
        if page_num is not None:
            pages.add(page_num + 1)  # PyPDFLoader usa 0-indexed

    if not pages:
        return ""

    sorted_pages = sorted(pages)
    page_list = ", ".join(str(p) for p in sorted_pages)
    return f"\n\n📄 **Fuentes:** página {page_list}"


def analyze_pdf_stats(file_bytes: bytes, pages: list) -> dict:
    """
    Analiza el PDF y retorna estadísticas por página:
    - Total de palabras (sin stopwords)
    - Palabras únicas
    - Palabras que se repiten (aparecen más de 1 vez) con su frecuencia
    - Cantidad de imágenes
    """
    stats = {"total_pages": len(pages), "pages": []}

    doc = None
    if FITZ_AVAILABLE:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.warning("PyMuPDF no pudo abrir el PDF para contar imágenes: %s", e)

    for i, page_doc in enumerate(pages):
        text = page_doc.page_content or ""

        # Tokenizar: solo palabras de 3+ letras, minúsculas, sin stopwords
        raw_words = re.findall(r"\b[a-záéíóúüña-z]{3,}\b", text.lower())
        words = [w for w in raw_words if w not in STOPWORDS]

        total_words = len(words)
        word_freq = Counter(words)
        repeated = dict(
            sorted(
                {w: c for w, c in word_freq.items() if c > 1}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Contar imágenes con PyMuPDF
        image_count = 0
        if doc and i < len(doc):
            try:
                image_count = len(doc[i].get_images(full=True))
            except Exception:
                image_count = 0

        stats["pages"].append({
            "page_num": i + 1,
            "total_words": total_words,
            "unique_words": len(word_freq),
            "repeated_words": repeated,
            "repeated_count": len(repeated),
            "image_count": image_count,
        })

    if doc:
        doc.close()

    logger.info(
        "Estadísticas calculadas: %d páginas, %d imágenes totales.",
        len(pages),
        sum(p["image_count"] for p in stats["pages"]),
    )
    return stats


def display_pdf_stats(stats: dict) -> None:
    """Muestra las estadísticas del PDF: tabla resumen + detalle por página."""
    with st.expander("📊 Estadísticas del PDF", expanded=True):
        total = stats["total_pages"]
        total_imgs = sum(p["image_count"] for p in stats["pages"])
        total_words = sum(p["total_words"] for p in stats["pages"])

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Total páginas", total)
        col2.metric("🔤 Total palabras", total_words)
        col3.metric("🖼️ Total imágenes", total_imgs)

        st.markdown("---")
        st.markdown("**Resumen por página**")

        summary = [
            {
                "Página": p["page_num"],
                "Palabras": p["total_words"],
                "Palabras únicas": p["unique_words"],
                "Palabras repetidas": p["repeated_count"],
                "Imágenes": p["image_count"],
            }
            for p in stats["pages"]
        ]
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Detalle por página**")
        page_labels = [f"Página {p['page_num']}" for p in stats["pages"]]
        selected_label = st.selectbox(
            "Selecciona una página", page_labels, key="stats_page_selector"
        )
        idx = page_labels.index(selected_label)
        detail = stats["pages"][idx]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Palabras", detail["total_words"])
        c2.metric("Únicas", detail["unique_words"])
        c3.metric("Repetidas", detail["repeated_count"])
        c4.metric("Imágenes", detail["image_count"])

        if detail["repeated_words"]:
            st.markdown("**Palabras repetidas (ordenadas por frecuencia):**")
            repeated_table = [
                {"Palabra": w, "Frecuencia": c}
                for w, c in detail["repeated_words"].items()
            ]
            st.dataframe(repeated_table, use_container_width=True, hide_index=True)
        else:
            st.info("No hay palabras repetidas relevantes en esta página.")


# ---------------------------------------------------------------------------
# 3a. Funciones de Voz — Fase 9
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio a texto usando OpenAI Whisper API.
    Entrada : bytes de audio (WAV/WebM/MP3).
    Proceso : envía el audio al modelo whisper-1 de OpenAI.
    Salida  : string con el texto transcrito.
    """
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    audio_buffer = io.BytesIO(audio_bytes)
    audio_buffer.name = "recording.wav"
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        language="es",
    )
    logger.info("Audio transcrito: %s", transcript.text[:80])
    return transcript.text


def text_to_speech(text: str, lang: str = "es") -> bytes:
    """
    Convierte texto a audio MP3 usando gTTS (Google Text-to-Speech).
    Entrada : string de texto.
    Proceso : llamada a Google TTS API.
    Salida  : bytes de audio MP3.
    """
    tts = gTTS(text=text[:800], lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# 3b. Funciones de Grafo Semántico — Fase 10
# ---------------------------------------------------------------------------

# Colores por tipo de entidad
ENTITY_COLORS = {
    "Technology": "#4CAF50",
    "Concept": "#2196F3",
    "Person": "#FF9800",
    "Organization": "#9C27B0",
    "Process": "#F44336",
    "Tool": "#00BCD4",
    "Other": "#607D8B",
}


def extract_entities_llm(chunks: list) -> tuple:
    """
    Extrae entidades y relaciones de los chunks usando el LLM (OpenAI).
    Entrada : lista de Document chunks del PDF.
    Proceso : envía cada chunk al LLM solicitando JSON con entidades y relaciones.
    Salida  : (entities: list[dict], relations: list[dict])

    Errores comunes:
      - JSON malformado: se captura con try/except y se omite el chunk.
      - Entidades duplicadas: se usa dict para deduplicar por nombre.
    """
    # Tomar muestra de chunks para no exceder límite de API / costos
    sample = chunks[:min(15, len(chunks))]
    all_entities: dict[str, str] = {}
    all_relations: list[dict] = []

    for chunk in sample:
        prompt = (
            "Analiza el siguiente texto y extrae entidades y relaciones.\n\n"
            f"Texto:\n{chunk.page_content[:600]}\n\n"
            "Responde ÚNICAMENTE con JSON válido (sin markdown, sin explicaciones):\n"
            '{\n'
            '  "entities": [{"name": "nombre", "type": "Technology|Concept|Person|Organization|Process|Tool|Other"}],\n'
            '  "relations": [{"source": "entidad1", "target": "entidad2", "relation": "verbo"}]\n'
            '}'
        )
        try:
            import openai as _openai
            _client = _openai.OpenAI(api_key=OPENAI_API_KEY)
            _resp = _client.chat.completions.create(
                model="gpt-4o-mini", temperature=0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            response_content = _resp.choices[0].message.content or ""
            # Limpiar posible markdown ```json ... ```
            raw = response_content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            data = json.loads(raw)

            for e in data.get("entities", []):
                name = e.get("name", "").strip()
                etype = e.get("type", "Other")
                if name and len(name) > 2:
                    all_entities[name] = etype

            for r in data.get("relations", []):
                src = r.get("source", "").strip()
                tgt = r.get("target", "").strip()
                rel = r.get("relation", "relacionado_con")
                if src and tgt and src in all_entities and tgt in all_entities:
                    all_relations.append({"source": src, "target": tgt, "relation": rel})

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Error extrayendo entidades del chunk: %s", exc)
            continue

    entities = [{"name": k, "type": v} for k, v in all_entities.items()]
    logger.info(
        "Grafo: %d entidades, %d relaciones extraídas.",
        len(entities), len(all_relations),
    )
    return entities, all_relations


def build_nx_graph(entities: list, relations: list):
    """
    Construye un grafo dirigido NetworkX a partir de entidades y relaciones.
    Entrada : listas de dicts con 'name'/'type' y 'source'/'target'/'relation'.
    Proceso : crea nodos con atributo de color por tipo y aristas con etiqueta.
    Salida  : nx.DiGraph
    """
    G = nx.DiGraph()
    for entity in entities:
        G.add_node(
            entity["name"],
            type=entity["type"],
            color=ENTITY_COLORS.get(entity["type"], "#607D8B"),
        )
    for rel in relations:
        G.add_edge(rel["source"], rel["target"], label=rel["relation"])
    return G


def render_pyvis_html(G) -> str:
    """
    Renderiza el grafo NetworkX como HTML interactivo con PyVis.
    Entrada : nx.DiGraph con atributos 'color' en nodos y 'label' en aristas.
    Proceso : convierte el grafo a PyVis Network y genera HTML.
    Salida  : string HTML con el grafo interactivo.
    """
    net = Network(
        height="520px",
        width="100%",
        bgcolor="#0D0D0D",
        font_color="#FFFFFF",
        directed=True,
    )
    net.from_nx(G)

    # Aplicar colores y tamaños a los nodos
    for node in net.nodes:
        node_id = node["id"]
        node["color"] = G.nodes[node_id].get("color", "#607D8B")
        node["size"] = 20
        node["title"] = G.nodes[node_id].get("type", "")

    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "edges": {
        "arrows": {"to": {"enabled": true}},
        "color": {"color": "#aaaaaa"},
        "font": {"color": "#cccccc", "size": 10}
      }
    }
    """)
    return net.generate_html()


def export_graph_png(G) -> bytes:
    """Renderiza el grafo NetworkX como imagen PNG usando matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#1C1C1C")
    ax.set_facecolor("#1C1C1C")

    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    node_colors = [G.nodes[n].get("color", "#607D8B") for n in G.nodes()]
    edge_labels = nx.get_edge_attributes(G, "label")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=700, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=8, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color="#666666", arrows=True,
        arrowsize=12, alpha=0.7, connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_color="#AAAAAA", font_size=7)

    ax.set_title("Grafo Semántico", color="#F0F0F0", fontsize=13, pad=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#1C1C1C")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. Estado de sesión
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = None

# Fase 9 — Voz
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "voice_response" not in st.session_state:
    st.session_state.voice_response = ""
if "voice_audio_bytes" not in st.session_state:
    st.session_state.voice_audio_bytes = None

# Fase 10 — Grafo semántico
if "graph_entities" not in st.session_state:
    st.session_state.graph_entities = []
if "graph_relations" not in st.session_state:
    st.session_state.graph_relations = []
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None

# Al iniciar, intentar cargar base vectorial existente
if st.session_state.vector_store is None:
    existing = load_existing_vector_store()
    if existing is not None:
        st.session_state.vector_store = existing
        st.session_state.chain = get_conversational_chain(existing)
        # pdf_processed permanece False — requiere upload explícito del usuario

# ---------------------------------------------------------------------------
# 4. Sidebar — Subida y procesamiento de PDF
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Subir PDF")

    uploaded_file = st.file_uploader(
        "Selecciona un archivo PDF",
        type=["pdf"],
        help="Tamaño máximo recomendado: 50 MB",
    )

    if uploaded_file is not None:
        # Validación de tamaño
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_PDF_SIZE_MB:
            st.error(
                f"❌ El PDF pesa {file_size_mb:.1f} MB. "
                f"El máximo permitido es {MAX_PDF_SIZE_MB} MB."
            )
        else:
            st.info(f"📎 **{uploaded_file.name}** — {file_size_mb:.2f} MB")

            if st.button("🚀 Procesar PDF", use_container_width=True):
                try:
                    # Verificar si es el mismo PDF (caché por hash)
                    file_bytes = uploaded_file.getvalue()
                    file_hash = _compute_file_hash(file_bytes)

                    if (
                        file_hash == st.session_state.pdf_hash
                        and st.session_state.pdf_processed
                    ):
                        st.success(
                            "✅ Este PDF ya fue procesado. "
                            "Puedes hacer preguntas directamente."
                        )
                        logger.info(
                            "PDF '%s' ya procesado (hash coincide). "
                            "Omitiendo reprocesamiento.",
                            uploaded_file.name,
                        )
                    else:
                        # Guardar temporalmente el archivo subido
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(file_bytes)
                            tmp_path = tmp.name

                        # Progreso página a página con visión multimodal
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()

                        def update_progress(current, total):
                            progress_bar.progress((current + 1) / total)
                            status_text.text(
                                f"🔍 Analizando página {current + 1} de {total}…"
                            )

                        try:
                            pages, chunks = load_and_split_pdf(
                                tmp_path, progress_callback=update_progress
                            )
                        finally:
                            progress_bar.empty()
                            status_text.empty()
                            os.unlink(tmp_path)

                        # Analizar estadísticas
                        pdf_stats = analyze_pdf_stats(file_bytes, pages)

                        if not chunks:
                            st.error("❌ No se generaron chunks del PDF.")
                            st.stop()

                        with st.spinner("🧠 Generando embeddings (solo una vez)…"):
                            vector_store = create_vector_store(chunks)

                        # Crear cadena
                        chain = get_conversational_chain(vector_store)

                        # Guardar en sesión
                        st.session_state.vector_store = vector_store
                        st.session_state.chain = chain
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_hash = file_hash
                        st.session_state.pdf_stats = pdf_stats
                        st.session_state.chat_history = []  # Reset chat

                        st.success(
                            f"✅ PDF procesado correctamente.\n\n"
                            f"- 📄 **Páginas:** {len(pages)}\n"
                            f"- 🧩 **Chunks:** {len(chunks)}"
                        )
                        logger.info(
                            "PDF '%s' procesado: %d páginas, %d chunks.",
                            uploaded_file.name,
                            len(pages),
                            len(chunks),
                        )

                except ValueError as ve:
                    st.error(f"❌ Error en el PDF: {ve}")
                    logger.error("Error de validación del PDF: %s", ve)
                except Exception as e:
                    st.error(f"❌ Error inesperado al procesar el PDF: {e}")
                    logger.exception(
                        "Error inesperado durante el procesamiento del PDF."
                    )

    st.divider()

    # Botones de gestión
    if st.session_state.pdf_processed:
        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state.chat_history = []
            # Recrear la cadena para limpiar la memoria interna
            if st.session_state.vector_store is not None:
                st.session_state.chain = get_conversational_chain(
                    st.session_state.vector_store
                )
            st.rerun()

        if st.button("⚠️ Resetear vector store", use_container_width=True):
            import shutil, gc
            vs = st.session_state.get("vector_store")
            # Liberar conexión ChromaDB ANTES de intentar borrar archivos
            st.session_state.vector_store = None
            st.session_state.chain = None
            try:
                if vs is not None and hasattr(vs, "_client"):
                    vs._client = None
                del vs
            except Exception:
                pass
            gc.collect()
            # Resetear estado de sesión
            for _k in ["pdf_hash", "pdf_stats", "voice_transcript", "voice_response", "voice_audio_bytes", "graph_html"]:
                st.session_state[_k] = None
            st.session_state.chat_history = []
            st.session_state.graph_entities = []
            st.session_state.graph_relations = []
            st.session_state.pdf_processed = False
            # Intentar borrar el directorio ChromaDB
            if os.path.exists(CHROMA_DIR):
                try:
                    shutil.rmtree(CHROMA_DIR)
                    os.makedirs(CHROMA_DIR, exist_ok=True)
                except PermissionError:
                    st.warning(
                        "⚠️ El archivo SQLite de ChromaDB sigue en uso. "
                        "Reinicia la app (`Ctrl+C` → `streamlit run app.py`) para limpiarlo completamente."
                    )
            st.rerun()

    # Mostrar configuración de debug en sidebar
    if DEBUG_RAG:
        st.divider()
        
    st.markdown("""
<div style="
    margin-top:1rem;padding:0.75rem;
    background:linear-gradient(135deg,rgba(30,30,30,0.8),rgba(22,22,22,0.9));
    border:1px solid #2A2A2A;border-radius:8px;text-align:center;
">
    <div style="
        font-size:0.75rem;font-weight:700;letter-spacing:0.1em;
        background:linear-gradient(90deg,#888,#F0F0F0);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin-bottom:0.4rem;
    ">NEXFLOW AI &copy; 2025</div>
    <div style="font-size:0.65rem;color:#64748B;line-height:1.6;">
        Samuel Aristizabal Botero<br>
        Santiago Chavarro Osorio<br>
        Santiago Andrés Giraldo Granada
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5. Área principal — Tabs: Chat / Voz / Grafo Semántico
# ---------------------------------------------------------------------------
def _render_lock_screen(feature_name: str) -> None:
    """Muestra una pantalla de bloqueo estilizada cuando no hay PDF cargado."""
    st.markdown(
        f"""
<div style="
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;min-height:300px;text-align:center;
    background:#161616;border:1px solid #2A2A2A;border-radius:12px;
    padding:3rem 2rem;margin:1rem 0;
">
    <div style="font-size:2.5rem;margin-bottom:1rem">🔒</div>
    <div style="color:#F0F0F0;font-size:1.05rem;font-weight:600;margin-bottom:0.5rem;">
        Sube un PDF para activar {feature_name}
    </div>
    <div style="color:#888888;font-size:0.82rem;line-height:1.6;">
        Ve a la barra lateral → carga tu documento →<br>
        presiona <b style="color:#C0392B">🚀 Procesar PDF</b>
    </div>
</div>""",
        unsafe_allow_html=True,
    )


st.markdown("""
<h1 style="
    background:linear-gradient(90deg,#888 0%,#F0F0F0 40%,#CCC 70%,#888 100%);
    background-size:200% auto;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    animation:shimmer 4s linear infinite;
    font-size:2.2rem;font-weight:800;margin-bottom:0.5rem;
">📄 RAG PDF Chat</h1>
""", unsafe_allow_html=True)

tab_chat, tab_voice, tab_graph = st.tabs(["💬 Chat", "🎤 Voz", "🕸️ Grafo Semántico"])

# ── Tab 1: Chat ──────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.pdf_processed or st.session_state.chain is None:
        _render_lock_screen("el chat con el documento")
    else:
        # Mostrar estadísticas si hay un PDF procesado
        if st.session_state.pdf_stats is not None:
            display_pdf_stats(st.session_state.pdf_stats)

        # Mostrar historial de chat
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Campo de entrada
        if prompt := st.chat_input("Escribe tu pregunta sobre el PDF…"):
            if not prompt.strip():
                st.warning("⚠️ Escribe una pregunta válida.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Pensando…"):
                        try:
                            start_time = time.time()
                            response = st.session_state.chain.invoke({"question": prompt})
                            answer = response.get("answer", "No pude generar una respuesta.")
                            source_docs = response.get("source_documents", [])
                            source_text = "" if NO_INFO_PHRASE in answer else format_sources(source_docs)

                            if DEBUG_RAG:
                                elapsed = time.time() - start_time
                                logger.debug("Pregunta: %s", prompt[:80])
                                logger.debug("Chunks recuperados: %d", len(source_docs))
                                logger.debug("Tiempo de respuesta: %.2fs", elapsed)
                                for i, doc in enumerate(source_docs):
                                    logger.debug(
                                        "  Chunk %d (pág. %s): %s…",
                                        i + 1,
                                        doc.metadata.get("page", "?"),
                                        doc.page_content[:100],
                                    )

                            full_answer = answer + source_text
                            logger.info("Pregunta: %s | Respuesta generada.", prompt[:80])

                        except Exception as e:
                            full_answer = f"❌ Error al generar la respuesta: {e}"
                            logger.exception("Error al invocar la cadena RAG.")

                    st.markdown(full_answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_answer}
                )

# ── Tab 2: Voz ───────────────────────────────────────────────────────────────
with tab_voice:
    col_avatar, col_controls = st.columns([2, 3])

    with col_avatar:
        avatar_placeholder = st.empty()

    # Determinar si hay PDF y micrófono disponibles
    if not st.session_state.pdf_processed or st.session_state.chain is None:
        with avatar_placeholder:
            render_avatar("idle")
        with col_controls:
            _render_lock_screen("la interacción por voz")
    elif not MIC_AVAILABLE:
        with avatar_placeholder:
            render_avatar("idle")
        with col_controls:
            st.error(
                "❌ `streamlit-mic-recorder` no está instalado. "
                "Ejecuta: `pip install streamlit-mic-recorder`"
            )
    else:
        with avatar_placeholder:
            render_avatar("idle")

        with col_controls:
            # --- Grabación de audio ---
            st.markdown("**Paso 1 — Graba tu pregunta:**")
            audio = mic_recorder(
                start_prompt="🔴 Iniciar grabación",
                stop_prompt="⏹️ Detener grabación",
                key="voice_recorder",
            )

            if audio and audio.get("bytes"):
                with avatar_placeholder:
                    render_avatar("listening")

                with st.spinner("🔊 Transcribiendo audio con Whisper…"):
                    try:
                        with avatar_placeholder:
                            render_avatar("processing")
                        transcript = transcribe_audio(audio["bytes"])
                        st.session_state.voice_transcript = transcript
                        st.session_state["voice_transcript_area"] = transcript
                        st.session_state.voice_response = ""
                        st.session_state.voice_audio_bytes = None
                    except Exception as e:
                        st.error(f"❌ Error al transcribir: {e}")
                        with avatar_placeholder:
                            render_avatar("idle")

            # --- Mostrar transcripción ---
            if st.session_state.voice_transcript:
                st.markdown("**Paso 2 — Transcripción:**")
                transcript_text = st.text_area(
                    "Texto transcrito (puedes editarlo antes de enviar):",
                    value=st.session_state.voice_transcript,
                    key="voice_transcript_area",
                    height=80,
                )

                # --- Obtener respuesta RAG ---
                st.markdown("**Paso 3 — Respuesta del documento:**")
                if st.button("🤖 Obtener respuesta", use_container_width=True):
                    with avatar_placeholder:
                        render_avatar("processing")
                    with st.spinner("Buscando en el documento…"):
                        try:
                            resp = st.session_state.chain.invoke(
                                {"question": transcript_text}
                            )
                            source_docs = resp.get("source_documents", [])
                            raw_answer = resp.get("answer", "No pude generar una respuesta.")
                            rag_answer = raw_answer + (
                                "" if NO_INFO_PHRASE in raw_answer else format_sources(source_docs)
                            )
                            st.session_state.voice_response = rag_answer

                            if GTTS_AVAILABLE:
                                with avatar_placeholder:
                                    render_avatar("speaking")
                                try:
                                    clean_answer = raw_answer
                                    st.session_state.voice_audio_bytes = text_to_speech(clean_answer)
                                except Exception as tts_err:
                                    logger.warning("Error generando TTS: %s", tts_err)

                        except Exception as e:
                            st.session_state.voice_response = f"❌ Error: {e}"
                            logger.exception("Error en flujo de voz.")

                        with avatar_placeholder:
                            render_avatar("idle")

                if st.session_state.voice_response:
                    st.markdown(st.session_state.voice_response)

                    if st.session_state.voice_audio_bytes:
                        st.markdown("**Paso 4 — Escucha la respuesta:**")
                        st.audio(st.session_state.voice_audio_bytes, format="audio/mp3")
                    elif not GTTS_AVAILABLE:
                        st.info(
                            "💡 Instala `gTTS` para escuchar la respuesta en audio: "
                            "`pip install gTTS`"
                        )

# ── Tab 3: Grafo Semántico ────────────────────────────────────────────────────
with tab_graph:
    st.subheader("🕸️ Grafo Semántico")
    st.markdown(
        "El grafo extrae **entidades** (tecnologías, conceptos, procesos…) y sus "
        "**relaciones** del documento usando el LLM. Los nodos se colorean por tipo "
        "y las aristas representan la relación semántica entre ellos."
    )

    if not st.session_state.pdf_processed:
        _render_lock_screen("el grafo semántico")
    elif not GRAPH_AVAILABLE:
        st.error(
            "❌ `networkx` y/o `pyvis` no están instalados. "
            "Ejecuta: `pip install networkx pyvis`"
        )
    else:
        # Leyenda de colores
        with st.expander("🎨 Leyenda de colores por tipo de entidad"):
            cols = st.columns(len(ENTITY_COLORS))
            for col, (etype, color) in zip(cols, ENTITY_COLORS.items()):
                col.markdown(
                    f"<span style='background:{color};padding:3px 8px;"
                    f"border-radius:4px;color:white;font-size:12px'>{etype}</span>",
                    unsafe_allow_html=True,
                )

        # Botón para extraer/regenerar grafo
        btn_label = (
            "🔄 Regenerar grafo" if st.session_state.graph_html else "🕸️ Extraer grafo del documento"
        )
        if st.button(btn_label, use_container_width=True):
            # Necesitamos los chunks — los reconstruimos desde ChromaDB
            with st.spinner("🤖 Extrayendo entidades y relaciones con el LLM…"):
                try:
                    # Recuperar todos los documentos almacenados
                    raw_docs = st.session_state.vector_store._collection.get(
                        include=["documents", "metadatas"]
                    )

                    # Convertir a objetos Document para reusar extract_entities_llm
                
                    chunks_for_graph = [
                        Document(page_content=doc, metadata=meta)
                        for doc, meta in zip(
                            raw_docs["documents"], raw_docs["metadatas"]
                        )
                    ]

                    entities, relations = extract_entities_llm(chunks_for_graph)
                    st.session_state.graph_entities = entities
                    st.session_state.graph_relations = relations

                    if entities:
                        G = build_nx_graph(entities, relations)
                        st.session_state.graph_html = render_pyvis_html(G)
                    else:
                        st.warning("No se encontraron entidades en el documento.")

                except Exception as e:
                    st.error(f"❌ Error extrayendo grafo: {e}")
                    logger.exception("Error en extracción de grafo semántico.")

        # Mostrar métricas del grafo
        if st.session_state.graph_entities:
            c1, c2 = st.columns(2)
            c1.metric("🔵 Entidades", len(st.session_state.graph_entities))
            c2.metric("➡️ Relaciones", len(st.session_state.graph_relations))

            # Tabla de entidades
            with st.expander("📋 Lista de entidades extraídas"):
                st.dataframe(
                    st.session_state.graph_entities,
                    use_container_width=True,
                    hide_index=True,
                )

        # Renderizar grafo interactivo
        if st.session_state.graph_html:
            st.markdown("**Grafo interactivo** — arrastra los nodos para reorganizarlos:")
            components.html(st.session_state.graph_html, height=540, scrolling=False)

            # Exportar grafo
            st.markdown("**Exportar grafo:**")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                try:
                    import matplotlib  # noqa — verificar disponibilidad
                    G_export = build_nx_graph(
                        st.session_state.graph_entities,
                        st.session_state.graph_relations,
                    )
                    png_bytes = export_graph_png(G_export)
                    st.download_button(
                        "⬇️ Descargar PNG",
                        data=png_bytes,
                        file_name="grafo_semantico.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                except ImportError:
                    st.info("💡 Instala matplotlib para descargar PNG: `pip install matplotlib`")
                except Exception as e:
                    st.error(f"❌ Error generando PNG: {e}")
            with col_exp2:
                st.download_button(
                    "⬇️ Descargar HTML interactivo",
                    data=st.session_state.graph_html.encode("utf-8"),
                    file_name="grafo_semantico.html",
                    mime="text/html",
                    use_container_width=True,
                )