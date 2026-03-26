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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

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
SIMILARITY_THRESHOLD = 0.3  # Umbral mínimo de relevancia (0-1, mayor = más estricto)
NO_INFO_PHRASE = "No se encontró información suficiente"  # Frase que indica respuesta vacía

st.set_page_config(
    page_title="📄 RAG PDF Chat",
    page_icon="📄",
    layout="centered",
)

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
2. Si la respuesta no está en el contexto, di claramente:
   "No se encontró información suficiente en el documento."
3. No inventes información.
4. Sé preciso y conciso.
5. Cuando sea posible, menciona el criterio o dato clave del documento.
6. Mantén coherencia con el historial de conversación.
7. Prioriza exactitud sobre fluidez.

FORMATO:
- Respuesta clara en español
- Sin relleno innecesario

CONTEXTO DEL DOCUMENTO:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{question}"),
])

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


def load_and_split_pdf(pdf_path: str):
    """Lee un PDF y lo divide en chunks optimizados."""
    logger.info("Cargando PDF: %s", pdf_path)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    if not pages:
        raise ValueError("El PDF no contiene páginas legibles.")

    logger.info("Páginas encontradas: %d", len(pages))

    # Chunking optimizado: mayor tamaño + overlap + separadores semánticos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
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


def get_conversational_chain(vector_store):
    """Construye la cadena RAG conversacional con MMR y prompt grounded."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1024,
        openai_api_key=OPENAI_API_KEY,
    )

    # Memoria con ventana deslizante: solo últimos 6 turnos
    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Retriever MMR: mayor diversidad y cobertura
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    logger.info("Cadena RAG conversacional creada (MMR, k=6, grounded prompt).")
    return chain


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
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

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
            response = llm.invoke(prompt)
            # Limpiar posible markdown ```json ... ```
            raw = response.content.strip()
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
        bgcolor="#0E1117",
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
        st.session_state.pdf_processed = True

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
                        with st.spinner("📖 Leyendo y procesando el PDF…"):
                            # Guardar temporalmente el archivo subido
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".pdf"
                            ) as tmp:
                                tmp.write(file_bytes)
                                tmp_path = tmp.name

                            # Cargar, dividir y generar embeddings
                            pages, chunks = load_and_split_pdf(tmp_path)

                            # Analizar estadísticas (antes de borrar el temp)
                            pdf_stats = analyze_pdf_stats(file_bytes, pages)

                            # Limpiar archivo temporal
                            os.unlink(tmp_path)

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

    # Botón para limpiar conversación
    if st.session_state.pdf_processed:
        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state.chat_history = []
            # Recrear la cadena para limpiar la memoria interna
            if st.session_state.vector_store is not None:
                st.session_state.chain = get_conversational_chain(
                    st.session_state.vector_store
                )
            st.rerun()

    # Mostrar configuración de debug en sidebar
    if DEBUG_RAG:
        st.divider()
        
    st.caption("Desarrollado por NEXFLOW AI © 2025")
    st.caption("SAMUEL ARISTIZABAL BOTERO")
    st.caption("SANTIAGO CHAVARRO OSORIO")
    st.caption("SANTIAGO ANDRES GIRALDO GRANADA")

# ---------------------------------------------------------------------------
# 5. Área principal — Tabs: Chat / Voz / Grafo Semántico
# ---------------------------------------------------------------------------
st.title("📄 RAG PDF Chat")

tab_chat, tab_voice, tab_graph = st.tabs(["💬 Chat", "🎤 Voz", "🕸️ Grafo Semántico"])

# ── Tab 1: Chat ──────────────────────────────────────────────────────────────
with tab_chat:
    # Mostrar estadísticas si hay un PDF procesado
    if st.session_state.pdf_stats is not None:
        display_pdf_stats(st.session_state.pdf_stats)

    # Mostrar historial de chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Campo de entrada
    if prompt := st.chat_input("Escribe tu pregunta sobre el PDF…"):
        if not st.session_state.pdf_processed or st.session_state.chain is None:
            st.warning("⚠️ Primero sube y procesa un PDF desde la barra lateral.")
        elif not prompt.strip():
            st.warning("⚠️ Escribe una pregunta válida.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Pensando…"):
                    try:
                        start_time = time.time()
                        is_relevant, scores = check_relevance(
                            st.session_state.vector_store, prompt
                        )

                        if not is_relevant:
                            answer = (
                                "No se encontró información suficiente "
                                "en el documento para responder esta pregunta."
                            )
                            source_text = ""
                            if DEBUG_RAG:
                                logger.debug(
                                    "Pregunta rechazada. Scores: %s",
                                    [f"{s:.3f}" for s in scores],
                                )
                        else:
                            response = st.session_state.chain.invoke({"question": prompt})
                            answer = response.get("answer", "No pude generar una respuesta.")
                            source_docs = response.get("source_documents", [])
                            source_text = "" if NO_INFO_PHRASE in answer else format_sources(source_docs)

                            if DEBUG_RAG:
                                elapsed = time.time() - start_time
                                logger.debug("Pregunta: %s", prompt[:80])
                                logger.debug("Chunks recuperados: %d", len(source_docs))
                                logger.debug("Scores: %s", [f"{s:.3f}" for s in scores])
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
    st.subheader("🎤 Interacción por Voz")
    st.markdown(
        "Graba tu pregunta con el micrófono. El sistema la transcribirá con "
        "**Whisper** y responderá usando el documento cargado. "
        "La respuesta también se puede escuchar en audio."
    )

    if not st.session_state.pdf_processed:
        st.warning("⚠️ Primero sube y procesa un PDF desde la barra lateral.")
    elif not MIC_AVAILABLE:
        st.error(
            "❌ `streamlit-mic-recorder` no está instalado. "
            "Ejecuta: `pip install streamlit-mic-recorder`"
        )
    else:
        # --- Grabación de audio ---
        st.markdown("**Paso 1 — Graba tu pregunta:**")
        audio = mic_recorder(
            start_prompt="🔴 Iniciar grabación",
            stop_prompt="⏹️ Detener grabación",
            key="voice_recorder",
        )

        if audio and audio.get("bytes"):
            with st.spinner("🔊 Transcribiendo audio con Whisper…"):
                try:
                    transcript = transcribe_audio(audio["bytes"])
                    st.session_state.voice_transcript = transcript
                    st.session_state.voice_response = ""
                    st.session_state.voice_audio_bytes = None
                except Exception as e:
                    st.error(f"❌ Error al transcribir: {e}")

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
                with st.spinner("Buscando en el documento…"):
                    try:
                        # Invocar la cadena directamente — el system prompt grounded
                        # maneja el caso de contexto irrelevante sin necesidad de
                        # check_relevance(), cuyo threshold numérico (0.3) rechaza
                        # consultas de voz con vocabulario natural ya que la
                        # transcripción de Whisper produce distancias L2 más altas.
                        resp = st.session_state.chain.invoke(
                            {"question": transcript_text}
                        )
                        source_docs = resp.get("source_documents", [])
                        raw_answer = resp.get("answer", "No pude generar una respuesta.")
                        rag_answer = raw_answer + (
                            "" if NO_INFO_PHRASE in raw_answer else format_sources(source_docs)
                        )

                        st.session_state.voice_response = rag_answer

                        # Generar audio de respuesta (solo texto, sin citas de página)
                        if GTTS_AVAILABLE:
                            try:
                                clean_answer = raw_answer
                                st.session_state.voice_audio_bytes = text_to_speech(clean_answer)
                            except Exception as tts_err:
                                logger.warning("Error generando TTS: %s", tts_err)

                    except Exception as e:
                        st.session_state.voice_response = f"❌ Error: {e}"
                        logger.exception("Error en flujo de voz.")

            if st.session_state.voice_response:
                st.markdown(st.session_state.voice_response)

                # --- Reproducir respuesta en audio ---
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
        st.warning("⚠️ Primero sube y procesa un PDF desde la barra lateral.")
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
                    from langchain.schema import Document
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