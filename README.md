# 📄 RAG PDF Chat

Aplicación para conversar con documentos PDF usando **Retrieval-Augmented Generation (RAG)** con OpenAI. Incluye interacción por **voz** (STT + TTS), visualización de **grafos semánticos** y una interfaz dark con avatar animado.

## 🚀 Inicio rápido

### 1. Crear entorno virtual

**Linux / Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar API key

```bash
cp .env.example .env
```

Abre `.env` y reemplaza el valor de `OPENAI_API_KEY` con tu clave real de OpenAI.

### 4. Ejecutar la app

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

---

## 🧩 ¿Cómo funciona?

1. **Sube un PDF** desde la barra lateral → haz clic en **"🚀 Procesar PDF"**.
2. Los embeddings se generan con `text-embedding-3-small` y se persisten en ChromaDB.
3. Usa cualquiera de las tres pestañas (bloqueadas hasta procesar el PDF):

| Pestaña | Descripción |
|---|---|
| 💬 **Chat** | Escribe preguntas sobre el documento. Muestra estadísticas del PDF y fuentes con número de página. |
| 🎤 **Voz** | Graba tu pregunta → Whisper la transcribe → el sistema responde y genera audio TTS. Avatar robot animado con estados (idle / escuchando / procesando / hablando). |
| 🕸️ **Grafo Semántico** | Extrae entidades y relaciones del PDF con el LLM → visualización interactiva PyVis → exporta como PNG o HTML. |

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|---|---|
| Interfaz | Streamlit + CSS personalizado (dark theme) |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Base vectorial | ChromaDB (persistente en `data/chroma/`) |
| Retrieval | MMR k=12, fetch_k=50, lambda=0.5 |
| Lectura de PDF | PyMuPDF (`fitz`) + PyPDF (fallback) |
| Chunking | RecursiveCharacterTextSplitter (1500 chars, 300 overlap) |
| Voz STT | OpenAI Whisper API (`whisper-1`) |
| Voz TTS | gTTS (Google Text-to-Speech) |
| Avatar | SVG animado personalizado (`voice_avatar.py`) |
| Grafo | NetworkX + PyVis + matplotlib |

---

## 📁 Estructura del proyecto

```
rag-pdf-chat/
├── app.py                  # Aplicación principal (RAG + UI)
├── voice_avatar.py         # Avatar robot SVG animado para tab de voz
├── requirements.txt        # Dependencias Python
├── .env.example            # Plantilla de variables de entorno
├── .streamlit/
│   └── config.toml         # Configuración de tema dark de Streamlit
├── CLAUDE.md               # Guía de arquitectura para Claude Code
├── README.md               # Este archivo
└── data/                   # ChromaDB local (auto-generado, excluido de git)
```

---

## ⚙️ Pipeline RAG

```
PDF
 └─► fitz.get_text() por página
      └─► RecursiveCharacterTextSplitter (1500 chars, 300 overlap)
           └─► text-embedding-3-small → ChromaDB
                └─► MMR Retriever (k=12, diversidad=0.5)
                     └─► gpt-4o-mini + system prompt grounded
                          └─► Respuesta con fuentes (número de página)
```

**Por qué MMR:** preguntas de agregación como "¿qué países menciona el documento?" se benefician de la diversificación forzada de MMR — después del chunk más relevante, busca chunks distintos en otras secciones del documento, cubriendo más información dispersa.

---

## ☁️ Despliegue en Streamlit Community Cloud

1. Sube el proyecto a GitHub (el `.gitignore` ya excluye `.env` y `data/`).
2. Ve a [share.streamlit.io](https://share.streamlit.io) → **New app** → conecta el repo → `app.py`.
3. En **Advanced settings → Secrets** agrega:
   ```toml
   OPENAI_API_KEY = "sk-proj-tu-clave-aqui"
   DEBUG_RAG = "false"
   ```
4. Deploy. El PDF debe subirse en cada sesión (almacenamiento efímero en Streamlit Cloud).

> **Nota:** `.streamlit/config.toml` debe estar commiteado para que el tema dark aplique en Cloud.

---

## 🔧 Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `OPENAI_API_KEY` | Clave de API de OpenAI (requerida) | — |
| `DEBUG_RAG` | Activa logs detallados en terminal | `true` |
