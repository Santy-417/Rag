# 📄 RAG PDF Chat

Aplicación para conversar con documentos PDF usando **Retrieval-Augmented Generation (RAG)** con OpenAI. Incluye interacción por **voz** (STT + TTS) y visualización de **grafos semánticos**.

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

1. **Sube un PDF** desde la barra lateral y haz clic en **"Procesar PDF"** (los embeddings se generan una sola vez y persisten en disco).
2. Usa cualquiera de las tres pestañas:

| Pestaña | Descripción |
|---|---|
| 💬 **Chat** | Escribe preguntas sobre el documento. Muestra fuentes con número de página. |
| 🎤 **Voz** | Graba tu pregunta, Whisper la transcribe, el sistema responde y genera audio. |
| 🕸️ **Grafo Semántico** | Extrae entidades y relaciones del PDF con el LLM y las visualiza como grafo interactivo. |

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|---|---|
| Interfaz | Streamlit |
| Orquestación | LangChain |
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Base vectorial | ChromaDB (persistente en `data/chroma/`) |
| Lectura de PDF | PyPDF + PyMuPDF |
| Voz STT | OpenAI Whisper API |
| Voz TTS | gTTS |
| Grafo | NetworkX + PyVis |

## 📁 Estructura del proyecto

```
rag-pdf-chat/
├── app.py              # Aplicación principal (única fuente de código)
├── requirements.txt    # Dependencias
├── .env.example        # Plantilla de variables de entorno
├── CLAUDE.md           # Guía para Claude Code
├── README.md           # Este archivo
└── data/               # ChromaDB local (generado automáticamente, no se sube a git)
```

## ☁️ Despliegue en Streamlit Community Cloud

1. Sube el proyecto a GitHub (el `.gitignore` ya excluye `.env` y `data/`).
2. Ve a [share.streamlit.io](https://share.streamlit.io) → **New app** → conecta el repo → `app.py`.
3. En **Advanced settings → Secrets** agrega:
   ```toml
   OPENAI_API_KEY = "sk-proj-tu-clave-aqui"
   DEBUG_RAG = "false"
   ```
4. Deploy. El PDF debe subirse en cada sesión (el almacenamiento es efímero en Streamlit Cloud).
