# 📄 RAG PDF Chat

Aplicación local para conversar con documentos PDF usando **Retrieval-Augmented Generation (RAG)** con OpenAI.

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

La aplicación se abrirá en tu navegador en `http://localhost:8501`.

---

## 🧩 ¿Cómo funciona?

1. **Sube un PDF** desde la barra lateral.
2. **Haz clic en "Procesar PDF"** para generar embeddings (solo se generan una vez).
3. **Escribe tu pregunta** en el campo de chat.
4. El sistema recupera los fragmentos más relevantes del PDF y genera una respuesta con GPT-4o-mini.
5. La conversación mantiene memoria — puedes hacer preguntas de seguimiento.

## 🛠️ Stack tecnológico

| Componente     | Tecnología             |
| -------------- | ---------------------- |
| Interfaz       | Streamlit              |
| Orquestación   | LangChain              |
| LLM            | OpenAI (gpt-4o-mini)   |
| Embeddings     | text-embedding-3-small |
| Base vectorial | ChromaDB               |
| Lectura de PDF | PyPDF                  |

## 📁 Estructura del proyecto

```
rag-pdf-chat/
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias con versiones fijadas
├── .env.example        # Plantilla de configuración
├── README.md           # Este archivo
└── data/               # Almacenamiento persistente de ChromaDB
```
