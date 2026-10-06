# 📖 Quranic AI Assistant Using RAG

An AI-powered Question-Answering application that applies Retrieval-Augmented Generation (RAG) to an English translation and commentary of the Holy Quran (Surah Al-Fatihah). Built with Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Groq.

---

## 🌟 Features

- **Document Ingestion**: Allows users to upload any English Quranic Surah PDF directly via the web interface.
- **Robust Text Extraction**: Uses `pdfplumber` and `pypdf` with fallback commentary context to ensure reliable retrieval.
- **Semantic Vector Search**: Generates embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` and indexes them in an in-memory FAISS vector database.
- **Fast LLM Responses**: Leverages Groq's high-speed inference engine (`qwen/qwen3.8-27b`) for context-grounded, respectful answers.
- **Transparent Source Chunks**: Expandable section displaying the exact retrieved verses and commentary used for each response.
- **Secure Configuration**: Supports Streamlit Secrets for cloud deployment and a sidebar input fallback for API keys.

---

## 🛠️ Tech Stack

- **UI / Web Framework**: Streamlit
- **RAG Framework**: LangChain (`langchain`, `langchain-community`, `langchain-text-splitters`)
- **Embeddings**: Hugging Face (`langchain-huggingface`, `sentence-transformers`)
- **Vector Database**: FAISS (`faiss-cpu`)
- **LLM Provider**: Groq API (`langchain-groq`)
- **PDF Processing**: `pdfplumber`, `pypdf`

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── secrets.toml          # Groq API key configuration (local only)
├── app.py                    # Main Streamlit and RAG pipeline script
├── surah.pdf                 # Sample Quranic Surah PDF in English
├── requirements.txt          # Python project dependencies
├── .gitignore                # Prevents secrets and virtual env from being tracked
└── README.md                 # Project documentation and guide
