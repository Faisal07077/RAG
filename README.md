# 🧠 Agentic RAG Chatbot (DocuChatAI)

A document-based chatbot that uses a **multi-agent architecture** with **Model Context Protocol (MCP)** for intelligent, document-driven question answering — **no OpenAI API required**.

---

## 🚀 Features

- Multi-agent architecture using 4 specialized agents
- Supports multiple document types: PDF, PPTX, DOCX, CSV, TXT, MD
- Uses local TF-IDF embeddings (no external API)
- Built-in FAISS vector store for semantic document search
- Interactive web UI using Streamlit
- Template-based intelligent response generation

---

## 🛠️ Setup Instructions

### 🔧 Prerequisites
Make sure you have the following installed:
- Python 3.9 or above
- pip (Python package manager)

### 📦 Step-by-Step

1. **Clone the repository** (or download ZIP):
```bash
git clone https://github.com/Faisal07077/RAG.git
cd RAG
pip install -r requirements.txt
streamlit run app.py
