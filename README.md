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

## 🧠 Agent Roles

- **Ingestion Agent** → Parses and chunks uploaded files
- **Retrieval Agent** → Performs semantic search with FAISS
- **LLM Response Agent** → Generates answers based on retrieved context
- **Coordinator Agent** → Controls agent interactions via MCP

---

