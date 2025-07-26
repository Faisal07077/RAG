# Multi-Agent RAG Chatbot System

## Overview

This repository contains a sophisticated Retrieval-Augmented Generation (RAG) chatbot system built with Streamlit and a multi-agent architecture. The system uses the Model Context Protocol (MCP) for inter-agent communication and supports multiple document formats including PDF, PPTX, DOCX, CSV, TXT, and Markdown files.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a multi-agent architecture pattern with specialized agents for different tasks:

1. **Coordinator Agent**: Orchestrates communication between all other agents
2. **Ingestion Agent**: Handles document parsing and preprocessing
3. **Retrieval Agent**: Manages embedding generation and semantic search
4. **LLM Response Agent**: Generates final responses using OpenAI's GPT-4o model

The system uses the Model Context Protocol (MCP) for structured communication between agents, ensuring reliable message passing and error handling throughout the pipeline.

## Key Components

### Frontend (Streamlit)
- **Main Interface**: Simple chat interface with document upload sidebar
- **Session Management**: Maintains conversation history and uploaded documents
- **File Upload**: Supports multiple document formats with real-time processing feedback

### Backend Agents
- **CoordinatorAgent**: Central orchestrator that routes messages between specialized agents
- **IngestionAgent**: Parses documents using format-specific parsers and chunks text for processing
- **RetrievalAgent**: Generates embeddings using OpenAI's text-embedding-3-small model and performs semantic search
- **LLMResponseAgent**: Uses GPT-4o for generating contextual responses based on retrieved information

### Utility Components
- **DocumentParser**: Unified parser supporting PDF (via pdfplumber), PPTX, DOCX, CSV, and text formats
- **EmbeddingGenerator**: OpenAI embedding service with rate limiting and batch processing
- **VectorStore**: FAISS-based vector database for efficient similarity search

## Data Flow

1. **Document Upload**: User uploads documents through Streamlit interface
2. **Document Processing**: CoordinatorAgent routes to IngestionAgent for parsing and chunking
3. **Embedding Generation**: RetrievalAgent generates embeddings and stores in FAISS vector store
4. **Query Processing**: User queries trigger semantic search across indexed documents
5. **Response Generation**: LLMResponseAgent combines retrieved context with conversation history to generate responses

## External Dependencies

### Required Services
- **OpenAI API**: Used for both embedding generation (text-embedding-3-small) and response generation (GPT-4o)
- Requires `OPENAI_API_KEY` environment variable

### Python Libraries
- **Streamlit**: Web interface framework
- **FAISS**: Vector similarity search
- **OpenAI**: API client for embeddings and completions
- **Document Processing**: PyPDF2, pdfplumber, python-pptx, python-docx, pandas
- **Data Processing**: NumPy for vector operations

## Deployment Strategy

The application is designed as a single-file Streamlit application (`app.py`) with modular agent architecture. Key deployment considerations:

1. **Environment Variables**: Requires OpenAI API key configuration
2. **Dependencies**: All required packages should be installed via pip
3. **Memory Management**: FAISS index and document storage are in-memory (suitable for development/small-scale deployment)
4. **Scalability**: Architecture supports future enhancements like persistent storage and distributed processing

### Development Setup
- Run with `streamlit run app.py`
- Ensure OpenAI API key is set in environment
- The system uses session state for maintaining conversation context and uploaded documents

### Architecture Benefits
- **Modularity**: Each agent has a specific responsibility, making the system maintainable
- **Extensibility**: New agents can be easily added to the coordinator's registry
- **Error Handling**: MCP provides structured error handling across all agent communications
- **Async Support**: Built for asynchronous processing of document ingestion and retrieval operations