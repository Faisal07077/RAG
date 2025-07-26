import asyncio
from typing import Dict, Any, List, Tuple
from agents.mcp import MCPMessage, MCPMessageTypes, create_response_message, create_error_message
from utils.vector_store import VectorStore
from utils.local_embeddings import LocalEmbeddingGenerator
import numpy as np

class RetrievalAgent:
    """
    Agent responsible for embedding generation and semantic retrieval
    """
    
    def __init__(self):
        self.name = "RetrievalAgent"
        self.vector_store = VectorStore()
        self.embedding_generator = LocalEmbeddingGenerator()
        self.document_chunks: Dict[str, Dict] = {}
    
    async def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP messages"""
        try:
            if message.type == MCPMessageTypes.DOCUMENT_PARSED:
                return await self._index_document(message)
            elif message.type == MCPMessageTypes.RETRIEVAL_REQUEST:
                return await self._retrieve_context(message)
            else:
                return create_error_message(
                    message,
                    self.name,
                    f"Unsupported message type: {message.type}"
                )
        except Exception as e:
            return create_error_message(
                message,
                self.name,
                f"Error processing message: {str(e)}"
            )
    
    async def _index_document(self, message: MCPMessage) -> MCPMessage:
        """Index document chunks in vector store"""
        try:
            document_id = message.payload.get("document_id")
            file_name = message.payload.get("file_name")
            chunks = message.payload.get("chunks", [])
            metadata = message.payload.get("metadata", {})
            
            if not chunks:
                return create_error_message(
                    message,
                    self.name,
                    "No chunks provided for indexing"
                )
            
            # Generate embeddings for chunks
            indexed_chunks = []
            for chunk in chunks:
                try:
                    # Generate embedding
                    embedding = await self.embedding_generator.generate_embedding(chunk["text"])
                    
                    # Add to vector store
                    chunk_id = chunk["id"]
                    self.vector_store.add_vector(
                        chunk_id,
                        embedding,
                        {
                            **chunk,
                            "document_id": document_id,
                            "document_name": file_name,
                            **metadata
                        }
                    )
                    
                    # Store chunk reference
                    self.document_chunks[chunk_id] = {
                        **chunk,
                        "document_id": document_id,
                        "document_name": file_name,
                        "embedding_dims": len(embedding)
                    }
                    
                    indexed_chunks.append(chunk_id)
                    
                except Exception as e:
                    print(f"Warning: Failed to index chunk {chunk.get('id', 'unknown')}: {str(e)}")
                    continue
            
            return create_response_message(
                message,
                self.name,
                MCPMessageTypes.INGESTION_COMPLETE,
                {
                    "status": "success",
                    "document_id": document_id,
                    "file_name": file_name,
                    "indexed_chunks": len(indexed_chunks),
                    "total_chunks": len(chunks)
                }
            )
            
        except Exception as e:
            return create_error_message(
                message,
                self.name,
                f"Document indexing failed: {str(e)}"
            )
    
    async def _retrieve_context(self, message: MCPMessage) -> MCPMessage:
        """Retrieve relevant context for query"""
        try:
            query = message.payload.get("query")
            top_k = message.payload.get("top_k", 5)
            
            if not query:
                return create_error_message(
                    message,
                    self.name,
                    "No query provided for retrieval"
                )
            
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Search vector store
            search_results = self.vector_store.search(query_embedding, top_k)
            
            # Format retrieved chunks
            retrieved_chunks = []
            sources = []
            
            for chunk_id, similarity_score in search_results:
                if chunk_id in self.document_chunks:
                    chunk_data = self.document_chunks[chunk_id]
                    retrieved_chunks.append({
                        "id": chunk_id,
                        "text": chunk_data["text"],
                        "source_file": chunk_data.get("document_name", "Unknown"),
                        "similarity_score": float(similarity_score),
                        "chunk_index": chunk_data.get("chunk_index", 0)
                    })
                    
                    # Add source reference
                    source_ref = f"{chunk_data.get('document_name', 'Unknown')} (chunk {chunk_data.get('chunk_index', 0)})"
                    if source_ref not in sources:
                        sources.append(source_ref)
            
            return create_response_message(
                message,
                self.name,
                MCPMessageTypes.RETRIEVAL_RESULT,
                {
                    "status": "success",
                    "query": query,
                    "retrieved_chunks": retrieved_chunks,
                    "sources": sources,
                    "total_results": len(retrieved_chunks)
                }
            )
            
        except Exception as e:
            return create_error_message(
                message,
                self.name,
                f"Context retrieval failed: {str(e)}"
            )
    
    def get_indexed_count(self) -> int:
        """Get number of indexed chunks"""
        return len(self.document_chunks)
    
    def clear_index(self) -> None:
        """Clear all indexed documents"""
        self.vector_store.clear()
        self.document_chunks.clear()
