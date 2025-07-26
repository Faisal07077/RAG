import asyncio
from typing import Dict, Any, List
from agents.mcp import MCPMessage, MCPMessageTypes, create_response_message, create_error_message
from utils.document_parsers import DocumentParser
import uuid

class IngestionAgent:
    """
    Agent responsible for parsing and preprocessing documents
    """
    
    def __init__(self):
        self.name = "IngestionAgent"
        self.document_parser = DocumentParser()
        self.processed_documents: Dict[str, Dict] = {}
    
    async def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP messages"""
        try:
            if message.type == MCPMessageTypes.DOCUMENT_UPLOAD:
                return await self._process_document(message)
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
    
    async def _process_document(self, message: MCPMessage) -> MCPMessage:
        """Process uploaded document"""
        try:
            file_name = message.payload.get("file_name")
            file_content = message.payload.get("file_content")
            file_type = message.payload.get("file_type")
            
            if not all([file_name, file_content]):
                return create_error_message(
                    message,
                    self.name,
                    "Missing required fields: file_name or file_content"
                )
            
            # Parse document content
            parsed_content = await self._parse_document(file_name, file_content, file_type)
            
            # Store processed document
            doc_id = str(uuid.uuid4())
            self.processed_documents[doc_id] = {
                "id": doc_id,
                "name": file_name,
                "type": file_type,
                "content": parsed_content["text"],
                "chunks": parsed_content["chunks"],
                "metadata": parsed_content["metadata"]
            }
            
            return create_response_message(
                message,
                self.name,
                MCPMessageTypes.DOCUMENT_PARSED,
                {
                    "status": "success",
                    "document_id": doc_id,
                    "file_name": file_name,
                    "chunks": parsed_content["chunks"],
                    "metadata": parsed_content["metadata"],
                    "total_chunks": len(parsed_content["chunks"])
                }
            )
            
        except Exception as e:
            return create_error_message(
                message,
                self.name,
                f"Document processing failed: {str(e)}"
            )
    
    async def _parse_document(self, file_name: str, file_content: bytes, file_type: str) -> Dict[str, Any]:
        """Parse document based on file type"""
        try:
            # Determine file extension
            if file_name.lower().endswith('.pdf'):
                parser_type = 'pdf'
            elif file_name.lower().endswith('.pptx'):
                parser_type = 'pptx'
            elif file_name.lower().endswith('.docx'):
                parser_type = 'docx'
            elif file_name.lower().endswith('.csv'):
                parser_type = 'csv'
            elif file_name.lower().endswith(('.txt', '.md')):
                parser_type = 'text'
            else:
                raise ValueError(f"Unsupported file type: {file_name}")
            
            # Parse document
            parsed_data = self.document_parser.parse(file_content, parser_type)
            
            # Chunk the text
            chunks = self._chunk_text(parsed_data["text"], file_name)
            
            return {
                "text": parsed_data["text"],
                "chunks": chunks,
                "metadata": {
                    "file_name": file_name,
                    "file_type": file_type,
                    "parser_type": parser_type,
                    "page_count": parsed_data.get("page_count", 1),
                    "word_count": len(parsed_data["text"].split()),
                    **parsed_data.get("metadata", {})
                }
            }
            
        except Exception as e:
            raise Exception(f"Failed to parse document {file_name}: {str(e)}")
    
    def _chunk_text(self, text: str, source_file: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk = {
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "source_file": source_file,
                "chunk_index": len(chunks),
                "word_count": len(chunk_words),
                "start_word": i,
                "end_word": min(i + chunk_size, len(words))
            }
            chunks.append(chunk)
            
            # Break if we've processed all words
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def get_processed_documents(self) -> Dict[str, Dict]:
        """Get all processed documents"""
        return self.processed_documents
    
    def get_document(self, doc_id: str) -> Dict:
        """Get specific document by ID"""
        return self.processed_documents.get(doc_id)
    
    def clear_documents(self) -> None:
        """Clear all processed documents"""
        self.processed_documents.clear()
