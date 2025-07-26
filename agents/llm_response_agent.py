import asyncio
import os
from typing import Dict, Any, List
from agents.mcp import MCPMessage, MCPMessageTypes, create_response_message, create_error_message
from utils.local_llm import LocalLLMGenerator

class LLMResponseAgent:
    """
    Agent responsible for generating final responses using LLM
    """
    
    def __init__(self):
        self.name = "LLMResponseAgent"
        self.model = "Local Template LLM"
        self.llm_generator = LocalLLMGenerator()
    
    async def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP messages"""
        try:
            if message.type == MCPMessageTypes.RETRIEVAL_RESULT:
                return await self._generate_response(message)
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
    
    async def _generate_response(self, message: MCPMessage) -> MCPMessage:
        """Generate response using retrieved context"""
        try:
            query = message.payload.get("query")
            retrieved_chunks = message.payload.get("retrieved_chunks", [])
            sources = message.payload.get("sources", [])
            conversation_history = message.payload.get("conversation_history", [])
            
            if not query:
                return create_error_message(
                    message,
                    self.name,
                    "No query provided for response generation"
                )
            
            # Generate response using local LLM
            response = await self.llm_generator.generate_response(
                query, 
                retrieved_chunks, 
                conversation_history
            )
            
            return create_response_message(
                message,
                self.name,
                MCPMessageTypes.LLM_RESPONSE,
                {
                    "status": "success",
                    "query": query,
                    "response": response,
                    "sources": sources,
                    "context_chunks_used": len(retrieved_chunks)
                }
            )
            
        except Exception as e:
            return create_error_message(
                message,
                self.name,
                f"Response generation failed: {str(e)}"
            )
    
    def set_model(self, model: str) -> None:
        """Set the LLM model to use"""
        self.model = model
