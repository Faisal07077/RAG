import asyncio
from typing import Dict, Any, List
from agents.mcp import MCPMessage, MCPMessageTypes, create_response_message, create_error_message, MCPRouter
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent

class CoordinatorAgent:
    """
    Coordinator agent that orchestrates communication between other agents
    """
    
    def __init__(self):
        self.name = "CoordinatorAgent"
        self.router = MCPRouter()
        
        # Initialize sub-agents
        self.ingestion_agent = IngestionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.llm_response_agent = LLMResponseAgent()
        
        # Agent registry
        self.agents = {
            "IngestionAgent": self.ingestion_agent,
            "RetrievalAgent": self.retrieval_agent,
            "LLMResponseAgent": self.llm_response_agent
        }
    
    async def handle_message(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming messages and coordinate agent interactions"""
        try:
            # Route message through MCP router
            self.router.route_message(message)
            
            if message.type == MCPMessageTypes.DOCUMENT_UPLOAD:
                return await self._handle_document_upload(message)
            elif message.type == MCPMessageTypes.QUERY_REQUEST:
                return await self._handle_query_request(message)
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
                f"Coordination error: {str(e)}"
            )
    
    async def _handle_document_upload(self, message: MCPMessage) -> MCPMessage:
        """Handle document upload workflow"""
        try:
            # Step 1: Send to Ingestion Agent
            ingestion_response = await self.ingestion_agent.handle_message(message)
            self.router.route_message(ingestion_response)
            
            if ingestion_response.type == MCPMessageTypes.ERROR:
                return create_response_message(
                    message,
                    self.name,
                    MCPMessageTypes.ERROR,
                    {
                        "status": "failed",
                        "error": f"Ingestion failed: {ingestion_response.payload.get('error')}"
                    }
                )
            
            # Step 2: Send to Retrieval Agent for indexing
            retrieval_response = await self.retrieval_agent.handle_message(ingestion_response)
            self.router.route_message(retrieval_response)
            
            if retrieval_response.type == MCPMessageTypes.ERROR:
                return create_response_message(
                    message,
                    self.name,
                    MCPMessageTypes.ERROR,
                    {
                        "status": "failed",
                        "error": f"Indexing failed: {retrieval_response.payload.get('error')}"
                    }
                )
            
            # Return success response
            return create_response_message(
                message,
                self.name,
                MCPMessageTypes.SUCCESS,
                {
                    "status": "success",
                    "document_id": ingestion_response.payload.get("document_id"),
                    "file_name": ingestion_response.payload.get("file_name"),
                    "indexed_chunks": retrieval_response.payload.get("indexed_chunks"),
                    "workflow": "document_upload_complete"
                }
            )
            
        except Exception as e:
            return create_error_message(
                message,
                self.name,
                f"Document upload workflow failed: {str(e)}"
            )
    
    async def _handle_query_request(self, message: MCPMessage) -> MCPMessage:
        """Handle query processing workflow"""
        try:
            query = message.payload.get("query")
            conversation_history = message.payload.get("conversation_history", [])
            
            # Step 1: Send retrieval request
            retrieval_message = MCPMessage(
                sender=self.name,
                receiver="RetrievalAgent",
                type=MCPMessageTypes.RETRIEVAL_REQUEST,
                trace_id=message.trace_id,
                payload={
                    "query": query,
                    "top_k": 5
                }
            )
            
            retrieval_response = await self.retrieval_agent.handle_message(retrieval_message)
            self.router.route_message(retrieval_response)
            
            if retrieval_response.type == MCPMessageTypes.ERROR:
                return create_response_message(
                    message,
                    self.name,
                    MCPMessageTypes.ERROR,
                    {
                        "status": "failed",
                        "error": f"Retrieval failed: {retrieval_response.payload.get('error')}"
                    }
                )
            
            # Step 2: Send to LLM Response Agent
            llm_message = MCPMessage(
                sender=self.name,
                receiver="LLMResponseAgent",
                type=MCPMessageTypes.RETRIEVAL_RESULT,
                trace_id=message.trace_id,
                payload={
                    **retrieval_response.payload,
                    "conversation_history": conversation_history
                }
            )
            
            llm_response = await self.llm_response_agent.handle_message(llm_message)
            self.router.route_message(llm_response)
            
            if llm_response.type == MCPMessageTypes.ERROR:
                return create_response_message(
                    message,
                    self.name,
                    MCPMessageTypes.ERROR,
                    {
                        "status": "failed",
                        "error": f"LLM response failed: {llm_response.payload.get('error')}"
                    }
                )
            
            # Return successful response
            return create_response_message(
                message,
                self.name,
                MCPMessageTypes.SUCCESS,
                {
                    "status": "success",
                    "query": query,
                    "response": llm_response.payload.get("response"),
                    "sources": llm_response.payload.get("sources", []),
                    "context_chunks_used": llm_response.payload.get("context_chunks_used", 0),
                    "workflow": "query_processing_complete"
                }
            )
            
        except Exception as e:
            return create_error_message(
                message,
                self.name,
                f"Query processing workflow failed: {str(e)}"
            )
    
    def get_message_history(self) -> List[MCPMessage]:
        """Get message routing history"""
        return self.router.message_history
    
    def get_trace_history(self, trace_id: str) -> List[MCPMessage]:
        """Get messages for specific trace"""
        return self.router.get_trace_history(trace_id)
    
    def clear_documents(self) -> None:
        """Clear all documents from all agents"""
        self.ingestion_agent.clear_documents()
        self.retrieval_agent.clear_index()
        self.router.clear_history()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "coordinator": "active",
            "processed_documents": len(self.ingestion_agent.get_processed_documents()),
            "indexed_chunks": self.retrieval_agent.get_indexed_count(),
            "total_messages": len(self.router.message_history),
            "agents": {
                "ingestion": "active",
                "retrieval": "active", 
                "llm_response": "active"
            }
        }
