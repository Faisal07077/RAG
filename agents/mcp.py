from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
import json
import uuid
from datetime import datetime

@dataclass
class MCPMessage:
    """
    Model Context Protocol message structure for agent communication
    """
    sender: str
    receiver: str
    type: str
    trace_id: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Create message from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

class MCPMessageTypes:
    """Standard MCP message types"""
    # Document processing
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    DOCUMENT_PARSED = "DOCUMENT_PARSED"
    INGESTION_COMPLETE = "INGESTION_COMPLETE"
    
    # Query processing
    QUERY_REQUEST = "QUERY_REQUEST"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESULT = "RETRIEVAL_RESULT"
    CONTEXT_RESPONSE = "CONTEXT_RESPONSE"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"
    
    # System messages
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    STATUS_UPDATE = "STATUS_UPDATE"

class MCPRouter:
    """Message router for MCP communication"""
    
    def __init__(self):
        self.message_history: List[MCPMessage] = []
        self.trace_sessions: Dict[str, List[MCPMessage]] = {}
    
    def route_message(self, message: MCPMessage) -> None:
        """Route message and store in history"""
        self.message_history.append(message)
        
        # Group by trace_id for session tracking
        if message.trace_id not in self.trace_sessions:
            self.trace_sessions[message.trace_id] = []
        self.trace_sessions[message.trace_id].append(message)
    
    def get_trace_history(self, trace_id: str) -> List[MCPMessage]:
        """Get all messages for a specific trace"""
        return self.trace_sessions.get(trace_id, [])
    
    def get_recent_messages(self, limit: int = 10) -> List[MCPMessage]:
        """Get recent messages"""
        return self.message_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear message history"""
        self.message_history.clear()
        self.trace_sessions.clear()

def create_response_message(
    original_message: MCPMessage,
    sender: str,
    message_type: str,
    payload: Dict[str, Any]
) -> MCPMessage:
    """Helper function to create response messages"""
    return MCPMessage(
        sender=sender,
        receiver=original_message.sender,
        type=message_type,
        trace_id=original_message.trace_id,
        payload=payload
    )

def create_error_message(
    original_message: MCPMessage,
    sender: str,
    error: str
) -> MCPMessage:
    """Helper function to create error messages"""
    return create_response_message(
        original_message,
        sender,
        MCPMessageTypes.ERROR,
        {"error": error, "original_type": original_message.type}
    )
