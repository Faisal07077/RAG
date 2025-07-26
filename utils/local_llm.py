import asyncio
from typing import Dict, Any, List
import re

class LocalLLMGenerator:
    """
    Local LLM replacement using template-based responses
    """
    
    def __init__(self):
        self.name = "Local Template LLM"
    
    async def generate_response(
        self, 
        query: str, 
        context_chunks: List[Dict[str, Any]], 
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """Generate response using retrieved context"""
        try:
            if not context_chunks:
                return self._generate_no_context_response(query)
            
            # Extract relevant information from context
            relevant_info = self._extract_relevant_info(query, context_chunks)
            
            # Generate response based on context
            response = self._generate_contextual_response(query, relevant_info, context_chunks)
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _generate_no_context_response(self, query: str) -> str:
        """Generate response when no context is available"""
        return (
            "I don't have enough information in the uploaded documents to answer your question. "
            "Please make sure you've uploaded relevant documents and try asking a more specific question "
            "about the content in those documents."
        )
    
    def _extract_relevant_info(self, query: str, context_chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract relevant information from context chunks"""
        relevant_info = []
        query_words = set(query.lower().split())
        
        for chunk in context_chunks:
            text = chunk.get("text", "")
            # Simple relevance scoring based on word overlap
            chunk_words = set(text.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            
            if overlap > 0:
                # Extract sentences that contain query words
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if any(word in sentence.lower() for word in query_words) and len(sentence) > 20:
                        relevant_info.append(sentence)
        
        return relevant_info[:5]  # Limit to top 5 relevant pieces
    
    def _generate_contextual_response(
        self, 
        query: str, 
        relevant_info: List[str], 
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """Generate response based on context"""
        
        # Determine response type based on query
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "what is", "define", "explain"]):
            return self._generate_explanation_response(relevant_info, context_chunks)
        elif any(word in query_lower for word in ["how", "how to", "process", "procedure"]):
            return self._generate_process_response(relevant_info, context_chunks)
        elif any(word in query_lower for word in ["list", "show", "enumerate", "what are"]):
            return self._generate_list_response(relevant_info, context_chunks)
        elif any(word in query_lower for word in ["when", "date", "time"]):
            return self._generate_temporal_response(relevant_info, context_chunks)
        elif any(word in query_lower for word in ["where", "location"]):
            return self._generate_location_response(relevant_info, context_chunks)
        elif any(word in query_lower for word in ["why", "reason", "because"]):
            return self._generate_causal_response(relevant_info, context_chunks)
        else:
            return self._generate_general_response(relevant_info, context_chunks)
    
    def _generate_explanation_response(self, relevant_info: List[str], context_chunks: List[Dict[str, Any]]) -> str:
        """Generate explanation-type response"""
        if not relevant_info:
            return "Based on the documents, I couldn't find a clear explanation for your question."
        
        response = "Based on the uploaded documents:\n\n"
        for i, info in enumerate(relevant_info, 1):
            response += f"{i}. {info}\n"
        
        # Add source information
        sources = set(chunk.get("source_file", "Unknown") for chunk in context_chunks)
        response += f"\nThis information comes from: {', '.join(sources)}"
        
        return response
    
    def _generate_process_response(self, relevant_info: List[str], context_chunks: List[Dict[str, Any]]) -> str:
        """Generate process/procedure response"""
        if not relevant_info:
            return "I couldn't find specific process information in the documents for your question."
        
        response = "According to the documents, here's the relevant process information:\n\n"
        
        # Try to identify steps
        steps = []
        for info in relevant_info:
            if any(step_word in info.lower() for step_word in ["step", "first", "second", "then", "next", "finally"]):
                steps.append(info)
            else:
                steps.append(info)
        
        for i, step in enumerate(steps, 1):
            response += f"Step {i}: {step}\n"
        
        return response
    
    def _generate_list_response(self, relevant_info: List[str], context_chunks: List[Dict[str, Any]]) -> str:
        """Generate list-type response"""
        if not relevant_info:
            return "I couldn't find list information relevant to your question in the documents."
        
        response = "From the documents, here are the relevant items:\n\n"
        
        # Extract list items
        items = []
        for info in relevant_info:
            # Look for bullet points, numbers, or comma-separated items
            if "," in info:
                items.extend([item.strip() for item in info.split(",") if item.strip()])
            else:
                items.append(info)
        
        for i, item in enumerate(items[:10], 1):  # Limit to 10 items
            response += f"• {item}\n"
        
        return response
    
    def _generate_temporal_response(self, relevant_info: List[str], context_chunks: List[Dict[str, Any]]) -> str:
        """Generate time-related response"""
        if not relevant_info:
            return "I couldn't find specific date or time information for your question."
        
        response = "Regarding timing information from the documents:\n\n"
        
        # Look for dates, times, or temporal words
        temporal_info = []
        for info in relevant_info:
            if any(time_word in info.lower() for time_word in ["date", "time", "when", "during", "after", "before", "year", "month", "day"]):
                temporal_info.append(info)
        
        if temporal_info:
            for info in temporal_info:
                response += f"• {info}\n"
        else:
            for info in relevant_info:
                response += f"• {info}\n"
        
        return response
    
    def _generate_location_response(self, relevant_info: List[str], context_chunks: List[Dict[str, Any]]) -> str:
        """Generate location-related response"""
        if not relevant_info:
            return "I couldn't find specific location information for your question."
        
        response = "Regarding location information from the documents:\n\n"
        for info in relevant_info:
            response += f"• {info}\n"
        
        return response
    
    def _generate_causal_response(self, relevant_info: List[str], context_chunks: List[Dict[str, Any]]) -> str:
        """Generate causal/reasoning response"""
        if not relevant_info:
            return "I couldn't find specific reasoning or causal information for your question."
        
        response = "Based on the information in the documents:\n\n"
        for info in relevant_info:
            response += f"• {info}\n"
        
        response += "\nThis appears to be the reasoning or explanation provided in the source material."
        return response
    
    def _generate_general_response(self, relevant_info: List[str], context_chunks: List[Dict[str, Any]]) -> str:
        """Generate general response"""
        if not relevant_info:
            return "I found some information in the documents, but it may not directly answer your question."
        
        response = "From the uploaded documents, here's the relevant information I found:\n\n"
        for info in relevant_info:
            response += f"• {info}\n"
        
        response += "\nIf this doesn't fully answer your question, please try rephrasing it or asking about specific aspects mentioned in the documents."
        return response