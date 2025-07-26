import streamlit as st
import asyncio
import uuid
from datetime import datetime
from agents.coordinator_agent import CoordinatorAgent
from agents.mcp import MCPMessage

# Initialize session state
if 'coordinator' not in st.session_state:
    st.session_state.coordinator = CoordinatorAgent()
    
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
    
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = []

st.title("🤖 Agentic RAG Chatbot")
st.markdown("Upload documents and ask questions using our multi-agent RAG system with Model Context Protocol")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'pptx', 'docx', 'csv', 'txt', 'md'],
        accept_multiple_files=True,
        help="Supported formats: PDF, PPTX, DOCX, CSV, TXT, Markdown"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [doc['name'] for doc in st.session_state.uploaded_documents]:
                # Process document
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Create MCP message for document ingestion
                        trace_id = str(uuid.uuid4())
                        message = MCPMessage(
                            sender="UI",
                            receiver="CoordinatorAgent",
                            type="DOCUMENT_UPLOAD",
                            trace_id=trace_id,
                            payload={
                                "file_name": uploaded_file.name,
                                "file_content": uploaded_file.read(),
                                "file_type": uploaded_file.type
                            }
                        )
                        
                        # Process document through coordinator
                        result = asyncio.run(st.session_state.coordinator.handle_message(message))
                        
                        if result.payload.get("status") == "success":
                            st.session_state.uploaded_documents.append({
                                "name": uploaded_file.name,
                                "type": uploaded_file.type,
                                "processed_at": datetime.now()
                            })
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}: {result.payload.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    # Display uploaded documents
    if st.session_state.uploaded_documents:
        st.subheader("📚 Processed Documents")
        for doc in st.session_state.uploaded_documents:
            st.write(f"• {doc['name']}")
    
    # Clear documents button
    if st.button("🗑️ Clear All Documents"):
        st.session_state.uploaded_documents = []
        st.session_state.coordinator.clear_documents()
        st.success("All documents cleared!")
        st.rerun()

# Main chat interface
st.header("💬 Chat Interface")

# Display conversation history
for message in st.session_state.conversation_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            if "sources" in message:
                with st.expander("📖 Source References"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:** {source}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.uploaded_documents:
        st.warning("Please upload some documents first!")
    else:
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process query through coordinator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    trace_id = str(uuid.uuid4())
                    message = MCPMessage(
                        sender="UI",
                        receiver="CoordinatorAgent",
                        type="QUERY_REQUEST",
                        trace_id=trace_id,
                        payload={
                            "query": prompt,
                            "conversation_history": st.session_state.conversation_history[-10:]  # Last 10 messages for context
                        }
                    )
                    
                    result = asyncio.run(st.session_state.coordinator.handle_message(message))
                    
                    if result.payload.get("status") == "success":
                        response = result.payload.get("response", "No response generated")
                        sources = result.payload.get("sources", [])
                        
                        st.write(response)
                        
                        if sources:
                            with st.expander("📖 Source References"):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"**Source {i}:** {source}")
                        
                        # Add assistant message to history
                        assistant_message = {
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        }
                        st.session_state.conversation_history.append(assistant_message)
                    else:
                        error_msg = result.payload.get("error", "Unknown error occurred")
                        st.error(f"❌ Error: {error_msg}")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

# Display system status
with st.expander("🔧 System Status"):
    st.write("**Agent Status:**")
    st.write("• CoordinatorAgent: ✅ Active")
    st.write("• IngestionAgent: ✅ Active")
    st.write("• RetrievalAgent: ✅ Active")
    st.write("• LLMResponseAgent: ✅ Active")
    
    st.write(f"**Documents Processed:** {len(st.session_state.uploaded_documents)}")
    st.write(f"**Conversation Turns:** {len(st.session_state.conversation_history)}")
