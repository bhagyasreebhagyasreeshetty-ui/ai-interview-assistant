import os
import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="RAG Document Chatbot", page_icon="📚", layout="centered")
st.title("📚 Knowledge-Based Document Chatbot")
st.write("Upload your PDF documentation and ask questions based on its content.")

# --- 2. CONFIGURING LLM & EMBEDDINGS ---
# Safely fetch the API key from environment variables or Streamlit secrets
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found! Please set it in your environment or terminal before running.")
    st.stop()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Directories for data and database
UPLOAD_DIR = "./data"
PERSIST_DIR = "./chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 3. CORE RAG PIPELINE FUNCTION ---
@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    """Initializes and caches the connection to ChromaDB and the Vector Store."""
    db = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = db.get_or_create_collection("knowledge_base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Check if we already have documents indexed on disk
    if chroma_collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    return None

# --- 4. SIDEBAR FOR FILE UPLOADS ---
with st.sidebar:
    st.header("Upload Documentation")
    uploaded_files = st.file_uploader(
        "Choose PDF files to add to the knowledge base", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Index Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents... This might take a moment."):
                # Save uploaded files locally to the data directory
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Ingest files via LlamaIndex
                documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()
                
                db = chromadb.PersistentClient(path=PERSIST_DIR)
                chroma_collection = db.get_or_create_collection("knowledge_base")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Create index and clear cache to force update
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
                st.cache_resource.clear()
                
                st.success(f"✅ Successfully indexed {len(uploaded_files)} document(s)!")
                st.rerun()
        else:
            st.warning("Please upload at least one PDF file first.")

# Try to load existing index
index = initialize_rag_pipeline()

# --- 5. CHAT INTERFACE ---
# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your documentation assistant. Upload some files on the left, and ask me anything!"}
    ]

# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if index is None:
            response_text = "⚠️ I don't have any data yet. Please upload and process your PDFs using the sidebar menu!"
            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            # Create chat engine on the fly
            chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", similarity_top_k=3)
            
            # Stream the response tokens directly into the UI
            response_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                response_stream = chat_engine.stream_chat(prompt)
                for token in response_stream.response_gen:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                    
            response_placeholder.markdown(full_response)
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})