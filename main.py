import os
import streamlit as st
import chromadb
import ollama  # Using native package to strictly control memory usage
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from pypdf import PdfReader

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Free AI Interview Prep Assistant", page_icon="💼", layout="wide")
st.title("💼 Free Local AI Interview Preparation Assistant")
st.write("Running ultra-low RAM configuration via native Ollama engine.")

# --- 2. CONFIGURING ULTRA-LIGHTWEIGHT LOCAL EMBEDDINGS ---
# We cap context here to protect your system memory from crashing
Settings.embed_model = OllamaEmbedding(
    model_name="llama3.2",
    options={"num_ctx": 1024}
)

PERSIST_DIR = "./chroma_interview_db"

def extract_pdf_text(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    # Only grab up to the first 3 pages to strictly prevent memory overloads
    for i, page in enumerate(pdf_reader.pages):
        if i >= 3: 
            break
        text += page.extract_text() or ""
    return text.strip()

db = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = db.get_or_create_collection("interview_history")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 3. SESSION STATE MANAGEMENT ---
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = []
if "interview_messages" not in st.session_state:
    st.session_state.interview_messages = []
if "current_question_idx" not in st.session_state:
    st.session_state.current_question_idx = 0
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📄 Step 1: Upload Resume")
    uploaded_file = st.file_uploader("Choose your Resume (PDF)", type=["pdf"])
    target_role = st.text_input("Target Job Role", value="Software Engineer")
    
    if st.button("Analyze & Generate Questions", type="primary"):
        if uploaded_file:
            with st.spinner("Analyzing resume locally via native low-memory engine..."):
                resume_text = extract_pdf_text(uploaded_file)
                st.session_state.resume_text = resume_text
                
                prompt = (
                    f"Based on the following resume text, generate exactly 3 specific interview questions "
                    f"tailored for a {target_role} position. Format them as a clean list with one question per line.\n\n"
                    f"Resume:\n{resume_text[:2000]}" # Truncate incoming string to protect RAM
                )
                
                # Calling native Ollama directly with hardcoded resource capping
                response = ollama.generate(
                    model="llama3.2",
                    prompt=prompt,
                    options={"num_ctx": 2048, "num_predict": 256}
                )
                
                raw_text = response.get('response', '')
                questions = [q.strip().lstrip("123456789.- ") for q in raw_text.split("\n") if q.strip()]
                st.session_state.generated_questions = questions[:3]
                st.session_state.current_question_idx = 0
                st.session_state.interview_messages = []
                st.session_state.interview_started = False
                st.success("🎯 Questions generated!")
        else:
            st.sidebar.warning("Please upload a PDF resume first.")

# --- 5. MAIN INTERFACE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Target Interview Questions")
    if st.session_state.generated_questions:
        for idx, q in enumerate(st.session_state.generated_questions):
            st.markdown(f"**Q{idx+1}:** {q}")
        
        if not st.session_state.interview_started:
            if st.button("🚀 Start Mock Interview", type="secondary"):
                st.session_state.interview_started = True
                first_q = st.session_state.generated_questions[0]
                st.session_state.interview_messages.append({
                    "role": "assistant", 
                    "content": f"Welcome to your local mock interview! Let's start with question 1:\n\n**{first_q}**"
                })
                st.rerun()
    else:
        st.info("Upload your resume in the sidebar to generate custom interview practice questions.")

with col2:
    st.header("💬 Live Interview Simulator")
    
    if st.session_state.interview_started:
        for msg in st.session_state.interview_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if st.session_state.current_question_idx < len(st.session_state.generated_questions):
            if user_answer := st.chat_input("Type your answer here..."):
                st.session_state.interview_messages.append({"role": "user", "content": user_answer})
                
                current_q = st.session_state.generated_questions[st.session_state.current_question_idx]
                eval_prompt = (
                    f"You are a technical interviewer. Evaluate this answer.\n"
                    f"Question: {current_q}\n"
                    f"Candidate Answer: {user_answer}\n\n"
                    f"Provide a brief 2-sentence constructive review and score out of 10."
                )
                
                with st.spinner("Local AI evaluating performance..."):
                    response = ollama.generate(
                        model="llama3.2",
                        prompt=eval_prompt,
                        options={"num_ctx": 2048, "num_predict": 256}
                    )
                    eval_text = response.get('response', 'Evaluation complete.')
                    st.session_state.interview_messages.append({"role": "assistant", "content": eval_text})
                
                st.session_state.current_question_idx += 1
                
                if st.session_state.current_question_idx < len(st.session_state.generated_questions):
                    next_q = st.session_state.generated_questions[st.session_state.current_question_idx]
                    st.session_state.interview_messages.append({
                        "role": "assistant", 
                        "content": f"Next question:\n\n**{next_q}**"
                    })
                else:
                    st.session_state.interview_messages.append({
                        "role": "assistant", 
                        "content": "🎉 **Mock interview complete!** Review your feedback ratings above."
                    })
                
                doc = Document(text=f"Q: {current_q} | A: {user_answer}")
                VectorStoreIndex.from_documents([doc], storage_context=storage_context)
                st.rerun()