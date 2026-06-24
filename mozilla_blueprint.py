import pymupdf4llm
import ollama
import os

def extract_and_segment_pdf(pdf_path):
    """
    Step 1 & 2: Converts PDF to structured Markdown via PyMuPDF4LLM
    and slices the document clean into a dictionary mapping Headings -> Text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find the target file: {pdf_path}")
        
    print(f"Extracting layout and structural data from {os.path.basename(pdf_path)}...")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    sections = {}
    current_heading = "Introduction/Overview"
    current_content = []
    
    # Process text lines to extract clean markdown header boundaries
    for line in md_text.split("\n"):
        if line.strip().startswith(("# ", "## ", "### ")):
            if current_content:
                sections[current_heading] = "\n".join(current_content).strip()
            current_heading = line.replace("#", "").strip()
            current_content = [line]
        else:
            current_content.append(line)
            
    if current_content:
        sections[current_heading] = "\n".join(current_content).strip()
        
    return sections

def find_relevant_section_with_router(query, section_titles):
    """
    Step 3: Roaming RAG Router. Passes ONLY titles to a lightweight local LLM.
    Zero vector embeddings needed.
    """
    titles_list = "\n".join([f"- {title}" for title in section_titles])
    
    system_prompt = (
        "You are an information routing assistant. Look at the list of available document section titles "
        "and select the single most relevant title that contains the answer to the user's question.\n"
        "Respond ONLY with the exact title string. No explanations, no introduction, no surrounding punctuation."
    )
    
    user_prompt = f"Available Sections:\n{titles_list}\n\nUser Question: {query}\n\nMost Relevant Section:"
    
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.0} # Absolute deterministic accuracy for strict selection
    )
    
    return response['message']['content'].strip()

def generate_isolated_answer(query, context):
    """
    Step 4: Answer generation using localized layout context boundaries.
    """
    system_prompt = (
        "You are a helpful, precise Q&A assistant. Answer the user question strictly using the provided "
        "document context segment. If the context does not contain the answer, say 'I cannot find that info'."
    )
    
    user_prompt = f"Document Context Block:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.2}
    )
    
    return response['message']['content'].strip()

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Point this to your uploaded resume file
    pdf_file = "uploaded_docs/S_Bhagyasaree_Resume(1).pdf" 
    user_query = "What technical skills or programming languages are mentioned?"
    
    try:
        # Step 1 & 2: Extract structured text chunks without chunk overlapping calculations
        document_chunks = extract_and_segment_pdf(pdf_file)
        available_titles = list(document_chunks.keys())
        
        print(f"Successfully tracked {len(available_titles)} markdown-backed structural zones.")
        
        # Step 3: Run the lightweight routing logic
        best_section = find_relevant_section_with_router(user_query, available_titles)
        print(f"-> Local Router LLM Selected Chapter: '{best_section}'")
        
        # Fallback security catch
        if best_section not in document_chunks:
            matched = [t for t in available_titles if t.lower() in best_section.lower()]
            best_section = matched[0] if matched else available_titles[0]
            
        retrieved_context = document_chunks[best_section]
        
        # Step 4: Extract and present the precise content
        final_answer = generate_isolated_answer(user_query, retrieved_context)
        
        print("\n=== Mozilla Blueprint Final Answer ===")
        print(final_answer)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")