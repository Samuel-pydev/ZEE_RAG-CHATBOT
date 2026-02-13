from langchain_community.vectorstores import Chroma, FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader    

from datetime import datetime

import os
import hashlib
import gradio as gr
import shared_state


# vectorstore = None 

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)



def get_pdf_id(pdf_file):
    """
    Creates a Unique Id for each PDF based on it's filename and size 
    to avoid processing the document everytime it's uploaded 
    """
    filename = os.path.basename(pdf_file.name)
    
    # Create a short unique ID from the filename and size
    return hashlib.md5(filename.encode()).hexdigest()

def file_handler(pdf_files, progress=gr.Progress()):
    """  
    Function handles Uploaded Files
    """
    
    # checks if pdf_files is not empty
    if not pdf_files:
        return "Please Select PDF Files First."
    
    # Get single file (we only process one at a time)
    pdf_file = pdf_files[0] if isinstance(pdf_files, list) else pdf_files
    
    # Get Unique ID for the PDF File
    pdf_id = get_pdf_id(pdf_file)
    db_path = f"./chroma_db_{pdf_id}"
    
    # Check if already cached
    if os.path.exists(db_path):
        progress(0.5, desc="Found cached version, loading...")
        
        global vectorstore
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        # Store in shared state
        shared_state.set_vectorestore(vectorstore)

        progress(1.0, desc="Loaded from cache!")
        return "✓ Loaded from cache! Ready to answer questions."
    
    # If not cached, process the file
    progress(0.1, desc="Reading PDF file...")
    
    # Read the PDF
    loader = PyPDFLoader(pdf_file.name)  # Use pdf_file, not pdf_files
    documents = loader.load()
    
    progress(0.3, desc=f"Loaded {len(documents)} pages. Splitting...")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_documents(documents)
    
    progress(0.7, desc=f"Created {len(chunks)} chunks. Saving to database...")
    
    # Save to database (this is the slow part)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    # Store in shared state
    shared_state.set_vectorestore(vectorstore)
    
    progress(1.0, desc="Complete!")
    
    return f"✓ Processed {len(documents)} pages into {len(chunks)} chunks. Ready!"
    