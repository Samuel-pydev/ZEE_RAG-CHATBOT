from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import gradio as gr

# Initialize models - using better settings for tutoring
llm = OllamaLLM(
    model="tinyllama:latest",
    temperature=0.7,  # Higher for more natural, conversational responses
    num_ctx=2048,
    num_predict=400  # Longer responses for explanations
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

def file_handler( pdf_files, progress=gr.progress()):
    """  
     Function handles Uploaded Files
    """
    
    # checks if pdf_files is not empty
    if not pdf_files:
        return "Please Select a PDF File First."
    
    documents = [] # Empty list to store document
    
    # Show Progress bar 
    progress(0, desc="Reading PDF Files ... ")
    
    total_files = len(pdf_files)
    for i, pdf_file in enumerate(pdf_files):
        # Update Progress bar
        progress((i + 1) / total_files, desc = f"Reading Files {i+1}/{total_files} ")
        
        # Read this PDF
        loader = PyPDFLoader(pdf_file.name)
        
        #Add all pages to list 
        documents.extend(loader.load())
        
        # Tell user how many pages we found 
        progress(0.3, desc=f"Loaded {len(documents)} pages. Now Processing ... ")
        
        # Split Document into Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100,
        )
        
        # Show Progress while splitting
        progress(0.5, desc="Splitting documents ... ")
        chunks = text_splitter.split_documents(documents)
        
        # Tell user how many chunks we created
        progress(0.7, desc=f"Splitted Documents into {len(chunks)}. Now Saving to Database. ")
        
        # Store chunks in vector database
        global vectorestore # Update the global vectore variable
        
        # show process while saving
        progress(0.9, desc="Saving to Database ... ")
        
        vectorestore = Chroma.from_documents(
            documents=chunks, # the chunks that was created
            embedding=embeddings ,
            persist_directory="./chroma_db"
            
        )
        
    

