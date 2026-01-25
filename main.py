from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

import os
import hashlib
import gradio as gr


# Initialize models - using better settings for tutoring
llm = OllamaLLM(
    model="qwen:0.5b",
    temperature=0.7,  # Higher for more natural, conversational responses
    num_ctx=2048,
    num_predict=400  # Longer responses for explanations
)

vectorstore = None 

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


def file_handler( pdf_files, progress=gr.Progress()):
    """  
     Function handles Uploaded Files
    """
    
    # checks if pdf_files is not empty
    if not pdf_files:
        return "Please Select  PDF Files First."
    
    pdf_file = pdf_files[0] if isinstance(pdf_files, list) else pdf_files
    
    # Get Unique ID for this PDF
    pdf_id = get_pdf_id(pdf_file)
    db_path = f"./chroma_db_{pdf_id}"
    
    if os.path.exists(db_path):
        
        # Load from cache instead of processing
        progress(0.5, desc="Found cached version, loading...")
        
        global vectorstore
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        progress(1.0, desc="Loaded from cache!")
        return "Loaded from cache! ready to answer questions."
    
    documents = [] # Empty list to store document
    
    # Show Progress bar 
    progress(0, desc="Reading PDF Files ... ")
    
    total_files = len(pdf_files)
    for i, pdf_file in enumerate(pdf_files):
        # Update progress bar
        progress((i + 1) / total_files, desc=f"Reading file {i+1}/{total_files}")
  
        
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
        
        
        # show process while saving
        progress(0.9, desc="Saving to Database ... ")
        
        vectorstore = Chroma.from_documents(
            documents=chunks, # the chunks that was created
            embedding=embeddings ,
            persist_directory=db_path,
            
        )
        
        # Done! Show success
        progress(1.0, desc="Complete!")
        
        # Return summary
    return f"✓ Processed {len(documents)} pages into {len(chunks)} chunks. Ready!"

def answer_question(question) :
    # This Function Answers Question Using the uploaded documents
    
    # Check if documents have been uploaded
    if vectorstore is None:
        return "Pls Upload a document first!"
    
    # Create question-answering chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, # TinyLLamma
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Find 3 most relevant chunks 
        return_source_documents=False,
    )
    
    # Get answer
    response = qa_chain.invoke({"query": question})
    
    return response["result"]


# Gradio interface

with gr.Blocks(title="Offline Study Assistant") as demo:
    # Title
    gr.Markdown("# 📚 Offline Study Assistant")
    gr.Markdown("Upload your course materials and ask questions!")
    
    # Tab1 Upload Documents
    with gr.Tab("Upload Documents"):
        # File Upload
        pdf_upload = gr.Files(
            label="Upload PDFs",
            file_count="multiple",
            file_types=[".pdf"],
        )
        
        # Process button
        upload_btn = gr.Button("Process Documents")
        
        # Status Output 
        Upload_Output = gr.Textbox(label="Status")
        
        # Connect button to function 
        upload_btn.click(
            fn=file_handler,
            inputs=[pdf_upload],
            outputs=[Upload_Output],
            show_progress=True, # Show Progress bar
        )
        
        gr.Text(" To make the upload faster Upload documents one at a time  ")

        
    with gr.Tab("Ask Questions"):
        # Question input
        question_input = gr.Textbox(
            label="Your Question",
            placeholder=" What is ..... "
        )
        
        # Answer Button
        answer_btn = gr.Button(" Get Answer ")
        
        # Answer Output
        answer_output = gr.Textbox(label="Answer", lines=5)
        
        # Connect button to function
        answer_btn.click(
            fn=answer_question,
            inputs=[question_input],
            outputs=[answer_output],
        )
        
# Run App
if __name__ == "__main__":
    demo.launch(
        show_error=True,
        theme=gr.themes.Origin()
    )
        