from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain


import os
import hashlib
import gradio as gr


# Initialize models - using better settings for tutoring
llm = OllamaLLM(
    model="qwen:1.8b",
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

def answer_question(question):
    # This function answers questions using uploaded documents
    
    # Check if documents were uploaded
    if vectorstore is None:
        return "Please upload documents first!"     
    
    
    template = """You are a patient and helpful tutor. Your goal is to TEACH the student, not just give quick answers.

    When answering:
    1. Explain the concept clearly using the context
    2. Break it down step-by-step if it's complex
    3. Give examples from the context when possible
    4. Help the student understand WHY, not just WHAT

    IMPORTANT: Only use information from the context below. If the answer isn't in the context, say "I don't have information about that in the uploaded document."

    Context from uploaded document:
    {context}

    Student's question: {question}

    Your explanation (remember to teach, not just answer):"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Create QA chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # Get answer
    response = qa_chain.invoke({"query": question})
    
    # Return the answer
    return response["result"]

def chat_response(question, history):
    # This function handles chat-style responses with history
    
    if not question.strip():
        return history, ""
    
    # Get answer from your existing function
    answer = answer_question(question)
    
    # Gradio 6.x uses dictionary format for messages
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    
    # Return updated history and clear input
    return history, ""

# Gradio interface

with gr.Blocks(title="Offline Study Assistant", theme=gr.themes.Origin()) as demo:
    
    gr.Markdown("# 📚 Offline Study Assistant")
    
    with gr.Row():
        # LEFT SIDEBAR - Upload Documents
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Documents")
            
            pdf_upload = gr.File(
                label="Select PDF",
                file_count="single",
                file_types=[".pdf"]
            )
            
            upload_btn = gr.Button("Process Document", variant="primary")
            upload_status = gr.Textbox(label="Status", lines=2, interactive=False)
            
            upload_btn.click(
                fn=file_handler,
                inputs=[pdf_upload],
                outputs=[upload_status],
                show_progress=True
            )
            
            gr.Markdown("---")
            gr.Markdown("💡 **Tip:** Upload one document at a time for faster processing")
        
        # RIGHT SIDE - Chat Interface
        with gr.Column(scale=3):
            gr.Markdown("### 💬 Ask Questions")
            
            # ChatBot component (stores history automatically)
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                buttons=['copy_all']
            )
            
            # Question input
            question_input = gr.Textbox(
                label="",
                placeholder="Ask a question about your document...",
                lines=2
            )
            
            # Buttons row
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")
            
            # Connect buttons
            submit_btn.click(
                fn=chat_response,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            question_input.submit(  # Also submit on Enter key
                fn=chat_response,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            clear_btn.click(
                fn=lambda: ([], ""),
                inputs=[],
                outputs=[chatbot, question_input]
             )

if __name__ == "__main__":
    demo.launch(show_error=True)