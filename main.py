from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains import LLMChain


from pathlib import Path
from datetime import datetime


import os
import hashlib
import gradio as gr
import json


# Folder for coversations
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

# Track current conversation ID
current_conversation_id = None

# Initialize models - using better settings for tutoring
llm = OllamaLLM(
    model="qwen:1.8b",
    temperature=0.8,  # Higher for more natural, conversational responses
    num_ctx=2048,
    num_predict=500  # Longer responses for explanations
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

def answer_question(question, history):
    # This function answers questions using uploaded documents
    
    # Check if documents were uploaded
    if vectorstore is None:
        return "Please upload documents first!"
    
    # Build conversation context from history
    conversation_context = ""
    if history:
        recent_history = history[-6:]
        for msg in recent_history:
            if msg["role"] == "user":
                conversation_context += f"Student: {msg['content']}\n"
            else:
                conversation_context += f"Tutor: {msg['content']}\n"
    
    # Check if this is casual conversation (greetings, thanks, etc.)
    casual_phrases = ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye", 
                     "how are you", "what's up", "good morning", "good night"]
    
    question_lower = question.lower().strip()
    is_casual = any(phrase in question_lower for phrase in casual_phrases)
    
    # If casual conversation, respond without document context
    if is_casual and len(question_lower.split()) < 10:  # Short casual messages

        
        
        casual_template = """You are a friendly tutor assistant. The student is greeting you or making casual conversation.

Respond naturally and briefly (1-2 sentences). Be warm and helpful.

Previous conversation:
{conversation}

Student says: {question}

Your response:"""
        
        casual_prompt = PromptTemplate(
            template=casual_template,
            input_variables=["conversation", "question"]
        )
        
        chain = LLMChain(llm=llm, prompt=casual_prompt)
        response = chain.invoke({
            "conversation": conversation_context,
            "question": question
        })
        
        return response["text"]
    
    # Otherwise, it's a real question - use document context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    
    template = """You are a patient and helpful tutor.

When answering academic questions:
1. Explain the concept clearly using the context
2. Break it down step-by-step if complex
3. Give examples from the context
4. Help the student understand WHY, not just WHAT

IMPORTANT: Only use information from the context below. If the answer isn't in the context, say "I don't have information about that in the uploaded document."

Previous conversation:
{conversation}

Context from document:
{context}

Student's question: {question}

Your teaching explanation:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["conversation", "context", "question"]
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.invoke({
        "conversation": conversation_context,
        "context": context,
        "question": question
    })
    
    return response["text"]

def chat_response(question, history):
    # Handle chat with auto-save
    
    global current_conversation_id
    
    if not question.strip():
        return history, ""
    
    # If no conversation started, create new one
    if current_conversation_id is None:
        current_conversation_id = generate_conversation_id()
    
    # Get answer with history context
    answer = answer_question(question, history)
    
    # Add to history
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    
    # Auto-save conversation
    save_conversation(current_conversation_id, history)
    
    return history, ""

def generate_conversation_id():
    # Creates unique ID for new conversation
    import uuid
    return str(uuid.uuid4())[:8]


def save_conversation(conv_id, history, title=None):
    # Saves conversation to file
    conv_file = CONVERSATIONS_DIR / f"{conv_id}.json"
    
    # If no title, use first user message
    if not title and history:
        for msg in history:
            if msg["role"] == "user":
                title = msg["content"][:50]  # First 50 chars
                break
    
    data = {
        "id": conv_id,
        "title": title or "New Chat",
        "history": history,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(conv_file, "w") as f:
        json.dump(data, f, indent=2)
        
        
def load_conversation(conv_id):
    # Loads conversation from file
    conv_file = CONVERSATIONS_DIR / f"{conv_id}.json"
    
    if conv_file.exists():
        with open(conv_file, "r") as f:
            return json.load(f)
    return None


def list_conversations():
    # Gets list of all conversations
    conversations = []
    
    for conv_file in CONVERSATIONS_DIR.glob("*.json"):
        with open(conv_file, "r") as f:
            data = json.load(f)
            conversations.append({
                "id": data["id"],
                "title": data["title"],
                "timestamp": data["timestamp"]
            })
            
# Sort by most recent first
    conversations.sort(key=lambda x: x["timestamp"], reverse=True)
    return conversations


def start_new_chat():
    # Starts a fresh conversation
    global current_conversation_id
    current_conversation_id = generate_conversation_id()
    return [], ""  # Empty history and input

def load_chat(conv_id):
    # Loads an existing conversation
    global current_conversation_id
    current_conversation_id = conv_id
    
    data = load_conversation(conv_id)
    if data:
        return data["history"], data["title"]
    return [], "Chat not found"

def get_conversation_list():
    # Returns list of conversations (for Gradio to display as buttons)
    conversations = list_conversations()
    
    if not conversations:
        return []
    
    # Return list of (id, display_text) tuples
    conv_options = []
    for conv in conversations:
        # Truncate title to 30 characters
        title = conv["title"]
        if len(title) > 30:
            title = title[:27] + "..."
        
        # Add timestamp
        timestamp = datetime.fromisoformat(conv["timestamp"])
        time_str = timestamp.strftime("%b %d, %H:%M")
        
        display_text = f"{title} - {time_str}"
        conv_options.append((display_text, conv["id"]))  # (label, value)
    
    return conv_options

def load_selected_chat(conv_id):
    # Loads selected conversation
    global current_conversation_id
    
    if not conv_id:
        return [], "### Current Chat"
    
    current_conversation_id = conv_id
    data = load_conversation(conv_id)
    
    if data:
        title = f"### {data['title']}"
        return data["history"], title
    
    return [], "### Chat not found"

def list_cached_documents():
    # Lists all cached documents with their info
    import os
    from pathlib import Path
    
    cached_docs = []
    
    # Find all chroma_db folders
    for folder in Path(".").glob("chroma_db_*"):
        if folder.is_dir():
            # Get folder size
            total_size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)  # Convert to MB
            
            # Get creation time
            created = folder.stat().st_ctime
            created_str = datetime.fromtimestamp(created).strftime("%b %d, %Y %H:%M")
            
            # Extract hash ID from folder name
            doc_id = folder.name.replace("chroma_db_", "")
            
            cached_docs.append({
                "id": doc_id,
                "folder": folder.name,
                "size": f"{size_mb:.2f} MB",
                "created": created_str
            })
    
    return cached_docs

def delete_cached_document(doc_id):
    # Deletes a cached document
    import shutil
    
    folder_name = f"chroma_db_{doc_id}"
    
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        return f"✓ Deleted cached document: {doc_id}"
    else:
        return "❌ Document not found"

def get_cached_docs_display():
    # Formats cached documents for display
    docs = list_cached_documents()
    
    if not docs:
        return "No cached documents", []
    
    # Format as table
    display = "**Cached Documents:**\n\n"
    for doc in docs:
        display += f"- **{doc['id']}** | {doc['size']} | Cached: {doc['created']}\n"
    
    # Return display text and list of IDs for dropdown
    doc_ids = [(f"{doc['id']} ({doc['size']})", doc['id']) for doc in docs]
    
    return display, doc_ids


with gr.Blocks(title="Offline Study Assistant", theme=gr.themes.Origin()) as demo:
    
    gr.Markdown("# 📚 Offline Study Assistant")
    
    with gr.Row():
        # LEFT SIDEBAR
        with gr.Column(scale=1):
            with gr.Tabs():
                # Tab 1: Upload Documents
                with gr.Tab("📤 Upload"):
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
                
                # Tab 2: Cached Documents
                with gr.Tab("💾 Cached"):
                    cached_display = gr.Markdown("No cached documents")
                    
                    cached_selector = gr.Dropdown(
                        label="Select document to delete",
                        choices=[],
                        interactive=True
                    )
                    
                    with gr.Row():
                        refresh_cache_btn = gr.Button("Refresh", size="sm")
                        delete_cache_btn = gr.Button("Delete", size="sm", variant="stop")
                    
                    delete_status = gr.Textbox(label="Status", lines=1, interactive=False)
            
            gr.Markdown("---")
            
            # Conversations section
            gr.Markdown("### 💬 Conversations")
            new_chat_btn = gr.Button("+ New Chat", variant="secondary")
            
            # Dropdown for selecting conversations
            conversation_selector = gr.Dropdown(
                label="Past Chats",
                choices=get_conversation_list(),
                interactive=True
            )
            
            load_chat_btn = gr.Button("Load Selected Chat")
            refresh_btn = gr.Button("Refresh List")
            
        
        # RIGHT SIDE - Chat Interface
        with gr.Column(scale=5):
            chat_title = gr.Markdown("### Current Chat")
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=700
            )
            
            question_input = gr.Textbox(
                label="",
                placeholder="Ask a question about your document...",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Current Chat")
            
            # Button connections
            submit_btn.click(
                fn=chat_response,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            question_input.submit(
                fn=chat_response,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            new_chat_btn.click(
                fn=start_new_chat,
                inputs=[],
                outputs=[chatbot, question_input]
            )
            
            clear_btn.click(
                fn=lambda: ([], ""),
                inputs=[],
                outputs=[chatbot, question_input]
            )
            
            # Load selected chat
            load_chat_btn.click(
                fn=load_selected_chat,
                inputs=[conversation_selector],
                outputs=[chatbot, chat_title]
            )
            
            # Refresh conversation list
            refresh_btn.click(
                fn=lambda: gr.Dropdown(choices=get_conversation_list()),
                inputs=[],
                outputs=[conversation_selector]
            )
            
            # Cached documents functions
            def refresh_cached():
                display, ids = get_cached_docs_display()
                return display, gr.Dropdown(choices=ids)
            
            refresh_cache_btn.click(
                fn=refresh_cached,
                inputs=[],
                outputs=[cached_display, cached_selector]
            )
            
            delete_cache_btn.click(
                fn=delete_cached_document,
                inputs=[cached_selector],
                outputs=[delete_status]
            )
            
            # Load cached docs on startup
            demo.load(
                fn=refresh_cached,
                inputs=[],
                outputs=[cached_display, cached_selector]
            )

if __name__ == "__main__":
    demo.launch(show_error=True)