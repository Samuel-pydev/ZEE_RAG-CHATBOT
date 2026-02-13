from model import answer_question
from pathlib import Path

import os
from datetime import datetime
import json

# Folder for coversations
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

# Track current conversation ID
current_conversation_id = None

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
        
        

def chat_response(question, history):
    """
        Handle chat with auto-save
    """
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

def load_conversation(conv_id):
    # Loads conversation from file
    conv_file = CONVERSATIONS_DIR / f"{conv_id}.json"
    
    if conv_file.exists():
        with open(conv_file, "r") as f:
            return json.load(f)
    return None

def list_conversations():
    conversations = []
    
    for conv_file in CONVERSATIONS_DIR.glob("*.json"):
        with open(conv_file, "r") as f:
            data = json.load(f)
            
            # Add timestamp if missing
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()
            
            conversations.append({
                "id": data["id"],
                "title": data["title"],
                "timestamp": data["timestamp"]
            })
    
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