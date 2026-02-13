import gradio as gr

from document_handler import file_handler
from chat_manager import get_conversation_list, chat_response, start_new_chat, load_selected_chat, get_cached_docs_display, delete_cached_document

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
                height=600
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
