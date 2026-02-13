from langchain_ollama import OllamaLLM
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain

# from document_handler import vectorstore
from shared_state import get_vectorstore


# Initialize models - using better settings for tutoring
llm = OllamaLLM(
    model="phi3:latest ",
    temperature=0.8,  # Higher for more natural, conversational responses
    num_ctx=2048,
    num_predict=500  # Longer responses for explanations
)

def answer_question(question, history):
    """
    This function answers questions using uploaded documents
    """

    vectorstore = get_vectorstore()

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
                