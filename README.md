# 📚 Offline Study Assistant

An AI-powered tutor that runs completely offline, designed for students with limited internet access and low-end hardware.

## The Problem

Students in regions with limited connectivity face these challenges:
- **No Internet Access**: Can't use cloud-based AI tools that require constant connection
- **Limited Data**: Expensive data plans make online tutoring impractical  
- **Low-End Hardware**: Most AI tools require expensive, high-performance computers
- **Shallow Answers**: Existing AI gives brief responses instead of teaching concepts

## Our Solution

A local AI tutor that:
- ✅ Works completely offline (no internet after setup)
- ✅ Runs on older computers (6th gen Intel and above)
- ✅ Teaches concepts step-by-step, not just quick answers
- ✅ Only answers from your uploaded course materials

## Key Features

**📁 Document Caching**  
Upload a PDF once, load instantly next time. No waiting to reprocess the same textbook.

**💬 Chat Interface**  
Conversation-style learning with history. Ask follow-up questions naturally.

**🎓 Teaching Mode**  
The AI explains concepts thoroughly, breaks down complex ideas, and helps you understand WHY, not just WHAT.

**⚡ Progress Tracking**  
Real-time feedback while processing documents - you always know what's happening.

## Technical Challenges & Solutions

### Challenge 1: Slow Processing on Weak Hardware
**Problem**: Processing 600-page PDFs took 30+ minutes, users thought it crashed.  
**Solution**: Document caching with MD5 hashing. First upload: ~2 minutes. Every upload after: instant.

### Challenge 2: Model Answering Off-Topic Questions  
**Problem**: AI would use general knowledge instead of uploaded documents.  
**Solution**: Used Qwen 1.8B with strict prompts and RAG (Retrieval Augmented Generation) to only answer from document context.

### Challenge 3: Memory Constraints
**Problem**: Large models don't run on 8GB RAM systems.  
**Solution**: Optimized with lightweight Qwen 1.8B, efficient chunking (1000 chars), and ChromaDB vector storage.

### Challenge 4: Poor User Feedback
**Problem**: Users waited with no indication of progress.  
**Solution**: Real-time progress bars with descriptive messages ("Reading file 1/3...", "Splitting documents...").

### Challenge 5: Brief, Unhelpful Answers
**Problem**: Model gave one-sentence responses instead of teaching.  
**Solution**: Teaching-focused prompts instructing the model to explain step-by-step, give examples, and break down complex concepts.

## How It Works

1. Upload your PDF (lecture notes, textbook, etc.)
2. AI processes and caches it (first time only)
3. Ask questions in natural language
4. Get detailed explanations based on your material
5. Chat history keeps context for follow-up questions

## Tech Stack

- **LLM**: Qwen 1.8B (via Ollama)
- **Embeddings**: Nomic Embed Text  
- **Vector DB**: ChromaDB
- **Interface**: Gradio 6.4.0
- **Framework**: LangChain

## Current Limitations

- PDF files only (for now)
- Best with text-based PDFs (not scanned images)
- Occasional inaccuracies with very technical content

---

**Built for students who need accessible, offline education tools**