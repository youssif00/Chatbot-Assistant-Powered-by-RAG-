# Chatbot Assistant Powered by RAG 🤖

An end-to-end Retrieval-Augmented Generation (RAG) based AI chatbot designed to assist users by answering technical questions from internal documentation. This system combines semantic search with large language models (LLMs) to provide accurate, context-aware, and grounded responses.

## 🌟 Features

- FastAPI-based REST API 
- RAG (Retrieval-Augmented Generation) architecture
- Semantic search using FAISS vector store
- Document embeddings using HuggingFace's nomic-embed-text-v1.5
- Chat history persistence using SQLite and SQLalchemy
- MMR (Maximal Marginal Relevance) retrieval for diverse results
- Session management for contextual conversations
- Google's Gemini-2.0-flash LLM integration

## 🔧 Tech Stack

- FastAPI
- SQLAlchemy
- FAISS
- LangChain
- HuggingFace Transformers
- Google Gemini API
- SQLite
- Pydantic

## 📋 Requirements

```pip
fastapi[all]==0.110.0
uvicorn[standard]==0.29.0
sqlalchemy==2.0.30
langchain==0.2.1
faiss-cpu==1.7.4
transformers==4.41.2
sentence-transformers==2.6.1
pydantic==2.7.1
pydantic-settings==2.10.1
pydantic-extra-types==2.10.5
```

## 🚀 Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/youssif00/Chatbot-Assistant-Powered-by-RAG-.git
cd Chatbot-Assistant-Powered-by-RAG-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the Gemini API:
   - Replace `"Your api from Gemini API"` in `llama_engine.py` with your actual Gemini API key

4. Initialize the vector store:
   - Make sure your documents are properly indexed in the `faiss_index_by_file` directory

5. Run the application:
```bash
uvicorn app:app --reload
```

## 🔄 API Endpoints

### POST /chat
Create a new chat message or continue an existing conversation.

Request body:
```json
{
    "session_id": "optional-session-id",
    "message": "Your question here"
}
```

Response:
```json
{
    "response": "Bot's response",
    "sources": ["list", "of", "source", "documents"]
}
```

### GET /history/{session_id}
Retrieve chat history for a specific session.

Response:
```json
[
    {
        "user": "User message",
        "bot": "Bot response",
        "timestamp": "Message timestamp"
    }
]
```

## 🏗️ Architecture

1. **Vector Store**: Uses FAISS for efficient similarity search with nomic-embed-text-v1.5 embeddings
2. **Retriever**: Implements MMR strategy for diverse and relevant document retrieval
3. **LLM Integration**: Uses Google's Gemini-2.0-flash model for response generation
4. **Database**: SQLite for storing chat history and session management
5. **API Layer**: FastAPI for handling HTTP requests and response generation

## 📝 Database Schema

```sql
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR,
    user_message TEXT,
    bot_response TEXT,
    timestamp DATETIME
)
```

## 🤝 Contributing

Feel free to open issues and pull requests to improve the project!

📄 License
This project is licensed under the MIT License. See LICENSE for details.

✉️ Contact
Yossif Reda – yossifreda500@gmail.com
