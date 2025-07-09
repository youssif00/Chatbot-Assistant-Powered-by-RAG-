from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database import SessionLocal, ChatHistory
from llama_engine import call_llama
from vector_store import load_and_index_documents_by_file  # your index builder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import uuid

app = FastAPI(title="Chatbot Assistant For Customer Service", debug=True)

# 1) Initialize the embedding model exactly as in your indexer
embedding_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"trust_remote_code": True}
)

# 2) Load the FAISS index with that same model
vectorstore = FAISS.load_local(
    "faiss_index_by_file",
    embedding_model,
    allow_dangerous_deserialization=True
)

# 3) Create an MMR retriever to call at query time
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 4}
)

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    db = SessionLocal()
    session_id = request.session_id or str(uuid.uuid4())

    # Retrieve previous chat history for context
    previous = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).all()
    context = "\n".join([f"User: {msg.user_message}\nBot: {msg.bot_response}" for msg in previous])

    # Perform semantic search to find relevant documentation chunks
    docs = retriever.get_relevant_documents(request.message)
    sources = list({doc.metadata['source'] for doc in docs})
    doc_context = "\n".join(doc.page_content for doc in docs)

    # Create an instructive prompt for the LLaMA model
    prompt = (
        """    You are an expert assistant specialized in answering questions strictly based on the provided context.

        You will receive a block of context, retrieved from a vector database, containing relevant information to answer the question.

        ### Your mission:

        * Use only the information provided in the context to generate your response.
        * Do not rely on external knowledge, your memory, assumptions, or personal opinions.
        * If the answer is not found in the context or cannot be fully deduced from it, explicitly respond:
        "The provided context does not contain enough information to answer the question."
        * Always base your reasoning, conclusions, and details directly on the context.
        * Include image tags that are present in Markdown format in the context with the same path (e.g., ![alt text](image_1.png)), include them in appropriate locations.

        ### Instructions:

        * Be clear, concise, and precise in your response.
        * *Do not fabricate any information* that is not found in the context.
        * Cite or reference specific parts of the context if necessary to support your response.
        * If the context provides *partial* information, clearly state that in your response.
        * Maintain a neutral and factual tone.
        * The response format must be in Markdown.

        ### Context:

        ### Your response format (in Markdown):

        *Answer:* [your answer based on the context]        
        """
        f"\nDocumentation Context:\n{doc_context}\n"
        f"\nChat History:\n{context}\n"
        f"\nUser: {request.message}\nAssistant:"
    )

    # Generate a response using the LLaMA model
    response = call_llama(prompt)

    # Save interaction to database
    chat_entry = ChatHistory(
        session_id=session_id,
        user_message=request.message,
        bot_response=response
    )
    db.add(chat_entry)
    db.commit()
    db.close()

    return ChatResponse(response=response, sources=sources)

@app.get("/history/{session_id}")
def get_history(session_id: str):
    db = SessionLocal()
    history = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).all()
    db.close()
    return [{"user": h.user_message, "bot": h.bot_response, "timestamp": h.timestamp} for h in history]
