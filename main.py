from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel
import asyncio
from agent import load_rag_assets, search_query, ask_openrouter
import os

app = FastAPI(title="LangChain RAG Chatbot API")

# Ensure templates directory exists for Jinja2
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

rag_state = {
    "vectorstore": None
}

@app.on_event("startup")
async def startup_event():
    print("Initializing LangChain RAG models...")
    rag_state["vectorstore"] = load_rag_assets()
    print("Models loaded successfully!")

class ChatRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not rag_state["vectorstore"]:
        return {"response": "System is still initializing. Please wait a moment."}
        
    query = request.query
    
    def _process_chat():
        # Retrieve context from the LangChain FAISS index
        context_chunks = search_query(query, rag_state["vectorstore"], top_k=3)
        if not context_chunks:
            return "I couldn't find relevant information in the context."
            
        # Pass the context and query to the LLM
        answer = ask_openrouter(query, context_chunks)
        return answer
        
    response = await asyncio.to_thread(_process_chat)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
