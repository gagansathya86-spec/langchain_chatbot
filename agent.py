import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
# Use dynamic path so it works both locally on Windows and remotely on Render (Linux)
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL = "openrouter/free"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Try to load from .env if running locally and key not in env vars
if not OPENROUTER_API_KEY:
    env_path = os.path.join(DIRECTORY, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    OPENROUTER_API_KEY = line.strip().split("=", 1)[1]

def load_rag_assets():
    print("Loading LangChain FAISS index...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Check if we need to rename index (1).faiss to index.faiss
    faiss_wrong_name = os.path.join(DIRECTORY, "index (1).faiss")
    faiss_correct_name = os.path.join(DIRECTORY, "index.faiss")
    if os.path.exists(faiss_wrong_name) and not os.path.exists(faiss_correct_name):
        try:
            os.rename(faiss_wrong_name, faiss_correct_name)
        except Exception as e:
            print(f"Warning: Could not rename file: {e}")
    
    # Load the local FAISS index created with LangChain
    vectorstore = FAISS.load_local(
        folder_path=DIRECTORY,
        embeddings=embedding_model,
        index_name="index",
        allow_dangerous_deserialization=True
    )
    print("Successfully loaded the LangChain FAISS index!")
    return vectorstore

def search_query(query, vectorstore, top_k=20):
    # Perform similarity search
    docs = vectorstore.similarity_search(query, k=top_k)
    context = [doc.page_content for doc in docs]
    
    # Deterministic fallback for exact USNs or Names from the CSV
    csv_path = os.path.join(DIRECTORY, "Students_2026_V2.csv")
    if os.path.exists(csv_path):
        import csv
        query_lower = query.lower()
        terms = [t for t in query_lower.split() if len(t) > 3]
        
        exact_matches = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_str = " ".join(str(v) for v in row.values()).lower()
                # Prioritize USN exact match or strong keyword match
                if any(term in row_str for term in terms):
                    exact_matches.append(f"USN: {row['USN']}, Name: {row['Name']}, Subject: {row['Subject']}, Faculty: {row['Faculty']}")
                    
        # Add up to 100 direct matches to the context to help with counting and exact lookups
        if exact_matches:
            context.append("--- DIRECT DATABASE MATCHES ---")
            context.extend(exact_matches[:100])
            
    return context

def ask_openrouter(query, context_chunks):
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY is not set. Please set it in a .env file or environment variables."
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    context_text = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question accurately. 
If the answer cannot be found in the context, say so. Do not invent information.

Context:
{context_text}

User Question: {query}
"""

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error from OpenRouter: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error communicating with OpenRouter: {str(e)}"
