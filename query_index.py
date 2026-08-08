import os
import warnings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    directory = r"c:\Users\gagan\Downloads\langchain"
    
    # Check if we need to rename 'index (1).faiss' to 'index.faiss' to match 'index.pkl'
    faiss_wrong_name = os.path.join(directory, "index (1).faiss")
    faiss_correct_name = os.path.join(directory, "index.faiss")
    
    if os.path.exists(faiss_wrong_name) and not os.path.exists(faiss_correct_name):
        try:
            os.rename(faiss_wrong_name, faiss_correct_name)
            print("Renamed 'index (1).faiss' to 'index.faiss' to match 'index.pkl'")
        except Exception as e:
            print(f"Warning: Could not rename file: {e}")
            print("You may need to manually rename 'index (1).faiss' to 'index.faiss'.")
        
    print("Loading embedding model...")
    # Initialize the same embeddings model used to build the index
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("Loading FAISS index...")
    try:
        # Load the local FAISS index
        vectorstore = FAISS.load_local(
            folder_path=directory,
            embeddings=embedding_model,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        print("Successfully loaded the FAISS index!")
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        return

    print("="*50)
    print("LangChain Vector Store Query Interface")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)
    
    # Interactive loop to ask the user for an input query
    while True:
        try:
            query = input("\nEnter your query: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        
        if query.strip().lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        if not query.strip():
            continue
            
        print("\nSearching...")
        try:
            # Perform similarity search
            docs = vectorstore.similarity_search(query, k=3)
            
            if not docs:
                print("No relevant documents found.")
                continue
                
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown source')
                print(f"\n--- Result {i+1} (Source: {source}) ---")
                print(doc.page_content)
        except Exception as e:
            print(f"An error occurred during search: {e}")

if __name__ == "__main__":
    main()
