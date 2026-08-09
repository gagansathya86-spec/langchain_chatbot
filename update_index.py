import os
import warnings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    
    all_docs = []
    
    print("Loading Timetable (CSV)...")
    timetable_csv = os.path.join(directory, "Timetable_2026.csv")
    if os.path.exists(timetable_csv):
        loader = CSVLoader(file_path=timetable_csv, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = "Timetable_2026.csv"
            d.metadata["file_type"] = "csv"
        all_docs.extend(docs)
    else:
        print(f" Warning: Could not find {timetable_csv}.")
        
    print("Loading Timetable (Text)...")
    timetable_txt = os.path.join(directory, "timetable_2026.txt")
    if os.path.exists(timetable_txt):
        loader = TextLoader(timetable_txt, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = "timetable_2026.txt"
            d.metadata["file_type"] = "txt"
        all_docs.extend(docs)
    else:
        print(f" Warning: Could not find {timetable_txt}.")
        
    print("Loading Student Lists (CSV)...")
    students_csv = os.path.join(directory, "Students_2026_V2.csv")
    if os.path.exists(students_csv):
        loader = CSVLoader(file_path=students_csv, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = "Students_2026_V2.csv"
            d.metadata["file_type"] = "csv"
        all_docs.extend(docs)
    else:
        print(f" Warning: Could not find {students_csv}.")
        
    # List of PDFs to ingest
    pdfs_to_load = [
        "brochure_compressed20-21.pdf",
        "7th semester COE 2026.pdf"
    ]
    
    for pdf_name in pdfs_to_load:
        print(f"Loading {pdf_name}...")
        pdf_path = os.path.join(directory, pdf_name)
        if os.path.exists(pdf_path):
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = pdf_name
                    d.metadata["file_type"] = "pdf"
                all_docs.extend(docs)
                print(f" Successfully loaded {pdf_name}")
            except Exception as e:
                print(f" Error loading {pdf_name}: {e}")
        else:
            print(f" Warning: Could not find {pdf_name}")
            
    if not all_docs:
        print("No documents were loaded. Exiting...")
        return
        
    print(f"Loaded {len(all_docs)} total document items.")
    
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} text chunks.")
    
    print("Initializing embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("Building FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print(f"Vector store created with {vectorstore.index.ntotal} vectors.")
    
    print("Saving vector index to disk (index.faiss, index.pkl)...")
    vectorstore.save_local(folder_path=directory, index_name="index")
    
    print("Done! The knowledge base has been successfully updated.")

if __name__ == "__main__":
    main()
