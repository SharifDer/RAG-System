import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
import shutil
PDF_DIR = "./data/raw_pdfs"
INDEX_DIR = "./data/faiss_index"
COLLECTION_NAME = "support_issues"

if os.path.exists(INDEX_DIR):
    shutil.rmtree(INDEX_DIR)
def load_and_split_pdfs(pdf_dir):
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"] , chunk_size = 1000 , chunk_overlap=50)
    documents = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir , file)
            loader = PyMuPDFLoader(path)
            documents.extend(loader.load_and_split(text_splitter=splitter))
    return documents

def ingest_to_chroma(docs):
    embedding_function = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path=INDEX_DIR)
    collection = client.get_or_create_collection(
                                                name=COLLECTION_NAME 
                                                ,embedding_function=embedding_function
                                                # ,metadata={"hnsw:space": "cosine"}
                                                )
    for i, doc in enumerate(docs):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[doc.page_content],
            metadatas=[doc.metadata],
        )
    print(f"✅ Persisted {len(docs)} chunks to ChromaDB at {INDEX_DIR}")


if __name__ == "__main__":
    docs = load_and_split_pdfs(PDF_DIR)
    print(f"➡️ {len(docs)} chunks ready for ingestion")
    ingest_to_chroma(docs)
