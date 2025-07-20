from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import chromadb
import time

INDEX_DIR = "./data/faiss_index"
COLLECTION_NAME = "support_issues"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

#Load existing chromadb stored collection 
vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=INDEX_DIR,
    # client_settings=chromadb.Settings(anonymized_telemetry=False),
    # collection_metadata={"hnsw:space": "cosine"}
)
print("metadatas",vectordb._collection.metadata)
def retrieve_top_n(query:str , k:int =3) -> List[Document]:
    start = time.time()
    query_embeddings = embeddings.embed_query(query)
    results = vectordb.similarity_search_by_vector(
        query_embeddings,
        k=k,
        )
    retrieve_time = time.time() - start
    print(f"retrieve time: {retrieve_time:.2f}s")
    return results