from langchain_community.vectorstores.chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List

INDEX_DIR = "./data/faiss_index"
COLLECTION_NAME = "support_issues"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

#Load existing chromadb stored collection 
vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=INDEX_DIR
)

def retrieve_top_n(query:str , k:int =3) -> List[Document]:
    query_embeddings = embeddings.embed_query(query)
    results = vectordb.similarity_search_by_vector(query_embeddings, k=k)
    return results