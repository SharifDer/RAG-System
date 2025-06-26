from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import chroma

INDEX_DIR = "./data/faiss_index"
COLLECTION_NAME = "support_issues"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vectordb = chroma(
    collection_name = COLLECTION_NAME,
    persist_directory = INDEX_DIR,
    embedding_function = embeddings
)