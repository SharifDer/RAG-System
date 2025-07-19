from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.retrieval import retrieve_top_n  # This should return List[Document]
from app.generation import generate_answer  # Your existing LLaMA-based function

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3  # Optional override for number of retrieved docs

@app.post("/query")
def query_answer(request: QueryRequest):
    # Retrieve relevant documents
    retrieved_docs = retrieve_top_n(request.query, k=request.top_k)

    # Generate answer using the model
    answer = generate_answer(request.query, retrieved_docs)

    return {
        "query": request.query,
        "answer": answer,
        "context_documents": [doc.page_content for doc in retrieved_docs]
    }
