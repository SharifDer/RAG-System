from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from app.retrieval import retrieve_top_n  # Must return List[Document]
from app.generation import generate_answer, generate_answer_streaming
import time

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add these lines right after creating your FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/query")
def query_answer(request: QueryRequest):
    retrieved_docs = retrieve_top_n(request.query, k=request.top_k)

    start = time.time()
    answer = generate_answer(request.query, retrieved_docs)
    gen_time = time.time() - start

    return {
        "query": request.query,
        "answer": answer,
        "context_documents": [doc.page_content for doc in retrieved_docs],
        "generation_time": f"{gen_time:.2f}s"
    }

@app.post("/query/stream")
def stream_answer(request: QueryRequest):
    retrieved_docs = retrieve_top_n(request.query, k=request.top_k)

    def token_generator():
        for chunk in generate_answer_streaming(request.query, retrieved_docs):
            yield chunk

    return StreamingResponse(token_generator(), media_type="text/plain")
