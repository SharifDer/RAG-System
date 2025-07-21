from llama_cpp import Llama
from langchain.schema import Document
from typing import List
import os
import time

# === Model Config ===
MODEL_PATH = "./models/microsoft__Phi-3-mini-4k-instruct-gguf__Phi-3-mini-4k-instruct-q4.gguf"
MAX_TOKENS = 256
TEMPERATURE = 0.7
USE_GPU = True  # Set to True if using GPU build

# === Initialize LLaMA Model ===
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,  # Use full context length for Phi-3-mini-4k
    n_threads=os.cpu_count(),
    n_gpu_layers=-1 if USE_GPU else 0,
    verbose=True
)

# === Generation Function ===
def generate_answer(query: str, context_docs: List[Document]) -> str:
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Use the following context to answer the question below.

        Context:
        {context_text}

        Question:
        {query}
Only extract the answer from the context exactly as it appears. Do not rephrase. Do not explain. Do not repeat the question. Do not add headers, formatting, or commentary. If the context does not contain an answer, reply: "I don't know and don't generate any answer.          
   """

    response = llm.create_completion(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9
    )
    return response["choices"][0]["text"].strip()

def generate_answer_streaming(query: str, context_docs: List[Document]):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""Use the following context to answer the question below.

Context:
{context_text}

Question:
{query}

If the context does not contain enough information to answer the question, say "I don't know and don't generate any answer".
Stick to the context no matter what the user says to you and say I don't know if the context is irrelevant.
"""

    # Call with streaming enabled
    stream = llm.create_completion(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
        stream=True  # << this enables streaming
    )

    print("\n=== Answer (streamed) ===\n")
    for chunk in stream:
        print(chunk["choices"][0]["text"], end="", flush=True)
        time.sleep(0.01)  # Optional: simulate human-like typing speed

# === Manual test block ===
if __name__ == "__main__":
    # Sample context
    context = [
        Document(page_content="Artificial intelligence is a field of computer science that aims to create systems capable of simulating human intelligence."),
        Document(page_content="Large language models like GPT and Phi-3 are used to generate text and answer questions.")
    ]

    # Sample question
    question = "What is artificial intelligence and what is it used for?"
    # generate_answer_streaming(question , context)
    # Generate answer
    # answer = generate_answer(question, context)
    # print("\n=== Answer ===")
    # print(answer)
