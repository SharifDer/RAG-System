from llama_cpp import Llama
from langchain.schema import Document
from typing import List
import os

# === Model Config ===
MODEL_PATH = "./models/microsoft__Phi-3-mini-4k-instruct-gguf__Phi-3-mini-4k-instruct-q4.gguf"
MAX_TOKENS = 256
TEMPERATURE = 0.7
USE_GPU = True

# === Initialize LLaMA Model ===
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=os.cpu_count(),
    n_gpu_layers=-1 if USE_GPU else 0,
    verbose=False
)

# === Non-Streaming Generation ===
def generate_answer(query: str, context_docs: List[Document]) -> str:
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""Use the following context to answer the question below.

Context:
{context_text}

Question:
{query}
Please follow these instructions very well :
1- You are a smart rag system that extracts answers from the context provided. make sure you send a clean well structured answer.
2- Don't include any tags like "answer" or "response" just give clear sentence or response without adding any tags.
3- Do not generate one word sentence. Instead say the full sentence.

"""

    response = llm.create_completion(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9
    )
    return response["choices"][0]["text"].strip()

# === Streaming Generation ===
def generate_answer_streaming(query: str, context_docs: List[Document]):
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""Context:
{context_text}

Q: {query}
A: Answer in a full sentence, rephrasing the question. Do not start with labels. Do not answer with just a number. Plus Don't go into any detailss.
"""


# 4- Make sure to just give the answer. Don't mention context or anything or justifiy your answer, Say your answer in clean short sentence or sentences.
# 5- Only give the answer to the query, Don't provide details unless you are asked to do so.



    stream = llm.create_completion(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
        stream=True
    )

    for chunk in stream:
        token = chunk["choices"][0]["text"]
        yield token

# === Manual test block ===
if __name__ == "__main__":
    # Sample context
    context = [
        Document(page_content="Artificial intelligence is a field of computer science that aims to create systems capable of simulating human intelligence."),
        Document(page_content="Large language models like GPT and Phi-3 are used to generate text and answer questions.")
    ]

    # Sample question
    question = "What is artificial intelligence and what is it used for?"
    generate_answer_streaming(question , context)
    # Generate answer
    # answer = generate_answer(question, context)
    # print("\n=== Answer ===")
    # print(answer)
