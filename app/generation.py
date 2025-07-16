from llama_cpp import Llama
from langchain.schema import Document
from typing import List
import os

# === Model Config ===
MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MAX_TOKENS = 256
TEMPERATURE = 0.7
USE_GPU = False  # Set to True if using GPU build

# === Initialize LLaMA Model ===
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=os.cpu_count(),
    n_gpu_layers=-1 if USE_GPU else 0,
    verbose=False
)

# === Generation Function ===
def generate_answer(query: str, context_docs: List[Document]) -> str:
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""استخدم السياق التالي للإجابة على السؤال أدناه.

السياق:
{context_text}

السؤال:
{query}

إذا لم يوفر السياق معلومات كافية للإجابة، قل "لا أعلم".
دائمًا أجب باللغة العربية، حتى لو كان السؤال باللغة الإنجليزية.
"""

    response = llm.create_completion(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9
    )
    return response["choices"][0]["text"].strip()
