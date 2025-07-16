from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import shutil

# Load token from .env
load_dotenv()

# Model details
REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
CACHE_DIR = "./models"

# Compose a unique local filename to avoid overwriting
safe_repo_id = REPO_ID.replace("/", "__")
DEST_FILENAME = f"{safe_repo_id}__{FILENAME}"
FLAT_MODEL_PATH = os.path.join(CACHE_DIR, DEST_FILENAME)

# Load HF token
token = os.getenv("HF_TOKEN")
if token is None:
    raise EnvironmentError("❌ Hugging Face token not found in environment.")

# ✅ Check if model already exists in flat ./models/ directory
if os.path.exists(FLAT_MODEL_PATH):
    print(f"✅ Model already exists at: {FLAT_MODEL_PATH}")
else:
    # Download to HF cache within ./models/
    print("⬇️ Downloading GGUF model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        cache_dir=CACHE_DIR,
        token=token
    )

    # Copy to flat path with unique name
    shutil.copy2(model_path, FLAT_MODEL_PATH)
    print(f"📁 Copied model to: {FLAT_MODEL_PATH}")

print("✅ Done.")
