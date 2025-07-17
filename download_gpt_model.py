from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import shutil

# Load token from .env
load_dotenv()

# Model details
REPO_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"
FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
CACHE_DIR = "./models"

# Compose unique filename
safe_repo_id = REPO_ID.replace("/", "__")
DEST_FILENAME = f"{safe_repo_id}__{FILENAME}"
FLAT_MODEL_PATH = os.path.join(CACHE_DIR, DEST_FILENAME)

# Ensure directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Download if not already present
if os.path.exists(FLAT_MODEL_PATH):
    print(f"✅ Model already exists at: {FLAT_MODEL_PATH}")
else:
    print("⬇️ Downloading GGUF model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        cache_dir=CACHE_DIR,
        token=None
    )
    shutil.copy2(model_path, FLAT_MODEL_PATH)
    print(f"📁 Copied model to: {FLAT_MODEL_PATH}")

print("✅ Done.")
