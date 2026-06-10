import os

from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model
_model_dir_raw = os.getenv("MODEL_DIR", "model")
MODEL_DIR = _model_dir_raw if os.path.isabs(_model_dir_raw) else os.path.join(_ROOT, _model_dir_raw)
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "Qwen3-4B-Q4_K_M.gguf")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_ID = os.getenv("MODEL_ID", "qwen3-4b")
REPO_ID = os.getenv("REPO_ID", "Qwen/Qwen3-4B-GGUF")

# Inference
N_CTX = int(os.getenv("N_CTX", "8192"))
N_THREADS = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))
N_BATCH = int(os.getenv("N_BATCH", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv("API_KEY", "")
AUTO_DOWNLOAD = os.getenv("AUTO_DOWNLOAD", "true").lower() == "true"
