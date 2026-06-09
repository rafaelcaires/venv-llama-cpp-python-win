import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(_ROOT, "model")
MODEL_FILENAME = "qwen2.5-7b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

REPO_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"

N_CTX = 4096
N_THREADS = os.cpu_count() or 4
N_GPU_LAYERS = 0
TEMPERATURE = 0.7
MAX_TOKENS = 512
