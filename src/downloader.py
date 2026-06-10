from __future__ import annotations

import os

from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

from src import config


def download_model(
    repo_id: str = config.REPO_ID,
    filename: str = config.MODEL_FILENAME,
    model_dir: str = config.MODEL_DIR,
) -> str:
    """Download the GGUF model from Hugging Face; skip if already present."""
    dest = os.path.join(model_dir, filename)
    if os.path.exists(dest):
        print(f"Model already exists: {dest}")
        return dest

    print(f"Downloading {filename} from {repo_id} ...")
    path: str = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
    )
    print(f"Saved to: {path}")
    return path


if __name__ == "__main__":  # pragma: no cover
    download_model()
