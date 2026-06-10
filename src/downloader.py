from __future__ import annotations

import os

from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore[import-not-found]
from huggingface_hub.errors import (  # type: ignore[import-not-found]
    EntryNotFoundError,
    RepositoryNotFoundError,
)

from src import config


def list_gguf_files(repo_id: str = config.REPO_ID) -> list[str]:
    """Return all .gguf filenames available in a Hugging Face repository."""
    return sorted(f for f in list_repo_files(repo_id) if f.endswith(".gguf"))


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
    try:
        path: str = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=model_dir,
        )
    except RepositoryNotFoundError:
        raise SystemExit(
            f"\nRepositório não encontrado: {repo_id}\n"
            "Verifique a variável REPO_ID no seu .env."
        ) from None
    except EntryNotFoundError:
        _suggest_files(repo_id, filename)
        raise SystemExit(1) from None

    print(f"Saved to: {path}")
    return path


def _suggest_files(repo_id: str, missing: str) -> None:
    print(f"\nArquivo '{missing}' não encontrado em '{repo_id}'.")
    try:
        available = list_gguf_files(repo_id)
        if available:
            print("\nArquivos .gguf disponíveis neste repositório:")
            for f in available:
                print(f"  {f}")
            print(
                "\nDefina MODEL_FILENAME no seu .env com um dos nomes acima e tente novamente."
            )
        else:
            print("Nenhum arquivo .gguf encontrado no repositório.")
    except Exception:
        print("Não foi possível listar os arquivos do repositório.")


if __name__ == "__main__":  # pragma: no cover
    download_model()
