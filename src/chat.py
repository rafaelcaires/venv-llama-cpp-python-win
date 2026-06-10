"""Interactive CLI chat with the loaded model."""

from __future__ import annotations

import os
import sys

from src import config
from src.downloader import download_model
from src.inference import LLMInference


def main() -> None:
    if not os.path.exists(config.MODEL_PATH):
        download_model()

    print("Loading model...")
    llm = LLMInference()
    print("Model ready. Type 'exit' to quit.\n")

    history: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            sys.exit(0)

        if user_input.lower() in {"exit", "quit", "sair"}:
            print("Bye!")
            break

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        response = llm.chat(history)
        history.append({"role": "assistant", "content": response})
        print(f"Assistant: {response}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
