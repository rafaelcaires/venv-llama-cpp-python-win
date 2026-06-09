"""Interactive CLI chat with the loaded model."""
import sys
import os

from src.downloader import download_model
from src.inference import LLMInference
from src import config


def main():
    if not os.path.exists(config.MODEL_PATH):
        download_model()

    print("Loading model...")
    llm = LLMInference()
    print("Model ready. Type 'exit' to quit.\n")

    history = [{"role": "system", "content": "You are a helpful assistant."}]

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


if __name__ == "__main__":
    main()
