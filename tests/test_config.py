import os
from src import config


def test_model_path_ends_with_gguf():
    assert config.MODEL_PATH.endswith(".gguf")


def test_model_dir_exists_or_creatable():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    assert os.path.isdir(config.MODEL_DIR)


def test_context_and_token_defaults():
    assert config.N_CTX > 0
    assert config.MAX_TOKENS > 0
    assert config.N_THREADS > 0


def test_temperature_range():
    assert 0.0 <= config.TEMPERATURE <= 2.0
