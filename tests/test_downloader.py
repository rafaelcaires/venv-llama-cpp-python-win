from unittest.mock import patch

import pytest

from src import config
from src.downloader import download_model, list_gguf_files


def test_skip_download_if_model_exists(tmp_path):
    model_file = tmp_path / config.MODEL_FILENAME
    model_file.write_text("fake")

    with patch("src.downloader.hf_hub_download") as mock_dl:
        result = download_model(model_dir=str(tmp_path))

    mock_dl.assert_not_called()
    assert result == str(model_file)


def test_download_called_when_model_missing(tmp_path):
    expected = str(tmp_path / config.MODEL_FILENAME)

    with patch("src.downloader.hf_hub_download", return_value=expected) as mock_dl:
        result = download_model(model_dir=str(tmp_path))

    mock_dl.assert_called_once_with(
        repo_id=config.REPO_ID,
        filename=config.MODEL_FILENAME,
        local_dir=str(tmp_path),
    )
    assert result == expected


def test_custom_repo_and_filename(tmp_path):
    expected = str(tmp_path / "custom.gguf")

    with patch("src.downloader.hf_hub_download", return_value=expected) as mock_dl:
        result = download_model(
            repo_id="org/custom-model",
            filename="custom.gguf",
            model_dir=str(tmp_path),
        )

    mock_dl.assert_called_once_with(
        repo_id="org/custom-model",
        filename="custom.gguf",
        local_dir=str(tmp_path),
    )
    assert result == expected


def test_repo_not_found_raises_system_exit(tmp_path):
    from huggingface_hub.errors import RepositoryNotFoundError

    with patch(
        "src.downloader.hf_hub_download", side_effect=RepositoryNotFoundError("404")
    ):
        with pytest.raises(SystemExit):
            download_model(repo_id="nonexistent/repo", model_dir=str(tmp_path))


def test_entry_not_found_raises_system_exit_with_suggestions(tmp_path, capsys):
    from huggingface_hub.errors import EntryNotFoundError

    with patch(
        "src.downloader.hf_hub_download", side_effect=EntryNotFoundError("404")
    ), patch(
        "src.downloader.list_repo_files",
        return_value=["ModelA-Q4_K_M.gguf", "ModelA-Q8_0.gguf", "notes.txt"],
    ):
        with pytest.raises(SystemExit):
            download_model(
                repo_id="org/repo",
                filename="missing.gguf",
                model_dir=str(tmp_path),
            )

    out = capsys.readouterr().out
    assert "missing.gguf" in out
    assert "ModelA-Q4_K_M.gguf" in out
    assert "notes.txt" not in out  # only .gguf files are shown


def test_entry_not_found_no_gguf_files(tmp_path, capsys):
    from huggingface_hub.errors import EntryNotFoundError

    with patch(
        "src.downloader.hf_hub_download", side_effect=EntryNotFoundError("404")
    ), patch("src.downloader.list_repo_files", return_value=["README.md"]):
        with pytest.raises(SystemExit):
            download_model(model_dir=str(tmp_path))

    assert "Nenhum arquivo .gguf" in capsys.readouterr().out


def test_entry_not_found_list_fails_gracefully(tmp_path, capsys):
    from huggingface_hub.errors import EntryNotFoundError

    with patch(
        "src.downloader.hf_hub_download", side_effect=EntryNotFoundError("404")
    ), patch("src.downloader.list_repo_files", side_effect=Exception("network error")):
        with pytest.raises(SystemExit):
            download_model(model_dir=str(tmp_path))

    assert "Não foi possível listar" in capsys.readouterr().out


def test_list_gguf_files_returns_only_gguf():
    with patch(
        "src.downloader.list_repo_files",
        return_value=["model-Q4.gguf", "model-Q8.gguf", "config.json", "README.md"],
    ):
        result = list_gguf_files("org/repo")

    assert result == ["model-Q4.gguf", "model-Q8.gguf"]
    assert "config.json" not in result
