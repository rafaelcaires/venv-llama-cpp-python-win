from unittest.mock import patch

from src import config
from src.downloader import download_model


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
