from unittest.mock import patch

import pytest

from src.chat import main


@pytest.fixture
def model_exists():
    with patch("src.chat.os.path.exists", return_value=True):
        yield


@pytest.fixture
def model_missing():
    with patch("src.chat.os.path.exists", return_value=False):
        yield


def test_exits_normally_on_exit_command(model_exists, capsys):
    with patch("src.chat.LLMInference"), patch("builtins.input", side_effect=["exit"]):
        main()
    assert "Bye!" in capsys.readouterr().out


def test_exits_normally_on_quit_command(model_exists, capsys):
    with patch("src.chat.LLMInference"), patch("builtins.input", side_effect=["quit"]):
        main()
    assert "Bye!" in capsys.readouterr().out


def test_exits_normally_on_sair_command(model_exists, capsys):
    with patch("src.chat.LLMInference"), patch("builtins.input", side_effect=["sair"]):
        main()
    assert "Bye!" in capsys.readouterr().out


def test_exits_on_keyboard_interrupt(model_exists):
    with patch("src.chat.LLMInference"), patch("builtins.input", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_exits_on_eof(model_exists):
    with patch("src.chat.LLMInference"), patch("builtins.input", side_effect=EOFError):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0


def test_skips_empty_input(model_exists):
    with patch("src.chat.LLMInference") as MockLLM, patch(
        "builtins.input", side_effect=["", "exit"]
    ):
        main()
    MockLLM.return_value.chat.assert_not_called()


def test_sends_message_and_prints_response(model_exists, capsys):
    with patch("src.chat.LLMInference") as MockLLM, patch(
        "builtins.input", side_effect=["Hello", "exit"]
    ):
        MockLLM.return_value.chat.return_value = "Hi there!"
        main()
    assert "Hi there!" in capsys.readouterr().out


def test_downloads_model_when_missing(model_missing, capsys):
    with patch("src.chat.download_model") as mock_dl, patch(
        "src.chat.LLMInference"
    ), patch("builtins.input", side_effect=["exit"]):
        main()
    mock_dl.assert_called_once()


def test_skips_download_when_model_exists(model_exists):
    with patch("src.chat.download_model") as mock_dl, patch(
        "src.chat.LLMInference"
    ), patch("builtins.input", side_effect=["exit"]):
        main()
    mock_dl.assert_not_called()
