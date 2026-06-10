"""
Mock heavy native packages that require compilation/download so that the
test suite can run in any environment without them installed.
"""

import sys
from unittest.mock import MagicMock

# Mock the top-level module and its submodules used in downloader.py
_hf_hub = MagicMock()
_hf_errors = MagicMock()

# Expose real exception classes so tests can catch them properly
_hf_errors.EntryNotFoundError = type("EntryNotFoundError", (Exception,), {})
_hf_errors.RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})

_hf_hub.errors = _hf_errors

sys.modules.setdefault("huggingface_hub", _hf_hub)
sys.modules.setdefault("huggingface_hub.errors", _hf_errors)
sys.modules.setdefault("llama_cpp", MagicMock())
