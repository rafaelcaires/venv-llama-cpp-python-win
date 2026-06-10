"""
Mock heavy native packages that require compilation/download so that the
test suite can run in any environment without them installed.
"""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("llama_cpp", MagicMock())
sys.modules.setdefault("huggingface_hub", MagicMock())
