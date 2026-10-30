"""
Pytest configuration and environment fixtures for smb-loan-visual-underwriter.
Ensures external dependencies are mocked gracefully if not installed in the test environment.
"""

import sys
from unittest.mock import MagicMock
import pytest

# Ensure missing third-party modules can be imported during tests
MOCK_MODULES = [
    "dotenv",
    "exif",
    "google",
    "google.cloud",
    "google.cloud.secretmanager",
    "googlemaps",
    "googlemaps.exceptions",
    "streamlit",
    "vertexai",
    "vertexai.generative_models",
]

for mod in MOCK_MODULES:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            mock_obj = MagicMock()
            sys.modules[mod] = mock_obj
            if mod == "googlemaps":
                mock_obj.exceptions = MagicMock()
                mock_obj.exceptions.ApiError = type("ApiError", (Exception,), {})
                sys.modules["googlemaps.exceptions"] = mock_obj.exceptions


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock Streamlit UI calls so tests run cleanly in headless mode."""
    import streamlit as st

    st.spinner = MagicMock(
        return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=None),
        )
    )
    st.write = MagicMock()
    st.markdown = MagicMock()
    st.error = MagicMock()
    st.title = MagicMock()
    st.set_page_config = MagicMock()
    st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
    st.image = MagicMock()
    st.button = MagicMock(return_value=False)
    st.session_state = {}
    yield st
