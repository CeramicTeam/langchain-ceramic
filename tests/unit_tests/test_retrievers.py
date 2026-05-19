import os
from unittest.mock import MagicMock, patch

import pytest

from langchain_ceramic import CeramicSearchRetriever, CeramicSearch


def test_retriever_init():
    with patch("langchain_ceramic.retrievers.Ceramic"), \
         patch("langchain_ceramic.retrievers.AsyncCeramic"):
        r = CeramicSearchRetriever(api_key="test-key")
    assert r.k == 10


def test_tool_init():
    with patch("langchain_ceramic.tools.Ceramic"), \
         patch("langchain_ceramic.tools.AsyncCeramic"):
        t = CeramicSearch(api_key="test-key")
    assert t.name == "ceramic_search"


def test_retriever_raises_without_key(monkeypatch):
    monkeypatch.delenv("CERAMIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Ceramic API key required"):
        CeramicSearchRetriever()


def test_tool_raises_without_key(monkeypatch):
    monkeypatch.delenv("CERAMIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Ceramic API key required"):
        CeramicSearch()


def test_retriever_reads_env_var(monkeypatch):
    monkeypatch.setenv("CERAMIC_API_KEY", "env-key")
    with patch("langchain_ceramic.retrievers.Ceramic") as mock_ceramic, \
         patch("langchain_ceramic.retrievers.AsyncCeramic"):
        r = CeramicSearchRetriever()
    mock_ceramic.assert_called_once_with(api_key="env-key")
    assert r.api_key is None  # api_key field stays None; key came from env


def test_tool_reads_env_var(monkeypatch):
    monkeypatch.setenv("CERAMIC_API_KEY", "env-key")
    with patch("langchain_ceramic.tools.Ceramic") as mock_ceramic, \
         patch("langchain_ceramic.tools.AsyncCeramic"):
        t = CeramicSearch()
    mock_ceramic.assert_called_once_with(api_key="env-key")


def test_retriever_get_documents():
    mock_result = MagicMock()
    mock_result.title = "Test Title"
    mock_result.description = "Test description"
    mock_result.url = "https://example.com"

    mock_response = MagicMock()
    mock_response.result.results = [mock_result] * 15

    with patch("langchain_ceramic.retrievers.Ceramic") as mock_ceramic, \
         patch("langchain_ceramic.retrievers.AsyncCeramic"):
        mock_ceramic.return_value.search.return_value = mock_response
        r = CeramicSearchRetriever(api_key="test-key", k=5)

    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    run_manager = MagicMock(spec=CallbackManagerForRetrieverRun)
    docs = r._get_relevant_documents("test query", run_manager=run_manager)

    assert len(docs) == 5
    assert docs[0].page_content == "Test description"
    assert docs[0].metadata == {"title": "Test Title", "url": "https://example.com"}
