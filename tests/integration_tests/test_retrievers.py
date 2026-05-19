import pytest
from langchain_ceramic import CeramicSearchRetriever


@pytest.mark.integration
def test_retriever_returns_docs():
    retriever = CeramicSearchRetriever()  # reads CERAMIC_API_KEY from env
    docs = retriever.invoke("latest AI news")
    assert len(docs) > 0
    assert docs[0].page_content
    assert "url" in docs[0].metadata
