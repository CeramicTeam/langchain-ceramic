from __future__ import annotations

import os
from typing import Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr, model_validator

from ceramic_ai import AsyncCeramic, Ceramic


class CeramicRetriever(BaseRetriever):
    """Retriever that uses Ceramic's web search API.

    Setup:
        Install the package and set your API key:

        .. code-block:: bash

            pip install langchain-ceramic
            export CERAMIC_API_KEY="your-api-key"

    Example:
        .. code-block:: python

            from langchain_ceramic import CeramicRetriever

            retriever = CeramicRetriever(k=5)
            docs = retriever.invoke("latest AI chip export restrictions")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: Optional[str] = None
    k: int = 10

    _client: Ceramic = PrivateAttr(default=None)
    _async_client: AsyncCeramic = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_api_key(self) -> "CeramicRetriever":
        key = self.api_key or os.environ.get("CERAMIC_API_KEY")
        if not key:
            raise ValueError(
                "Ceramic API key required. Pass api_key= or set CERAMIC_API_KEY."
            )
        self._client = Ceramic(api_key=key)
        self._async_client = AsyncCeramic(api_key=key)
        return self

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        response = self._client.search(query=query)
        return [
            Document(
                page_content=result.description or "",
                metadata={"title": result.title, "url": result.url},
            )
            for result in response.result.results[: self.k]
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        response = await self._async_client.search(query=query)
        return [
            Document(
                page_content=result.description or "",
                metadata={"title": result.title, "url": result.url},
            )
            for result in response.result.results[: self.k]
        ]
