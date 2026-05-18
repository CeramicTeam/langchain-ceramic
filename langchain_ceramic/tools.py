from __future__ import annotations

import os
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ceramic_ai import AsyncCeramic, Ceramic


class CeramicSearchInput(BaseModel):
    query: str = Field(description="Search query to look up on the web.")


class CeramicSearch(BaseTool):
    """Tool that queries the Ceramic web search API.

    Setup:
        .. code-block:: bash

            pip install langchain-ceramic
            export CERAMIC_API_KEY="your-api-key"

    Example:
        .. code-block:: python

            from langchain_ceramic import CeramicSearch

            tool = CeramicSearch(max_results=5)
            tool.invoke("latest developments in GLP-1 drugs")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "ceramic_search"
    description: str = (
        "A web search tool powered by Ceramic. "
        "Use this to find current information from the web. "
        "Input should be a search query string."
    )
    args_schema: Type[BaseModel] = CeramicSearchInput

    api_key: Optional[str] = None
    max_results: int = 5

    _client: Ceramic = PrivateAttr(default=None)
    _async_client: AsyncCeramic = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_api_key(self) -> "CeramicSearch":
        key = self.api_key or os.environ.get("CERAMIC_API_KEY")
        if not key:
            raise ValueError(
                "Ceramic API key required. Pass api_key= or set CERAMIC_API_KEY."
            )
        self._client = Ceramic(api_key=key)
        self._async_client = AsyncCeramic(api_key=key)
        return self

    def _run(self, query: str) -> str:
        response = self._client.search(query=query)
        results = response.result.results[: self.max_results]
        return "\n\n".join(
            f"**{r.title}**\n{r.description}\nSource: {r.url}" for r in results
        )

    async def _arun(self, query: str) -> str:
        response = await self._async_client.search(query=query)
        results = response.result.results[: self.max_results]
        return "\n\n".join(
            f"**{r.title}**\n{r.description}\nSource: {r.url}" for r in results
        )
