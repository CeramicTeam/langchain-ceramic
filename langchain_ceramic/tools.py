from __future__ import annotations

import os
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ceramic_ai import AsyncCeramic, Ceramic


class CeramicSearchInput(BaseModel):
    query: str = Field(description="Keyword search query (2-10 words). Use specific entities, topics, locations, and dates — not natural language questions.")


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
        """
        Search the web using Ceramic.
        Use for accurate current information — news, prices, recent events, documentation, general fact checking.
        Returns up to 10 ranked results with titles, URLs, and descriptions.
        Ceramic matches exact keywords — it does not interpret natural language or synonyms automatically. Call Ceramic search with a keyword query version of the user's question.
        Keyword query conversion rules:
        - Queries must be 2-8 words
        - Extract specific entities, topics, locations, and dates
        - Replace conversational phrasing with concrete keywords
        - Do not include uninformative words such as articles (the, a, an). Avoid prepositions (on, about, in, for, of, at, by, with) unless they are within established phrases or names (United States of America, Into the Wild).
        - Include relevant synonyms explicitly when terminology is ambiguous
        - Keep word order meaningful (`house cat` and `cat house` return different results)
        - Good keyword query examples:
            - "2026 Super Bowl halftime performer"
            - "climate change effects global warming impact"
            - "beginner investing strategies stocks bonds basics"
        If the search returns no useful results, retry with a more specific keyword query.
        """
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
