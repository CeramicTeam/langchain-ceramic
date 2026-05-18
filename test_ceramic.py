"""
Test script for langchain-ceramic.

Demonstrates two use cases:
  1. CeramicRetriever in a RAG chain (with Claude as the LLM)
  2. CeramicSearch as a tool in a LangGraph ReAct agent

Ceramic uses lexical (keyword) search, not semantic search. Queries must be
rewritten into 2-10 word keyword form before being sent to the API.

Requires:
  CERAMIC_API_KEY   - from platform.ceramic.ai/keys
  ANTHROPIC_API_KEY - from console.anthropic.com
"""

import asyncio
import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic

def _load_env_local():
    env_file = Path(__file__).parent / ".env-local"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

_load_env_local()
from langchain_ceramic import CeramicRetriever, CeramicSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.agents import create_agent

QUERY = "What are the latest AI chip export restrictions?"

# Ceramic is lexical — queries must be rewritten into keywords first.
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite the following question as a 2-10 word keyword query for a lexical "
    "search engine. Extract specific entities, topics, locations, and dates. "
    "Return ONLY the keyword query — no explanation, no punctuation.\n\n"
    "Question: {question}"
)

# Tool description that tells the agent to keyword-rewrite before calling.
SEARCH_DESCRIPTION = (
    "Search the web using Ceramic. "
    "Ceramic uses lexical (keyword) search — it does NOT interpret natural language. "
    "Before calling this tool, rewrite the user's question as a 2-10 word keyword "
    "query: extract specific entities, topics, locations, and dates; replace "
    "conversational phrasing with concrete keywords (e.g. 'AI chip export "
    "restrictions 2025' not 'What are the latest AI chip export restrictions?'). "
    "Input must be a keyword query string."
)

# System prompt for the agent: keyword rewriting + citation format.
AGENT_SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to a web search tool.\n\n"
    "When using ceramic_search:\n"
    "- Rewrite the user's question as a 2-10 word keyword query before calling the tool\n"
    "- Extract specific entities, topics, locations, and dates\n"
    "- Replace conversational phrasing with concrete keywords\n\n"
    "After receiving results, write a concise answer and list only the sources "
    "whose descriptions contributed to the answer:\n\n"
    "**Sources**\n"
    "1. [Title](url)\n"
    "2. [Title](url)"
)


# ---------------------------------------------------------------------------
# 1. RAG chain: query rewrite → CeramicRetriever → Claude
# ---------------------------------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata['title']}]({doc.metadata['url']})\n{doc.page_content}"
        for doc in docs
    )


def run_rag_chain():
    print("\n=== RAG Chain (CeramicRetriever) ===")

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    retriever = CeramicRetriever(k=5)

    # Step 1: rewrite the natural-language question into keywords
    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

    # Step 2: retrieve using the keyword query, then answer with citations
    answer_prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following search results. "
        "End your response with a **Sources** section listing each source you used "
        "as a numbered markdown link.\n\n"
        "{context}\n\n"
        "Question: {question}"
    )

    def retrieve_with_rewrite(question: str):
        keyword_query = rewrite_chain.invoke({"question": question})
        print(f"  Keyword query: {keyword_query!r}")
        docs = retriever.invoke(keyword_query)
        return format_docs(docs)

    chain = (
        {"context": RunnableLambda(retrieve_with_rewrite), "question": RunnablePassthrough()}
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(QUERY)
    print(answer)


# ---------------------------------------------------------------------------
# 2. ReAct agent: CeramicSearch tool + Claude
# ---------------------------------------------------------------------------

def run_agent():
    print("\n=== ReAct Agent (CeramicSearch) ===")

    llm = ChatAnthropic(model="claude-sonnet-4-6")

    # Override description so Claude knows to keyword-rewrite before calling.
    tool = CeramicSearch(max_results=5)
    tool.description = SEARCH_DESCRIPTION

    agent = create_agent(llm, tools=[tool], system_prompt=AGENT_SYSTEM_PROMPT)

    result = agent.invoke({"messages": [{"role": "user", "content": QUERY}]})
    print(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# 3. Async retriever (bonus)
# ---------------------------------------------------------------------------

async def run_async_retriever():
    print("\n=== Async CeramicRetriever ===")

    llm = ChatAnthropic(model="claude-sonnet-4-6")
    retriever = CeramicRetriever(k=3)

    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    keyword_query = await rewrite_chain.ainvoke({"question": QUERY})
    print(f"  Keyword query: {keyword_query!r}")

    docs = await retriever.ainvoke(keyword_query)
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.metadata['title']}")
        print(f"   {doc.metadata['url']}")
        print(f"   {doc.page_content[:120]}...")
        print()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    missing = [v for v in ("CERAMIC_API_KEY", "ANTHROPIC_API_KEY") if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")

    run_rag_chain()
    run_agent()
    asyncio.run(run_async_retriever())
