# LangChain Ceramic

LangChain integration for [Ceramic](https://ceramic.ai) — a web search API built for LLMs.

## Installation

```bash
pip install langchain langchain-openai langchain-ceramic
```

## Setup

Generate an API key at [platform.ceramic.ai/keys](https://platform.ceramic.ai/keys), then export it:

```bash
export CERAMIC_API_KEY="your-api-key"
```

Also set up any additional API keys you will need, e.g., OpenAI via 
```bash 
export OPENAI_API_KEY="your-api-key"
```

## Example usage

### Tool calling
LangChain agents can use Ceramic search via tool calling to support their response with sources from the web.

Ceramic uses lexical (keyword-based) search. See [Best Practices](https://docs.ceramic.ai/api/search/best-practices) for information on how to use Ceramic Search most effectively. When calling Ceramic search via a tool call, the LLM automatically converts the natural language query into an optimized keyword-based query for search.

```python
from langchain_ceramic import CeramicSearch
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# Initialize the Ceramic search tool and retrieve a maximum of five results
ceramic_search = CeramicSearch(max_results=5)

# Initialize the agent with the Ceramic search tool
agent = create_agent(
    model=ChatOpenAI(model="gpt-5.5"), 
    tools=[ceramic_search],
    system_prompt="You are a helpful research assistant. Use web search to find accurate, up-to-date information."
)

# Generate a response using natural language queries
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about California rental laws."}]}
)
print(result["messages"][-1].content)
```

### RAG pipeline
Use the retriever tool `CeramicSearchRetriever` to obtain relevant documents for RAG pipelines.

Because Ceramic uses lexical search, we first convert the natural language query into keywords using an LLM before retrieval. The original natural language query is still passed through to the answer prompt.

```python
from langchain_ceramic import CeramicSearchRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Initialize the LLM and Ceramic Search Retriever
llm = ChatOpenAI(model="gpt-5.5")
retriever = CeramicSearchRetriever(k=5)

# Convert the natural language query to keywords before retrieval
keyword_prompt = PromptTemplate.from_template(
    "Rewrite the following question as a 2-10 word keyword query for a lexical search engine.\n"
    "Rules:\n"
    "- Extract specific entities, topics, locations, and dates\n"
    "- Replace conversational phrasing with concrete keywords\n"
    "- Include relevant synonyms explicitly when terminology is ambiguous\n"
    "- Keep word order meaningful\n"
    "Example good keyword queries: 2026 Super Bowl halftime performer, Serena Williams Grand Slam titles, California rent increase causes housing shortage 2025\n"
    "Return only the keyword query with no explanation.\n\n"
    "Question: {query}"
)
keyword_chain = keyword_prompt | llm | StrOutputParser()

# Format the prompt with the query and retrieved search context
answer_prompt = ChatPromptTemplate.from_template(
    "Answer the query based on the provided context.\n\nQuery: {query}\n\nContext: {context}"
)

# Create the complete chain, which involves keyword_chain and passes the formatted prompt to the LLM
# RunnablePassthrough() preserves the natural language query for the answer prompt
chain = (
    {"query": RunnablePassthrough(), "context": keyword_chain | retriever}
    | answer_prompt
    | llm
    | StrOutputParser()
)

# Generate the response
answer = chain.invoke("What are the latest AI chip export restrictions?")
print(answer)
```

Each retrieved `Document` has:
- `page_content`: the result description
- `metadata["title"]`: page title
- `metadata["url"]`: source URL

### Async usage

Both `CeramicSearchRetriever` and `CeramicSearch` support async:

```python
docs = await retriever.ainvoke("California rental laws")
```

## API reference

### `CeramicSearch`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | Ceramic API key (falls back to `CERAMIC_API_KEY` env var) |
| `max_results` | `int` | `5` | Maximum number of results to include in the response string |

### `CeramicSearchRetriever`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | Ceramic API key (falls back to `CERAMIC_API_KEY` env var) |
| `k` | `int` | `10` | Maximum number of results to return |