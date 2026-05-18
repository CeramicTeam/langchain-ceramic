# LangChain Ceramic

LangChain integration for [Ceramic](https://ceramic.ai) — a web search API built for LLMs.

## Installation

```bash
pip install langchain-ceramic
```

## Setup

Generate an API key at [platform.ceramic.ai/keys](https://platform.ceramic.ai/keys), then export it:

```bash
export CERAMIC_API_KEY="your-api-key"
```

Or pass it directly when constructing the retriever or tool.

## Usage

### CeramicRetriever in a RAG chain

```python
from langchain_ceramic import CeramicRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

retriever = CeramicRetriever(k=5)  # reads CERAMIC_API_KEY from env

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-5.5")
    | StrOutputParser()
)

answer = chain.invoke("AI chip export restrictions 2025")
print(answer)
```

Each retrieved `Document` has:
- `page_content`: the result description
- `metadata["title"]`: page title
- `metadata["url"]`: source URL

### CeramicSearch in an agent

```python
from langchain_ceramic import CeramicSearch
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

tools = [CeramicSearch(max_results=5)]
agent = create_agent(ChatOpenAI(model="gpt-5.5"), tools=tools)

result = agent.invoke({"messages": [{"role": "user", "content": "Find recent news about GLP-1 drugs"}]})
print(result["messages"][-1].content)
```

### Async usage

Both `CeramicRetriever` and `CeramicSearch` support async:

```python
docs = await retriever.ainvoke("California rental laws")
```

## API reference

### `CeramicRetriever`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | Ceramic API key (falls back to `CERAMIC_API_KEY` env var) |
| `k` | `int` | `10` | Maximum number of results to return |

### `CeramicSearch`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | Ceramic API key (falls back to `CERAMIC_API_KEY` env var) |
| `max_results` | `int` | `5` | Maximum number of results to include in the response string |
