# Agent

Lets whip up a quick langgraph agent that loops with a single tool.

```python
# Reload
%load_ext autoreload
%autoreload 2

```

```python
%pip install -q langgraph langchain-openai langchain-core


```

```python
from utils.data_collection import load_df, MovieSearchTool

# df = load_df("../raw/top_50000.pkl")

```

```python
movie_data_tool = MovieSearchTool()

```

```python
movie_data_tool.semantic_search("Matrix", k=3)

```

```python
from utils.langgraph import make_graph

graph = make_graph([movie_data_tool])
```

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
results = graph.invoke(input={"messages": [HumanMessage(content="When was the matrix reloaded made?")]})
results["messages"][-1].content
```

```python
results["messages"][-1]
```

```python

messages = results["messages"]
# Remove last message
messages = messages[:-1]

messages

```

```python

```

