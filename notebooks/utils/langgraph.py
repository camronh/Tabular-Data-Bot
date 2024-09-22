from typing import Annotated, Sequence, TypedDict
from utils.data_collection import MovieSearchTool

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor, ToolInvocation, ToolNode, tools_condition
from langgraph.graph import END, StateGraph, START

from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    output: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: list[dict]


system_message = SystemMessage(
    content="""You are a helpful assistant that can answer questions about movies. Only answer based on the context provided when using the search tool. \
If the context does not contain the answer, apologize and say so. Otherwise, provide the answer concisely and in a friendly tone.

Consider the number of votes to be how popular a movie is""")


class Nodes:
    def __init__(self, tools):
        self.tools = tools
        self.tool_executor = ToolExecutor(tools)

    def call_model(self, state: AgentState):
        llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0).bind_tools(self.tools)

        response = llm.invoke([system_message] + state["messages"])

        if not response.tool_calls:
            return {"output": response.content, "messages": [response]}

        return {"messages": [response]}


def make_graph(tools):

    nodes = Nodes(tools)
    tools_node = ToolNode(tools, handle_tool_errors=False)

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("Agent", nodes.call_model)  # agent
    workflow.add_node("tools", tools_node)  # retrieval

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "Agent", tools_condition
    )

    workflow.add_edge("tools", "Agent")

    workflow.set_entry_point("Agent")

    # Compile
    graph = workflow.compile()

    return graph


def create_agent():

    movies_tool = MovieSearchTool()

    graph = make_graph([movies_tool])

    return graph
