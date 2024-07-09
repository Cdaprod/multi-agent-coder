import os
import mesop as me
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from autogen import AssistantAgent, UserProxyAgent

# LangGraph setup

@tool
def search(query: str):
    """Searches the web and returns results."""
    return ["Search results for: " + query]

tools = [search]
tool_node = ToolNode(tools)

model = ChatAnthropic(model="claude-3-5", temperature=0).bind_tools(tools)

def should_continue(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "END"

def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", 'agent')

# AutoGen setup

config_list = [{"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")}]

assistant = AssistantAgent(name="assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent(name="user_proxy")

user_proxy.initiate_chat(
    assistant,
    message="Search the web for documentation for 'pip install mesop' to be able to create an app with it. Create a directory named app that is a python webpagr using googles mesop library. The app should have a route that returns a message that says 'Hello, world!' using the mesop components."
)
