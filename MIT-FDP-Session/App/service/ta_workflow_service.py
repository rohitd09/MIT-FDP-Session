from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, HumanMessage

from typing_extensions import TypedDict, Annotated

import operator

from App.agents.ta_agent import TeachingAssistantAgent

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: MessageState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "use_tool"

    return "__end__"

class TAWorkflowService:
    def __init__(self):
        agent_builder = StateGraph(MessageState)

        # Building Workflow Nodes
        agent_builder.add_node("bot_node", TeachingAssistantAgent().call_llm)
        agent_builder.add_node("tool_node", TeachingAssistantAgent().tool_node)

        # Adding edges between the nodes
        agent_builder.add_edge(START, "bot_node")
        agent_builder.add_conditional_edges(
            "bot_node",
            should_continue,
            {
                "use_tool": "tool_node",
                "__end__": END
            }
        )

        agent_builder.add_edge("tool_node", "bot_node")

        self.workflow = agent_builder.compile()

    def run_ta_workflow(self, query):
        result = self.workflow.invoke(
            {
                "messages": [HumanMessage(content=query)]
            }
        )

        return result