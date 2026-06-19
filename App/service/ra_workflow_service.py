from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, HumanMessage

from typing_extensions import TypedDict, Annotated

import operator

from App.agents.ra_agent import ResearchAssistantAgent

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: MessageState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "use_tool"

    return "__end__"

class RAWorkflowService:
    def __init__(self):
        agent_builder = StateGraph(MessageState)

        # Building Workflow Nodes
        agent_builder.add_node("research_node", ResearchAssistantAgent().call_llm)
        agent_builder.add_node("research_tool_node", ResearchAssistantAgent().tool_node)

        # Adding edges between the nodes
        agent_builder.add_edge(START, "research_node")
        agent_builder.add_conditional_edges(
            "research_node",
            should_continue,
            {
                "use_tool": "research_tool_node",
                "__end__": END
            }
        )

        agent_builder.add_edge("research_tool_node", "research_node")

        self.workflow = agent_builder.compile()

    def run_ra_workflow(self, query):
        result = self.workflow.invoke(
            {
                "messages": [HumanMessage(content=query)]
            }
        )

        return result