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
    def __init__(self, mcp_tools):
        ra_agent = ResearchAssistantAgent(mcp_tools=mcp_tools)
        agent_builder = StateGraph(MessageState)

        # Building Workflow Nodes
        agent_builder.add_node("research_node", ra_agent.call_llm)
        agent_builder.add_node("research_tool_node", ra_agent.tool_node)

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

    async def run_ra_workflow(self, query):
        result = await self.workflow.ainvoke(
            {
                "messages": [HumanMessage(content=query)],
                "llm_calls": 0
            },
            config={"recursion_limit": 6}
        )

        return result