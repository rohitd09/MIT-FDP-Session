from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, HumanMessage

from typing_extensions import TypedDict, Annotated

import operator

from App.agents.ta_agent import TeachingAssistantAgent
from App.agents.question_agent import QuestionGenerationAgent

class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def should_continue(state: MessageState):
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "use_tool"

    return "__end__"

class HomeworkWorkflowService:
    def __init__(self):
        agent_builder = StateGraph(MessageState)

        # Building Workflow Nodes
        agent_builder.add_node("answering_node", TeachingAssistantAgent().call_llm)
        agent_builder.add_node("answering_tool_node", TeachingAssistantAgent().tool_node)
        agent_builder.add_node("question_generation_node", QuestionGenerationAgent().call_llm)
        agent_builder.add_node("question_generation_tool_node", QuestionGenerationAgent().tool_node)

        # Adding edges between the nodes
        agent_builder.add_edge(START, "question_generation_node")
        agent_builder.add_conditional_edges(
            "question_generation_node",
            should_continue,
            {
                "use_tool": "question_generation_tool_node",
                "__end__": "answering_node"
            }
        )

        agent_builder.add_edge("question_generation_tool_node", "question_generation_node")
    
        agent_builder.add_conditional_edges(
            "answering_node",
            should_continue,
            {
                "use_tool": "answering_tool_node",
                "__end__": END
            }
        )

        agent_builder.add_edge("answering_tool_node", "answering_node")

        self.workflow = agent_builder.compile()

    def run_homework_workflow(self, query):
        result = self.workflow.invoke(
            {
                "messages": [HumanMessage(content=query)]
            }
        )

        return result