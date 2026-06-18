from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.messages import SystemMessage, ToolMessage

from dotenv import load_dotenv

from App.service.embedding_service import EmbeddingService

load_dotenv()

# query = "What is Generative AI?"
# response = llm.invoke(query).content

# Building Custom LangChain Tool
@tool
def retrieve_data_from_pdf(query):
    """
    This tool is used to retireve data from a pdf file which answers questions.
    On C programming language, which is used in teaching the course in MIT AOE.
    Use this tool to gain information for C programming.
    """
    embedding_service = EmbeddingService()
    context = embedding_service.retrieve_from_pdf(query)
    return context

class QuestionGenerationAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=512
        )

        tools_list = [retrieve_data_from_pdf]
        self.llm_with_tools = self.llm.bind_tools(tools_list)
        self.tools_by_name = {tool.name: tool for tool in tools_list}

    def call_llm(self, state):        
        return {
            "messages": [
                self.llm_with_tools.invoke(
                    [
                        SystemMessage(
                    content="""
                    You are a helful Question Generation Agent of C Programming language.
                    You generate questions only using the curriculum of MIT AOE.
                    You have access to `retrieve_data_from_pdf` tool, which will let you get 
                    content about MIT AOE's C programming course. You have to ensure that the questions
                    you generate do not deviate from the syllabus. The questions are to be generated 
                    only for the specific modules mentioned in the syllabus.
                    Strictly maintain that you are a C programming question generation agent only.

                    The user shall provide either a topic name, which you would use to generate exactly 3 questions.
                    Or user shall provide just a number, in that case you shall think of C programming topics search for its
                    contents in the curriculum and generate exactly the number of questions asked.

                    Do not answer to any irrelevant questions. If asked, inform the user about your directive.

                    Output Format: 
                    1. [Question 1]
                    2. [Question 2]
                    3. [Question 3]
                    """
                        )
                    ]
                    + state['messages']
                )
            ],
            "llm_calls": state.get('llm_calls', 0) + 1
        }

    def tool_node(self, state):
        """Performs the tool calls"""

        result = []

        for tool_call in state['messages'][-1].tool_calls:

            if 'self' in tool_call["args"]:
                del tool_call["args"]['self']

            tool = self.tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])

            if isinstance(observation, list):
                content_string = "\n".join(observation)
            else:
                content_string = str(observation)

            result.append(
                ToolMessage(
                    content=content_string,
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": result}