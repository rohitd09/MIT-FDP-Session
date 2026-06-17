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

# Building In-built LangChain Tool
web_search = DuckDuckGoSearchRun()

class PDFBot:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=256
        )

        tools_list = [retrieve_data_from_pdf, web_search]
        self.llm_with_tools = self.llm.bind_tools(tools_list)
        self.tools_by_name = {tool.name: tool for tool in tools_list}

    def call_llm(self, state):        
        return {
            "messages": [
                self.llm_with_tools.invoke(
                    [
                        SystemMessage(
                    content="""
                    You are a helful Teaching Assistant of C Programming language,
                    at MIT AOE, Pune.
                    You have access to `retrieve_data_from_pdf` tool, which will let you get 
                    content about MIT AOE's C programming course.
                    Any information not available is to be attained using the `web_search` tool.
                    Strictly maintain that you are a C programming teaching agent only. Do not answer 
                    to any irrelevant questions. If asked, inform the user about your directive.
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