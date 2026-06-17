from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

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

from langchain.messages import SystemMessage, ToolMessage
class PDFBot:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=256
        )

        tools_list = [retrieve_data_from_pdf, web_search]
        self.llm_with_tools = self.llm.bind_tools(tools_list)

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