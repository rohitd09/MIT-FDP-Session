import asyncio

from langchain_groq import ChatGroq
# from langchain_core.tools import tool
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain.messages import SystemMessage, ToolMessage

from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv

from App.service.embedding_service import EmbeddingService

load_dotenv()

# query = "What is Generative AI?"
# response = llm.invoke(query).content

# Building Custom LangChain Tool
# @tool
# def retrieve_data_from_pdf(query):
#     """
#     This tool is used to retireve data from a pdf file which answers questions.
#     On C programming language, which is used in teaching the course in MIT AOE.
#     Use this tool to gain information for C programming.
#     """
#     embedding_service = EmbeddingService()
#     context = embedding_service.retrieve_from_pdf(query)
#     return context

# # Building In-built LangChain Tool
# web_search = DuckDuckGoSearchRun()

class ResearchAssistantAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.7,
            max_tokens=1024
        )

        # tools_list = [retrieve_data_from_pdf, web_search]
        # self.llm_with_tools = self.llm.bind_tools(tools_list)
        # self.tools_by_name = {tool.name: tool for tool in tools_list}

        self.tools_list = []
        self.tools_by_name = {}
        self.llm_with_tools = None

        self._initialize_mcp_tools()

    def _initialize_mcp_tools(self):
        mcp_server_url = "http://localhost:8080/sse/"

        try:
            async def fetch_tools():
                client = MultiServerMCPClient({
                    "research_server": {
                        "transport": "sse",
                        "url": mcp_server_url
                    }
                })

                return await client.get_tools()

            self.tools_list = asyncio.run(fetch_tools())

            if not self.tools_list:
                raise ValueError("MCP Server Connected, No tools returned")
            
            self .tools_by_name = {tool.name: tool for tool in self.tools_list}
            self.llm_with_tools = self.llm.bind_tools(self.tools_list)
            print(f"Connected to MCP Server, fetched {len(self.tools_list)} tools.")

        except Exception as e:
            raise RuntimeError(
                f"Could not initialize MCP Client at {mcp_server_url}.\n"
                f"Reason: {e}"
            )



    def call_llm(self, state):        
        return {
            "messages": [
                self.llm_with_tools.invoke(
                    [
                        SystemMessage(
                    content="""
                    You are an expert Research Assistant.
    
                    Your task is to look for complex topics, reference from academic and online
                    databases and generate high quality summaries that can help me start writing
                    my literature review.

                    Follow this process:

                    1. You have tools that can help with web searches. Use them to find links or
                    extract detailed web content. Refer to blogs, articles & documentations.

                    2. You have access to research repositories as well, use them to extract paper 
                    content and generate educated summaries.

                    3. Compare all the found content, create technical breakdown, compare introductions,
                    to provide the said output.

                    Rely ONLY on your provided tools for real-world factual claims.
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
            observation = asyncio.run(tool.ainvoke(tool_call["args"]))

            # if isinstance(observation, list):
            #     content_string = "\n".join(observation)
            # else:
            #     content_string = str(observation)
            content_string = str(observation)
            
            result.append(
                ToolMessage(
                    content=content_string,
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": result}