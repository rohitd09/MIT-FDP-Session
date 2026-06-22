import httpx
from fastapi import FastAPI
from contextlib import asynccontextmanager

from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

from App.service.ta_workflow_service import TAWorkflowService
from App.service.embedding_service import EmbeddingService
from App.service.homework_workflow_service import HomeworkWorkflowService
from App.service.ra_workflow_service import RAWorkflowService

@asynccontextmanager
async def lifespan(app: FastAPI):
    mcp_base_url = "https://research-mcp-server-32764074468.asia-south1.run.app/mcp"
    print(f"Connecting directly to remote HTTP MCP server at {mcp_base_url}...")

    app.state.mcp_tools = []
    app.state.mcp_session = None
    app.state.mcp_transport_cm = None
    app.state.mcp_session_cm = None

    try:
        transport_cm = streamable_http_client(mcp_base_url)
        read_stream, write_stream, get_session_id = await transport_cm.__aenter__()

        session_cm = ClientSession(read_stream, write_stream)
        session = await session_cm.__aenter__()

        await session.initialize()

        app.state.mcp_transport_cm = transport_cm
        app.state.mcp_session_cm = session_cm
        app.state.mcp_session = session
        app.state.mcp_get_session_id = get_session_id

        response = await session.list_tools()
        raw_tools = response.tools if hasattr(response, "tools") else []

        app.state.mcp_tools = [
            convert_mcp_tool_to_langchain_tool(session, t)
            for t in raw_tools
        ]

        print(f"Successfully loaded {len(app.state.mcp_tools)} tools from Cloud Run!")
        print(f"MCP session id: {get_session_id()}")

    except Exception as e:
        print(f"CRITICAL: Failed to connect to Cloud Run MCP server: {e}")
        import traceback
        traceback.print_exc()
        app.state.mcp_tools = []

    try:
        yield
    finally:
        if app.state.mcp_session_cm is not None:
            await app.state.mcp_session_cm.__aexit__(None, None, None)
        if app.state.mcp_transport_cm is not None:
            await app.state.mcp_transport_cm.__aexit__(None, None, None)
        print("Shutting down application backend...")

app = FastAPI(title="MIT Assistant", version="0.1", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Welcome to the MIT Assistant API"}

@app.get("/test-api")
def test_app():
    return {"message": "New endpoint created"}

@app.get("/run-llm/{query}")
def run_llm(query):
    workflow = TAWorkflowService()
    result = workflow.run_ta_workflow(query)
    
    return {"Workflow Response": result}

@app.get("/embed_pdf")
def embed_pdf():
    embedding_service = EmbeddingService(
        collection_name="teaching_assistant_collection",
        persist_directory="./mit_aoe_db"
    )

    embedding_service.process_pdf_document()
    return {"message": "PDF embedded successfully"}

@app.get("/generate-homework/{query}")
def generate_homework(query):
    workflow = HomeworkWorkflowService()
    result = workflow.run_homework_workflow(query)
    
    return {"Workflow Response": result}

@app.get("/generate_research_summary/{query}")
async def generate_research_summary(query: str):
    print("1. endpoint entered")
    tools = app.state.mcp_tools
    print(f"2. tools count = {len(tools)}")
    workflow = RAWorkflowService(mcp_tools=tools)
    print("3. workflow created")
    result = await workflow.run_ra_workflow(query)
    print("4. workflow finished")
    return {"Workflow Response": result}