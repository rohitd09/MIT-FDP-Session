from fastapi import FastAPI

from App.service.ta_workflow_service import TAWorkflowService
from App.service.embedding_service import EmbeddingService
from App.service.homework_workflow_service import HomeworkWorkflowService
from App.service.ra_workflow_service import RAWorkflowService

app = FastAPI(title="MIT Assistant", version="0.1")

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
def generate_research_summary(query):
    workflow = RAWorkflowService()
    result = workflow.run_ra_workflow(query)

    return {"Workflow Response": result}