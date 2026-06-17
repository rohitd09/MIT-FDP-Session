from fastapi import FastAPI
from App.service.ta_workflow_service import TAWorkflowService
from App.service.embedding_service import EmbeddingService

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