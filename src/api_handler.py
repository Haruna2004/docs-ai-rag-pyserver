from fastapi import FastAPI
import uvicorn
from rag_app.query_model import query_model
from models import APIResponse

app = FastAPI()


@app.get("/")
def index():
    return {"Hello": "World"}

@app.post("/query_docs_ai", response_model=APIResponse)
def query_docs_endpoint(query_text: str):
    response = query_model(query_text)
    return APIResponse(success=True, message="AI answered successfully", data=response)


if __name__ == "__main__":
    # Run this as a server directly
    port = 8000
    print(f"Running the Py-server on port {port}")
    uvicorn.run("api_handler:app",host="127.0.0.1",port=port)