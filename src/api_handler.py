from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from rag_app.query_model import query_model
from models import APIResponse


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","*"],
    allow_credentials=True,
    allow_methods=["GET","POST","PUT","DELETE"],
    allow_headers=["*"], 
)


@app.get("/")
def index():
    return {"Hello World": "Python Server activated"}

@app.post("/query_ai", response_model=APIResponse)
def query_docs_endpoint(query_text: str):
    response = query_model(query_text)
    # create api response
    apiResponse = APIResponse(success=True, message="API Request is successfull", data=response)
    # Handle error in response
    if len(response.sources) == 0:
        apiResponse.success = False
        apiResponse.message = "API Request not successfull"
    return apiResponse


if __name__ == "__main__":
    # Run this as a server directly
    port = 8000
    print(f"Running the Py-server on port {port}")
    uvicorn.run("api_handler:app",host="127.0.0.1",port=port)