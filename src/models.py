from pydantic import BaseModel
from typing import List,Optional

class Source(BaseModel):
    source_link: Optional[str]
    relevance: float

class ResponseData(BaseModel):
    content: str
    sources: List[Source]
    total_tokens: int

class APIResponse(BaseModel):
    success: bool 
    message: str 
    data: Optional[ResponseData]