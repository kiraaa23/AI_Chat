# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DocumentInfo(BaseModel):
    filename: str
    doc_id: str # Unique identifier for the document within a case

class CaseInfo(BaseModel):
    case_id: str
    documents: List[DocumentInfo] = Field(default_factory=list)

class UploadResponse(BaseModel):
    message: str
    case_id: str
    doc_ids: List[str]

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list)
    # Advanced options (optional, can be inferred by the backend)
    decompose_question: bool = False # Hint to attempt decomposition

class ChatResponse(BaseModel):
    answer: str
    relevant_chunks: List[str] # Show context used (for debugging/transparency)
    chat_history: List[ChatMessage]
    sub_questions_and_answers: Optional[Dict[str, str]] = None # If decomposition was used

class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    embedding_model_status: str