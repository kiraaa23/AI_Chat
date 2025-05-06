import os
import uuid
import logging
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, Path, Body, Depends
from fastapi.middleware.cors import CORSMiddleware

from models import (
    UploadResponse, ChatRequest, ChatResponse, HealthResponse,
    ChatMessage, CaseInfo, DocumentInfo
)

from services.document_processor import doc_processor
from services.embedding_service import embedding_service
from services.llm_service import llm_service
from services.chat_service import chat_service
from config import settings # Ensure config runs early

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced Document Chat Assistant",
    description="API for uploading documents and chatting with an AI about their content.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Allows requests from frontend applications (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify frontend origin e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory State (Replace with DB for production) ---
# Store basic info about cases and their documents
case_data: Dict[str, CaseInfo] = {}

# --- Dependency for Case Validation ---
async def get_case_info(case_id: str = Path(...)) -> CaseInfo:
    # Ensure vector store exists/is loaded for the case
    # This doesn't strictly validate existence *before* upload, but ensures
    # the embedding service is aware of the case_id for subsequent operations.
    embedding_service.load_vector_store(case_id)

    # Retrieve or create case metadata
    if case_id not in case_data:
        # If not in our metadata store, but maybe store exists on disk from previous run
        # We create metadata entry. A more robust system would sync disk/DB state.
        logger.info(f"Case '{case_id}' not found in memory, creating metadata entry.")
        case_data[case_id] = CaseInfo(case_id=case_id)
        # Potentially list files in the case's data directory if needed
    return case_data[case_id]

# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Management"])
async def health_check():
    """Performs health checks on the service and its dependencies."""
    ollama_conn, ollama_status = llm_service.check_connection()
    embed_status = embedding_service.get_status()
    overall_status = "ok" if ollama_conn and embed_status == "OK" else "error"

    return HealthResponse(
        status=overall_status,
        ollama_status=ollama_status,
        embedding_model_status=embed_status
    )

@app.post("/upload/{case_id}", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(
    case_info: CaseInfo = Depends(get_case_info), # Validates/loads case
    files: List[UploadFile] = File(...)
):
    """Uploads one or more documents (PDF, DOCX, PNG, JPG) to a specific case."""
    case_id = case_info.case_id
    processed_doc_ids = []
    errors = []

    for file in files:
        if not file.filename:
             errors.append({"filename": "N/A", "error": "File has no filename."})
             continue

        logger.info(f"Processing file: {file.filename} for case: {case_id}")
        try:
            # Generate a unique ID for this document within the case
            doc_id = f"{uuid.uuid4()}"
            content = await file.read()

            # 1. Extract Text
            text, error = doc_processor.process_document(file.filename, content)
            if error:
                errors.append({"filename": file.filename, "error": error})
                continue
            if not text:
                 errors.append({"filename": file.filename, "error": "No text could be extracted."})
                 continue

            # 2. Chunk Text
            chunks = doc_processor.chunk_text(text, file.filename, doc_id)
            if not chunks:
                 errors.append({"filename": file.filename, "error": "Text extracted but resulted in no chunks."})
                 continue

            # 3. Add to Vector Store
            success = embedding_service.add_chunks(case_id, chunks)
            if success:
                processed_doc_ids.append(doc_id)
                # Add document info to our case metadata
                case_info.documents.append(DocumentInfo(filename=file.filename, doc_id=doc_id))
                logger.info(f"Successfully processed and added '{file.filename}' (doc_id: {doc_id}) to case '{case_id}'.")
            else:
                 errors.append({"filename": file.filename, "error": "Failed to add document chunks to vector store."})

        except Exception as e:
            logger.error(f"Failed to process file {file.filename} for case {case_id}: {e}", exc_info=True)
            errors.append({"filename": file.filename, "error": f"Internal server error during processing: {e}"})
        finally:
            await file.close() # Ensure file handle is closed

    if not processed_doc_ids and errors:
         # If all files failed
         raise HTTPException(status_code=400, detail={"message": "Document processing failed for all files.", "errors": errors})

    response_message = f"Processed {len(processed_doc_ids)} out of {len(files)} files for case '{case_id}'."
    if errors:
         response_message += f" Encountered {len(errors)} errors."
         logger.warning(f"Upload errors for case {case_id}: {errors}")
         # Return 207 Multi-Status if some succeeded and some failed
         # Note: FastAPI doesn't have a built-in 207, common practice is often 200 with details
         # For simplicity here, we return 200 but include error details.
         # Consider raising HTTPException with status 207 if strict adherence is needed.

    return UploadResponse(message=response_message, case_id=case_id, doc_ids=processed_doc_ids)


@app.post("/chat/{case_id}", response_model=ChatResponse, tags=["Chat"])
async def handle_chat(
    request: ChatRequest,
    case_info: CaseInfo = Depends(get_case_info), # Validates/loads case
):
    """Handles a chat message, retrieves context, and generates an AI response."""
    case_id = case_info.case_id

    # Get current history (or initialize if needed) - ChatService manages this internally now
    current_history = chat_service.get_chat_history(case_id)

    # If user provided history in request, use it (e.g., for stateless clients)
    # Otherwise, rely on server-side history managed by ChatService.
    # Logic could be added here to merge/prioritize if needed. For now, use ChatService's state.
    # If request.chat_history is provided, maybe replace server state? Be careful.

    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = await chat_service.process_chat_message(
            case_id=case_id,
            question=request.question,
            current_chat_history=current_history, # Pass the history known to the server
            attempt_decomposition=request.decompose_question # Pass hint from request
        )
        return response
    except Exception as e:
        logger.error(f"Error during chat processing for case {case_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during chat: {e}")


@app.get("/history/{case_id}", response_model=List[ChatMessage], tags=["Chat"])
async def get_history(
     case_info: CaseInfo = Depends(get_case_info), # Validates/loads case
):
    """Retrieves the chat history for a specific case."""
    return chat_service.get_chat_history(case_info.case_id)

@app.delete("/history/{case_id}", status_code=204, tags=["Chat"])
async def clear_history(
     case_info: CaseInfo = Depends(get_case_info), # Validates/loads case
):
     """Clears the chat history for a specific case."""
     case_id = case_info.case_id
     if case_id in chat_service.chat_histories:
         del chat_service.chat_histories[case_id]
         logger.info(f"Cleared chat history for case '{case_id}'.")
     else:
          logger.info(f"No chat history found for case '{case_id}' to clear.")
     # Return No Content
     return None

@app.get("/cases", response_model=List[CaseInfo], tags=["Management"])
async def list_cases():
    """Lists all known cases (from in-memory store)."""
    # Potentially enhance by scanning the VECTOR_STORE_PATH directory
    # for case_ids that might exist on disk but not in memory.
    disk_cases = set(os.listdir(settings.VECTOR_STORE_PATH))
    for case_id in disk_cases:
        if case_id not in case_data and os.path.isdir(os.path.join(settings.VECTOR_STORE_PATH, case_id)):
             # Add cases found on disk but not in memory metadata
             case_data[case_id] = CaseInfo(case_id=case_id)
             # Could try loading document list from metadata file if it exists
    return list(case_data.values())

@app.get("/documents/{case_id}", response_model=List[DocumentInfo], tags=["Documents"])
async def get_case_documents(
    case_info: CaseInfo = Depends(get_case_info), # Validates/loads case
):
    """Retrieves the list of documents associated with a specific case."""
    return case_info.documents


# --- Optional: Application Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    # Pre-load models or perform other initializations if needed
    # (Embedding model is loaded on EmbeddingService init, Ollama checked on first request)
    logger.info("Checking Ollama connection...")
    connected, status = llm_service.check_connection()
    if not connected:
        logger.warning(f"Ollama connection check failed on startup: {status}")
    else:
        logger.info(f"Ollama connection successful: {status}")
    if embedding_service.get_status() != "OK":
         logger.warning(f"Embedding model status on startup: {embedding_service.get_status()}")

    # Load existing case metadata and vector stores?
    # This could be slow if there are many/large cases. Done lazily via get_case_info now.
    logger.info("Advanced Document Chat Assistant is ready.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    # Perform cleanup, e.g., explicitly save any in-memory state if needed
    # (Vector stores are saved after each addition currently)

if __name__ == "__main__":
    import uvicorn
    # Run this file using: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    # Or directly: python main.py (if uvicorn is run programmatically below)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
