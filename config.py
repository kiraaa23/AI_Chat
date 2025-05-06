# config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2" # Or another suitable Sentence Transformer
    LLM_MODEL_NAME: str = "llama3.2:latest" # Or the Ollama model you pulled (e.g., "mistral:latest")
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    VECTOR_STORE_PATH: str = "data/vector_stores"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TESSERACT_CMD: str = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Optional: Explicit path to tesseract executable if not in PATH
                                     # Example for Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    DEVICE: str = "cpu" # "cuda" if using GPU

    class Config:
        env_file = '.env' # Optional: Load from .env file
        extra = 'ignore'

settings = Settings()

# --- Tesseract Configuration ---
# Set Tesseract command path if specified in settings or environment variable
if settings.TESSERACT_CMD:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        print(f"Using Tesseract command: {pytesseract.pytesseract.tesseract_cmd}")
    except ImportError:
        print("pytesseract not installed, OCR functionality will be unavailable.")
    except Exception as e:
        print(f"Error setting tesseract command path: {e}")

# --- Ensure data directory exists ---
os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)