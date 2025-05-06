import os
import logging
from typing import List, Tuple
from io import BytesIO

import pypdf
import docx2txt
from PIL import Image
import pytesseract # Ensure tesseract is installed and configured in config.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract path if needed (done globally in config.py now)
# if settings.TESSERACT_CMD:
#     pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

class DocumentProcessor:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        try:
            reader = pypdf.PdfReader(BytesIO(file_content))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info(f"Extracted {len(text)} characters from PDF.")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
            return ""

    def extract_text_from_docx(self, file_content: bytes) -> str:
        try:
            text = docx2txt.process(BytesIO(file_content))
            logger.info(f"Extracted {len(text)} characters from DOCX.")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}", exc_info=True)
            return ""

    def extract_text_from_image(self, file_content: bytes) -> str:
        try:
            img = Image.open(BytesIO(file_content))
            text = pytesseract.image_to_string(img)
            logger.info(f"Extracted {len(text)} characters from Image using OCR.")
            return text
        except ImportError:
             logger.error("pytesseract not installed. Cannot process images.")
             return "[OCR Error: pytesseract not installed]"
        except Exception as e:
            logger.error(f"Error extracting text from image with OCR: {e}", exc_info=True)
            # Check if Tesseract is installed and in PATH or if path is correctly set
            if "Tesseract is not installed or isn't in your PATH" in str(e):
                 logger.error("Tesseract OCR engine not found. Please install it and ensure it's in your system's PATH or configure TESSERACT_CMD in config.py.")
            return f"[OCR Error: {e}]"


    def process_document(self, filename: str, file_content: bytes) -> Tuple[str, str]:
        """
        Detects file type and extracts text.
        Returns: (extracted_text, error_message | None)
        """
        _, ext = os.path.splitext(filename.lower())
        text = ""
        error = None

        if ext == ".pdf":
            text = self.extract_text_from_pdf(file_content)
        elif ext in [".docx"]:
            text = self.extract_text_from_docx(file_content)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            text = self.extract_text_from_image(file_content)
        else:
            error = f"Unsupported file type: {ext}"
            logger.warning(error)
            return "", error

        if not text and not error:
             error = f"Could not extract text from {filename} (empty or extraction failed)."
             logger.warning(error)

        return text, error

    def chunk_text(self, text: str, filename: str, doc_id: str) -> List[dict]:
        """Chunks text and adds metadata."""
        if not text:
            return []
        chunks = self.text_splitter.split_text(text)
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
             chunk_dicts.append({
                 "text": chunk,
                 "metadata": {
                     "source": filename,
                     "doc_id": doc_id,
                     "chunk_index": i
                 }
             })
        logger.info(f"Split text from {filename} into {len(chunk_dicts)} chunks.")
        return chunk_dicts
    
    # Instantiate for easy import
doc_processor = DocumentProcessor()