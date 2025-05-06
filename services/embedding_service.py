import os
import logging
import numpy as np
from typing import List, Dict, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME, device: str = settings.DEVICE):
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded Sentence Transformer model '{model_name}' on device '{device}'. Embedding dim: {self.embedding_dim}")
            self.status = "OK"
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model '{model_name}': {e}", exc_info=True)
            self.model = None
            self.embedding_dim = None
            self.status = f"Error: {e}"

        self.vector_stores: Dict[str, Tuple[faiss.Index, List[dict]]] = {} # {case_id: (index, chunks_with_meta)}
        self.store_path_base = settings.VECTOR_STORE_PATH

    def get_status(self) -> str:
        return self.status

    def _get_store_paths(self, case_id: str) -> Tuple[str, str]:
        case_dir = os.path.join(self.store_path_base, case_id)
        os.makedirs(case_dir, exist_ok=True)
        index_path = os.path.join(case_dir, "vector_store.index")
        metadata_path = os.path.join(case_dir, "metadata.npy") # Using numpy for simplicity
        return index_path, metadata_path

    def load_vector_store(self, case_id: str):
        """Loads a FAISS index and metadata for a given case_id if it exists."""
        if case_id in self.vector_stores:
            return # Already loaded

        index_path, metadata_path = self._get_store_paths(case_id)

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                index = faiss.read_index(index_path)
                # Load metadata (adjust if using a different serialization method)
                chunks_with_meta = np.load(metadata_path, allow_pickle=True).tolist()
                if index.d != self.embedding_dim:
                     logger.error(f"Loaded index dimension ({index.d}) for case '{case_id}' does not match model dimension ({self.embedding_dim}). Index ignored.")
                     self._create_empty_store(case_id)
                else:
                    self.vector_stores[case_id] = (index, chunks_with_meta)
                    logger.info(f"Loaded vector store for case '{case_id}' with {index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Failed to load vector store for case '{case_id}': {e}. Creating new store.", exc_info=True)
                # If loading fails, create an empty store to avoid repeated errors
                self._create_empty_store(case_id)
        else:
            logger.info(f"No existing vector store found for case '{case_id}'. Creating new store.")
            self._create_empty_store(case_id)

    def save_vector_store(self, case_id: str):
        """Saves the FAISS index and metadata for a given case_id."""
        if case_id not in self.vector_stores:
            logger.warning(f"Attempted to save non-existent vector store for case '{case_id}'")
            return

        index_path, metadata_path = self._get_store_paths(case_id)
        index, chunks_with_meta = self.vector_stores[case_id]

        try:
            faiss.write_index(index, index_path)
            # Save metadata (using numpy here, consider json or pickle for more complex metadata)
            np.save(metadata_path, np.array(chunks_with_meta, dtype=object))
            logger.info(f"Saved vector store for case '{case_id}' with {index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Failed to save vector store for case '{case_id}': {e}", exc_info=True)


    def _create_empty_store(self, case_id: str):
         """Initializes an empty FAISS index and metadata list for a case."""
         if self.embedding_dim is None:
              logger.error("Cannot create vector store: Embedding model not loaded.")
              return
         # Using IndexFlatL2, suitable for many cases. Consider IndexIVFFlat for larger datasets.
         index = faiss.IndexFlatL2(self.embedding_dim)
         self.vector_stores[case_id] = (index, [])
         logger.info(f"Created new empty vector store for case '{case_id}'.")


    def add_chunks(self, case_id: str, chunks_with_meta: List[dict]):
        """Generates embeddings and adds chunks to the specified case's vector store."""
        if not self.model:
            logger.error("Embedding model not loaded. Cannot add chunks.")
            return False
        if not chunks_with_meta:
            logger.warning(f"No chunks provided to add for case '{case_id}'.")
            return False

        self.load_vector_store(case_id) # Ensure store is loaded or created

        index, existing_chunks = self.vector_stores[case_id]

        texts = [chunk["text"] for chunk in chunks_with_meta]
        try:
            logger.info(f"Generating embeddings for {len(texts)} chunks for case '{case_id}'...")
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            logger.info(f"Generated {len(embeddings)} embeddings.")

            # Ensure embeddings are float32, as required by FAISS
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            index.add(embeddings)
            existing_chunks.extend(chunks_with_meta) # Add new metadata

            self.vector_stores[case_id] = (index, existing_chunks) # Update store in memory
            self.save_vector_store(case_id) # Persist changes
            logger.info(f"Added {len(embeddings)} vectors to case '{case_id}'. Total vectors: {index.ntotal}")
            return True

        except Exception as e:
            logger.error(f"Error generating embeddings or adding to FAISS for case '{case_id}': {e}", exc_info=True)
            return False

    def search(self, case_id: str, query: str, k: int = 5) -> List[Tuple[float, dict]]:
        """Searches the vector store for the most relevant chunks."""
        if not self.model:
            logger.error("Embedding model not loaded. Cannot perform search.")
            return []
        if case_id not in self.vector_stores:
            self.load_vector_store(case_id) # Attempt to load if not in memory
            if case_id not in self.vector_stores: # Still not found after load attempt
                logger.warning(f"Vector store for case '{case_id}' not found.")
                return []

        index, chunks_with_meta = self.vector_stores[case_id]

        if index.ntotal == 0:
            logger.warning(f"Vector store for case '{case_id}' is empty.")
            return []

        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            if query_embedding.dtype != np.float32:
                 query_embedding = query_embedding.astype(np.float32)

            # Ensure k is not greater than the number of vectors in the index
            actual_k = min(k, index.ntotal)
            if actual_k == 0: return []

            distances, indices = index.search(query_embedding, actual_k)

            results = []
            for i in range(actual_k):
                idx = indices[0][i]
                distance = distances[0][i]
                if 0 <= idx < len(chunks_with_meta):
                     # Score could be 1/ (1 + distance) for L2, or simply the distance itself
                     score = float(distance) # Lower distance is better for L2
                     results.append((score, chunks_with_meta[idx]))
                else:
                    logger.warning(f"Search returned invalid index {idx} for case '{case_id}'. Max index: {len(chunks_with_meta)-1}")


            # Sort by score (distance - lower is better)
            results.sort(key=lambda x: x[0])

            logger.info(f"Search for '{query[:50]}...' in case '{case_id}' returned {len(results)} results.")
            return results

        except Exception as e:
            logger.error(f"Error during FAISS search for case '{case_id}': {e}", exc_info=True)
            return []

# Instantiate for easy import
embedding_service = EmbeddingService()
