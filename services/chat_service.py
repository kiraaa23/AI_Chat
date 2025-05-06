import logging
from typing import List, Dict, Optional, Tuple

from models import ChatMessage, ChatResponse
from services.embedding_service import embedding_service
from services.llm_service import llm_service
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Context Management ---
MAX_CONTEXT_TOKENS = 3500 # Rough estimate, adjust based on LLM's context window and typical chunk size/embedding overhead
# Simple token counter (approximation)
def count_tokens(text: str) -> int:
    return len(text.split())

class ChatService:

    def __init__(self):
        # In-memory store for chat histories (replace with DB for persistence)
        self.chat_histories: Dict[str, List[ChatMessage]] = {}

    def get_chat_history(self, case_id: str) -> List[ChatMessage]:
        return self.chat_histories.get(case_id, []).copy() # Return copy

    def _add_to_history(self, case_id: str, role: str, content: str):
        if case_id not in self.chat_histories:
            self.chat_histories[case_id] = []
        self.chat_histories[case_id].append(ChatMessage(role=role, content=content))
        # Optional: Limit history size
        # MAX_HISTORY_LEN = 20
        # self.chat_histories[case_id] = self.chat_histories[case_id][-MAX_HISTORY_LEN:]

    def _build_context(self, relevant_chunks_with_scores: List[Tuple[float, dict]]) -> str:
        """Builds context string, managing token limits."""
        context_str = ""
        current_tokens = 0
        added_sources = set()
        included_chunks_texts = []

        # Sort by relevance score (lower distance is better for L2)
        # relevant_chunks_with_scores.sort(key=lambda x: x[0]) # Already sorted by search

        for score, chunk_data in relevant_chunks_with_scores:
            text = chunk_data['text']
            source = chunk_data['metadata'].get('source', 'Unknown')
            chunk_idx = chunk_data['metadata'].get('chunk_index', -1)
            doc_id = chunk_data['metadata'].get('doc_id', 'N/A')

            chunk_tokens = count_tokens(text)
            header = f"---\nSource: {source} (DocID: {doc_id}, Chunk: {chunk_idx}, Score: {score:.4f})\n"
            header_tokens = count_tokens(header)

            if current_tokens + chunk_tokens + header_tokens <= MAX_CONTEXT_TOKENS:
                context_str += header + text + "\n"
                current_tokens += chunk_tokens + header_tokens
                added_sources.add(f"{source} (DocID: {doc_id})")
                included_chunks_texts.append(f"Chunk {chunk_idx} from {source}: {text[:100]}...") # For response transparency
            else:
                logger.warning(f"Context limit ({MAX_CONTEXT_TOKENS} tokens) reached. Skipping further chunks.")
                break # Stop adding chunks if limit exceeded

        if not context_str:
             return "[No relevant context found]", []

        final_context = f"Use the following context from documents ({', '.join(sorted(list(added_sources)))}) to answer the question. If the context doesn't contain the answer, say you don't know.\n\n{context_str}"
        logger.info(f"Built context with {current_tokens} tokens from {len(included_chunks_texts)} chunks.")
        return final_context, included_chunks_texts # Return context and the list of texts used

    def _generate_response_with_context(self, question: str, case_id: str, chat_history: List[ChatMessage], k: int = 5) -> Tuple[str, List[str]]:
        """Helper to retrieve context and generate response for a single question."""
        logger.info(f"Retrieving context for question: '{question[:50]}...' in case '{case_id}'")
        # 1. Retrieve relevant chunks
        relevant_chunks_with_scores = embedding_service.search(case_id, question, k=k)

        if not relevant_chunks_with_scores:
            logger.warning("No relevant documents found.")
            # Ask LLM without specific context, maybe it can answer from history or general knowledge
            context_str = "[No relevant context found in documents for this question]"
            included_chunk_texts = []
        else:
            # 2. Build context string, managing token limit
            context_str, included_chunk_texts = self._build_context(relevant_chunks_with_scores)

        # 3. Construct prompt for LLM
        prompt = f"{context_str}\n\nUser Question: {question}"

        # 4. Generate response using LLM (with history)
        logger.info(f"Generating LLM response for: '{question[:50]}...'")
        answer = llm_service.generate_response(prompt, chat_history=chat_history) or "[LLM failed to generate a response]"

        return answer, included_chunk_texts

    async def process_chat_message(self, case_id: str, question: str, current_chat_history: List[ChatMessage], attempt_decomposition: bool = False) -> ChatResponse:
        """
        Main method to process a user's chat message, incorporating advanced features.
        """
        logger.info(f"Processing chat for case '{case_id}'. Question: '{question[:50]}...' Decompose hint: {attempt_decomposition}")

        # Add user message to history immediately (important for context in sub-steps)
        self._add_to_history(case_id, "user", question)
        # Use the updated history for all subsequent LLM calls in this turn
        updated_history = self.get_chat_history(case_id)

        sub_questions_and_answers: Optional[Dict[str, str]] = None
        final_answer: str
        all_relevant_chunks = [] # Collect chunks used across all steps

        # --- Advanced Feature: Intent Analysis & Question Decomposition ---
        intent = llm_service.analyze_question_intent(question)
        logger.info(f"Question intent analyzed as: {intent}")

        should_decompose = attempt_decomposition or intent in ["COMPLEX_QUERY", "COMPARISON"]

        if should_decompose:
            logger.info("Attempting question decomposition...")
            sub_questions = llm_service.decompose_question(question)

            if len(sub_questions) > 1:
                logger.info(f"Decomposed into: {sub_questions}")
                sub_questions_and_answers = {}
                temp_history_for_subquestions = updated_history[:-1] # Use history *before* the complex question

                for i, sub_q in enumerate(sub_questions):
                    logger.info(f"Processing sub-question {i+1}/{len(sub_questions)}: '{sub_q[:50]}...'")
                    # Pass history *excluding* the original complex question and subsequent sub-answers
                    sub_answer, relevant_chunks_sub = self._generate_response_with_context(
                        sub_q, case_id, temp_history_for_subquestions, k=3 # Use fewer chunks per sub-q?
                    )
                    sub_questions_and_answers[sub_q] = sub_answer
                    all_relevant_chunks.extend(relevant_chunks_sub)
                    # Add sub-Q&A to a temporary history for subsequent sub-questions *within this turn*?
                    # This could help if sub-questions depend on each other, but adds complexity.
                    # For now, let's keep it simpler and use the history *before* the complex question for all sub-qs.
                    # temp_history_for_subquestions.append(ChatMessage(role="user", content=f"[Sub-question] {sub_q}"))
                    # temp_history_for_subquestions.append(ChatMessage(role="assistant", content=sub_answer))


                # --- Advanced Feature: Multi-level Response Synthesis ---
                logger.info("Synthesizing final answer from sub-answers...")
                final_answer = llm_service.synthesize_answers(question, sub_questions_and_answers) or "[LLM failed to synthesize answers]"

            else:
                # Decomposition resulted in one question (or failed), treat as simple query
                logger.info("Decomposition resulted in single question, processing directly.")
                final_answer, relevant_chunks_simple = self._generate_response_with_context(
                    question, case_id, updated_history, k=5 # Use updated_history including the user question
                )
                all_relevant_chunks.extend(relevant_chunks_simple)
        else:
            # Process as a simple query without decomposition
            logger.info("Processing as simple query (no decomposition).")
            final_answer, relevant_chunks_simple = self._generate_response_with_context(
                question, case_id, updated_history, k=5 # Use updated_history including the user question
            )
            all_relevant_chunks.extend(relevant_chunks_simple)


        # Remove duplicates from relevant chunks list for the final response
        unique_relevant_chunks = list(dict.fromkeys(all_relevant_chunks))

        # Add final assistant response to history
        self._add_to_history(case_id, "assistant", final_answer)

        return ChatResponse(
            answer=final_answer,
            relevant_chunks=unique_relevant_chunks, # Show context used
            chat_history=self.get_chat_history(case_id), # Return the full updated history
            sub_questions_and_answers=sub_questions_and_answers
        )

# Instantiate for easy import
chat_service = ChatService()
