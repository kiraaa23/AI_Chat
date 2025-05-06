import logging
import requests
import json
from typing import List, Dict, Optional, Tuple

from config import settings
from models import ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.LLM_MODEL_NAME):
        self.base_url = base_url
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"
        self.chat_api_url = f"{self.base_url}/api/chat" # Using chat endpoint for history management

    def check_connection(self) -> Tuple[bool, str]:
        """Checks if the Ollama server is reachable."""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            # Additionally check if the specified model is available
            models_res = requests.get(f"{self.base_url}/api/tags")
            models_res.raise_for_status()
            available_models = [m['name'] for m in models_res.json().get('models', [])]
            if self.model not in available_models:
                 logger.warning(f"Ollama server is running, but model '{self.model}' not found. Available: {available_models}")
                 return True, f"Ollama running, but model '{self.model}' not found"
            logger.info(f"Successfully connected to Ollama at {self.base_url} with model '{self.model}' available.")
            return True, "OK"
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            return False, str(e)
        except Exception as e:
             logger.error(f"An unexpected error occurred checking Ollama connection: {e}")
             return False, f"Unexpected error: {str(e)}"

    def _prepare_chat_payload(self, prompt: str, chat_history: Optional[List[ChatMessage]] = None) -> Dict:
        messages = []
        if chat_history:
            for msg in chat_history:
                messages.append({"role": msg.role, "content": msg.content})
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False, # Keep it simple for now, set to True for streaming
             "options": { # Optional: Adjust generation parameters if needed
                # "temperature": 0.7,
                # "top_k": 40,
                # "top_p": 0.9,
            }
        }
        return payload

    def generate_response(self, prompt: str, chat_history: Optional[List[ChatMessage]] = None) -> Optional[str]:
        """
        Generates a response from the LLM using the /api/chat endpoint.
        Includes chat history context.
        """
        is_connected, status = self.check_connection()
        if not is_connected:
            logger.error(f"Cannot generate response: Ollama connection failed ({status}).")
            return f"[LLM Connection Error: {status}]"
        if "model not found" in status:
             return f"[LLM Error: {status}]"


        payload = self._prepare_chat_payload(prompt, chat_history)
        logger.debug(f"Ollama request payload: {json.dumps(payload, indent=2)}")

        try:
            response = requests.post(self.chat_api_url, json=payload, timeout=120) # Increased timeout
            response.raise_for_status()
            result = response.json()

            if "message" in result and "content" in result["message"]:
                ai_response = result["message"]["content"].strip()
                logger.info(f"LLM generated response (length: {len(ai_response)}).")
                return ai_response
            elif "error" in result:
                 logger.error(f"Ollama API returned an error: {result['error']}")
                 return f"[LLM API Error: {result['error']}]"
            else:
                 logger.error(f"Unexpected Ollama response format: {result}")
                 return "[LLM Error: Unexpected response format]"

        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after 120 seconds.")
            return "[LLM Error: Request timed out]"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama chat API: {e}", exc_info=True)
            return f"[LLM Request Error: {e}]"
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding Ollama JSON response: {e}", exc_info=True)
             return f"[LLM Response Error: Invalid JSON]"
        except Exception as e:
             logger.error(f"Unexpected error during LLM generation: {e}", exc_info=True)
             return f"[LLM Error: Unexpected error - {e}]"

    # --- Methods for Advanced Features ---

    def analyze_question_intent(self, question: str) -> str:
        """
        Uses the LLM to determine the intent (simple, complex, comparison, summary etc.).
        This is a basic example; more sophisticated intent classification could be used.
        """
        prompt = f"""Analyze the following user question and classify its intent. Choose one category:
- SIMPLE_QUERY: Asking for a specific fact or piece of information.
- COMPLEX_QUERY: Requires combining information from multiple sources or steps.
- COMPARISON: Asking to compare or contrast two or more items.
- SUMMARY: Asking for a summary of a topic or document.
- AMBIGUOUS: The question is unclear or lacks context.
- OTHER: Does not fit well into the above categories.

User Question: "{question}"

Intent Classification:"""

        # Use the generate endpoint directly for this classification task (no history needed)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1} # Low temp for classification
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            intent = result.get("response", "OTHER").strip().upper()
            # Basic validation/cleanup
            valid_intents = ["SIMPLE_QUERY", "COMPLEX_QUERY", "COMPARISON", "SUMMARY", "AMBIGUOUS", "OTHER"]
            if intent not in valid_intents:
                logger.warning(f"LLM returned unexpected intent '{intent}', defaulting to OTHER.")
                return "OTHER"
            logger.info(f"Analyzed intent for '{question[:50]}...' as: {intent}")
            return intent
        except Exception as e:
            logger.error(f"Error analyzing question intent: {e}", exc_info=True)
            return "OTHER" # Default on error

    def decompose_question(self, complex_question: str) -> List[str]:
        """
        Uses the LLM to break down a complex question into simpler sub-questions.
        """
        prompt = f"""Break down the following complex question into smaller, self-contained sub-questions that can likely be answered individually using provided document context. If the question is already simple, return just the original question in the list.

Complex Question: "{complex_question}"

Sub-questions (one per line):"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
             "options": {"temperature": 0.3}
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            decomposition = result.get("response", "").strip()
            sub_questions = [q.strip() for q in decomposition.split('\n') if q.strip()]

            if not sub_questions: # If LLM failed or returned empty
                 logger.warning(f"LLM failed to decompose question, returning original: '{complex_question}'")
                 return [complex_question]

            # Basic check: if only one sub-question is returned and it's very similar to the original, treat as non-decomposed
            if len(sub_questions) == 1 and len(sub_questions[0]) > 0.8 * len(complex_question):
                logger.info(f"Decomposition resulted in single similar question for '{complex_question[:50]}...'")
                # Might still be useful if LLM rephrased slightly
                # return [complex_question] # Option: force original
                return sub_questions # Option: use LLM's (potentially rephrased) version

            logger.info(f"Decomposed '{complex_question[:50]}...' into {len(sub_questions)} sub-questions.")
            return sub_questions
        except Exception as e:
            logger.error(f"Error decomposing question: {e}", exc_info=True)
            return [complex_question] # Fallback to original question


    def synthesize_answers(self, original_question: str, sub_answers: Dict[str, str]) -> Optional[str]:
        """
        Uses the LLM to synthesize a final answer from the answers to sub-questions.
        """
        if not sub_answers:
            return "[Synthesis Error: No sub-answers provided]"

        sub_qa_pairs = "\n".join([f"Sub-question: {q}\nAnswer: {a}" for q, a in sub_answers.items()])

        prompt = f"""Given the original user question and the following answers generated for its sub-questions based on document context, synthesize a single, coherent, and comprehensive final answer. Address the original question directly.

Original Question: "{original_question}"

Sub-questions and their Answers:
{sub_qa_pairs}

Synthesized Final Answer:"""

        # Use the chat endpoint so it potentially has context from previous turns if needed, though likely not necessary here
        # Providing an empty history ensures it focuses only on the provided info.
        return self.generate_response(prompt, chat_history=[])


# Instantiate for easy import
llm_service = LLMService()