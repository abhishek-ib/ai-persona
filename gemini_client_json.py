"""
JSON-Based Gemini Client for AI Persona Project

This client works with complete conversation JSON files instead of fragmented context,
providing much better conversation understanding for Gemini.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class JSONGeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", enable_logging: bool = True):
        """
        Initialize JSON-based Gemini client with chat session support
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model name
            enable_logging: Enable detailed logging
        """
        self.api_key = api_key or os.getenv(
            'GOOGLE_AI_API_KEY') or ''
        if not self.api_key:
            raise ValueError("Google AI API key not provided. Set GOOGLE_AI_API_KEY environment variable.")

        self.model_name = model_name
        self.enable_logging = enable_logging

        # Setup logging
        if enable_logging:
            self._setup_logging()

        # Configure Gemini client with new SDK
        self.client = genai.Client(api_key=self.api_key)

        # Generation config
        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=1024
        )

        # Chat sessions per user (user_name -> chat_session)
        self.chat_sessions: Dict[str, Any] = {}

    def _setup_logging(self):
        """Setup logging for context and responses"""
        os.makedirs("logs", exist_ok=True)

        # Context logger
        self.context_logger = logging.getLogger('json_gemini_context')
        self.context_logger.setLevel(logging.INFO)

        context_handler = logging.FileHandler('logs/json_gemini_context.log')
        context_formatter = logging.Formatter('%(asctime)s - %(message)s')
        context_handler.setFormatter(context_formatter)

        self.context_logger.handlers.clear()
        self.context_logger.addHandler(context_handler)

        # Response logger
        self.response_logger = logging.getLogger('json_gemini_responses')
        self.response_logger.setLevel(logging.INFO)

        response_handler = logging.FileHandler('logs/json_gemini_responses.log')
        response_handler.setFormatter(context_formatter)

        self.response_logger.handlers.clear()
        self.response_logger.addHandler(response_handler)

    def get_or_create_chat_session(self, user_name: str, mode: str = "interactive") -> Any:
        """Get or create a chat session with helpful coworker setup"""

        if user_name not in self.chat_sessions:
            # Create new chat session with context based on mode
            initial_prompt = self._create_initial_prompt(mode)

            try:
                # Initialize the chat with context using new SDK
                initial_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=initial_prompt,
                    config=self.generation_config
                )

                # Store the conversation history for this session
                self.chat_sessions[user_name] = [
                    types.Content(role="user", parts=[types.Part.from_text(text=initial_prompt)]),
                    types.Content(role="model", parts=[types.Part.from_text(text=initial_response.text)])
                ]

            except Exception as e:
                print(f"❌ Failed to initialize chat session: {e}")
                return None

        return self.chat_sessions[user_name]

    def generate_response(self, target_user: Dict[str, Any], query: str,
                         similar_conversations: List[Dict[str, Any]],
                         user_conversations: List[Dict[str, Any]],
                         is_first_message: bool = False,
                         mode: str = "interactive") -> Dict[str, Any]:
        """
        Generate response using Gemini chat sessions for context continuity
        
        Args:
            target_user: Target user information
            query: User query
            similar_conversations: List of complete conversation JSON data
            user_conversations: User's recent conversation JSON data
            is_first_message: Whether this is the first message in a session
            
        Returns:
            Response dictionary
        """
        user_name = target_user['name']

        # Log the complete context
        if self.enable_logging:
            self._log_complete_context(target_user, query, similar_conversations,
                                     user_conversations, is_first_message)

        try:
            # Get or create chat session for this user
            if is_first_message or user_name not in self.chat_sessions:
                conversation_history = self.get_or_create_chat_session(user_name, mode)
                if not conversation_history:
                    raise Exception("Failed to create chat session")
            else:
                conversation_history = self.chat_sessions[user_name]

            # Retry mechanism: start with all conversations, trim if 400 error (too many tokens)
            max_retries = 8
            conversations_to_send = similar_conversations
            response = None

            for attempt in range(max_retries):
                try:
                    # Create message with attached relevant conversations
                    message_content = self._create_message_with_attachments(
                        query, conversations_to_send
                    )
                    print(f"   [GeminiClient] Attempt {attempt + 1}: Query parts: {len(message_content.parts)} parts")

                    # Add the new user message to conversation history
                    conversation_history.append(message_content)

                    # Send the conversation history to generate response using new SDK
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=conversation_history,
                        config=self.generation_config
                    )

                    # Check if response has valid content
                    if hasattr(response, 'text') and response.text:
                        # If we get here, the request was successful
                        break
                    else:
                        # Response exists but no text - check finish reason
                        finish_reason = getattr(response, 'finish_reason', 'unknown')
                        print(f"   [GeminiClient] Attempt {attempt + 1} failed: No valid response text (finish_reason: {finish_reason})")

                        # Debug: Check candidates for finish reason
                        if hasattr(response, 'candidates') and response.candidates:
                            for i, candidate in enumerate(response.candidates):
                                candidate_finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                                print(f"   [GeminiClient] Debug: Candidate {i} finish_reason: {candidate_finish_reason}")
                                if candidate_finish_reason != 'unknown':
                                    finish_reason = candidate_finish_reason

                        # Check if it's a retryable issue (token limit, safety, or unknown finish reason)
                        if finish_reason in ['MAX_TOKENS', 'SAFETY'] or (finish_reason == 'unknown' and attempt < max_retries - 1):
                            # Trim conversations by 1 and retry
                            if len(conversations_to_send) > 1:
                                conversations_to_send = conversations_to_send[:-1]
                                print(f"   [GeminiClient] Trimming conversations to {len(conversations_to_send)} due to {finish_reason}")
                                # Remove the failed message from conversation history
                                conversation_history.pop()
                            else:
                                print(f"   [GeminiClient] No more conversations to trim, giving up")
                                raise Exception(f"No valid response text (finish_reason: {finish_reason})")
                        else:
                            # Not a retryable error or max retries reached
                            raise Exception(f"No valid response text (finish_reason: {finish_reason})")

                except Exception as e:
                    error_str = str(e)
                    print(f"   [GeminiClient] Attempt {attempt + 1} failed: {error_str}")

                    # Check if it's a 400 error (likely too many tokens)
                    if "400" in error_str and "INVALID_ARGUMENT" in error_str and attempt < max_retries - 1:
                        # Trim conversations by 1 and retry
                        if len(conversations_to_send) > 1:
                            conversations_to_send = conversations_to_send[:-1]
                            print(f"   [GeminiClient] Trimming conversations to {len(conversations_to_send)} due to 400 error")
                            # Remove the failed message from conversation history
                            conversation_history.pop()
                        else:
                            print(f"   [GeminiClient] No more conversations to trim, giving up")
                            raise e
                    else:
                        # Not a 400 error or max retries reached, re-raise
                        raise e

            if response is None:
                raise Exception("All retry attempts failed")

            # Response should be valid at this point since we broke out of the retry loop
            result = {
                'success': True,
                'response': response.text.strip(),
                'user_name': 'Helpful Coworker',
                'user_id': target_user['id'],
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'context_conversations': len(conversations_to_send),  # Show actual conversations used
                'user_conversations': len(user_conversations),
                'has_session_context': user_name in self.chat_sessions,
                'retry_attempts': attempt + 1  # Show how many attempts were needed
            }

            # Log successful response
            if self.enable_logging:
                self.response_logger.info(f"JSON_RESPONSE: {json.dumps(result, indent=2)}")

            return result

        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'user_name': target_user['name'],
                'user_id': target_user['id'],
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name
            }

            if self.enable_logging:
                self.response_logger.error(f"JSON_ERROR: {json.dumps(error_result, indent=2)}")

            return error_result

    def clear_chat_session(self, user_name: str):
        """Clear chat session for a user"""
        if user_name in self.chat_sessions:
            del self.chat_sessions[user_name]
            print(f"🗑️  Cleared chat session for {user_name}")

    def _create_initial_prompt(self, mode: str = "interactive") -> str:
        """Create initial prompt for chat session based on mode"""

        if mode == "search":
            prompt = """You are a helpful coworker answering questions based on Slack conversations. For all questions, you will be given attached files of relevant conversations from Slack.

Your task is to analyze the conversations and provide structured responses in this EXACT JSON format:

{
  "response": "Your helpful answer here based on the conversations",
  "references": ["<file_name_1>", "<file_name_2>"]
}

IMPORTANT - How to find conversation IDs:
- Each conversation JSON file has a file name --- JSON File: <file_name>
- If you use information from a conversation to build your answer, you MUST include that conversation's file name in the references array

Guidelines:
- In the references array, include ONLY the conversation file names from conversations that you actually used to form your answer
- If you can't find relevant information, say so in the response and use an empty references array
- Look for the file name at the top level of each JSON conversation file
- Prefer the latest information over older information based on timestamps
- Keep responses conversational, helpful, and reference the conversations when relevant
- If information seems outdated based on timestamps, mention that it might be outdated

Example:
If you use information from a conversation with file name "ch_C05L87V014J_2025-03-07_1741374080.130969_thread", then include "ch_C05L87V014J_2025-03-07_1741374080.130969_thread" in your references array.

Always respond in the EXACT JSON format specified above.

Respond with 'Ready' to confirm you understand."""
        else:
            # Default interactive mode prompt
            prompt = """Reply like a helpful coworker. For all questions, you will be given attached files of relevant conversations from Slack. Look through these conversations to see if you can find anything helpful and respond accordingly.

If you can find relevant information in the conversations, provide a helpful answer based on what you found.
If you can't find information that answers the question with high degree of certainty, politely say you can't find information on that topic. Each message should also have a timestamp, prefer the latest information over the oldest information, if you feel that the timestamp is old, mention that the information might be outdated. don't mention the timestamp itself, just that the information might be outdated.

Keep responses conversational, short and concise, helpful, and reference the conversations when relevant.

Respond with 'Ready' to confirm you understand."""

        return prompt

    def _create_message_with_attachments(self, query: str,
                                       conversation_files: List[Dict[str, Any]]) -> types.Content:
        """Create message with attached relevant conversations using new SDK types"""


        # Create parts list starting with the query
        parts = [types.Part.from_text(text=query)]

        if conversation_files:
            # Add each JSON file as a separate part
            for i, file_info in enumerate(conversation_files, 1):
                raw_json = file_info.get('raw_json_content', '')
                file_name = file_info.get('file_name', 'unknown')
                similarity_score = file_info.get('similarity_score', 0)

                # Add the raw JSON content as text with a clear label
                json_label = f"\n--- JSON File: {file_name} (similarity: {similarity_score:.3f}) ---\n"
                file_content = types.Part.from_text(
                    text=json_label + raw_json
                    )

                parts.append(file_content)
        else:
            parts.append(types.Part.from_text(text="\nNo relevant conversations found."))

        # Return the Content object with all parts
        return types.Content(
            role="user",
            parts=parts
        )

    def _create_json_prompt(self, target_user: Dict[str, Any], query: str,
                           similar_conversations: List[Dict[str, Any]],
                           user_conversations: List[Dict[str, Any]],
                           session_context: str = "") -> str:
        """Create prompt using complete conversation JSON data"""
        user_name = target_user['name']

        prompt_parts = [
            f"You are an AI assistant that responds exactly as {user_name} would respond.",
            f"You have access to {user_name}'s complete conversation history.",
            f"Respond in {user_name}'s typical style and tone based on the examples below.",
            ""
        ]

        # Add session context if available
        if session_context:
            prompt_parts.extend([
                "## Recent Chat Session Context:",
                session_context,
                ""
            ])

        # Add similar conversations with complete context
        if similar_conversations:
            prompt_parts.extend([
                "## Similar Conversations for Context:",
                ""
            ])

            for i, conv in enumerate(similar_conversations[:3], 1):
                score = conv.get('similarity_score', 0)
                conv_type = conv.get('conversation_type', 'single')
                channel_name = conv.get('channel_name', 'DM')
                participants = ', '.join(conv.get('participants', []))

                prompt_parts.append(f"{i}. [{score:.3f}] {channel_name} - {conv_type} ({participants})")

                # Add complete conversation
                messages = conv.get('messages', [])
                for msg in messages:
                    sender = msg.get('sender', 'Unknown')
                    message = msg.get('message', '')
                    time_str = msg.get('time', '')
                    is_reply = msg.get('is_reply', False)

                    if is_reply:
                        prompt_parts.append(f"   └─ [{time_str}] {sender}: {message}")
                    else:
                        prompt_parts.append(f"   [{time_str}] {sender}: {message}")

                prompt_parts.append("")  # Empty line between conversations

        # Add user's recent conversations
        if user_conversations:
            prompt_parts.extend([
                f"## {user_name}'s Recent Conversation Style:",
                ""
            ])

            for i, conv in enumerate(user_conversations[:2], 1):
                conv_type = conv.get('conversation_type', 'single')
                channel_name = conv.get('channel_name', 'DM')
                participants = ', '.join(conv.get('participants', []))

                prompt_parts.append(f"{i}. {channel_name} - {conv_type} ({participants})")

                # Add complete conversation
                messages = conv.get('messages', [])
                for msg in messages:
                    sender = msg.get('sender', 'Unknown')
                    message = msg.get('message', '')
                    time_str = msg.get('time', '')
                    is_reply = msg.get('is_reply', False)

                    if is_reply:
                        prompt_parts.append(f"   └─ [{time_str}] {sender}: {message}")
                    else:
                        prompt_parts.append(f"   [{time_str}] {sender}: {message}")

                prompt_parts.append("")  # Empty line between conversations

        # Add the actual query
        prompt_parts.extend([
            "## Current Query to Respond To:",
            f"User: {query}",
            "",
            f"Respond as {user_name} would, using their typical communication style from the examples above:",
            f"{user_name}:"
        ])

        return "\n".join(prompt_parts)

    def _log_complete_context(self, target_user: Dict[str, Any], query: str,
                            similar_conversations: List[Dict[str, Any]],
                            user_conversations: List[Dict[str, Any]],
                            is_first_message: bool = False):
        """Log the complete context being sent to Gemini"""
        if not self.enable_logging:
            return

        # Extract JSON file names for logging
        json_files_sent = [file_info.get('file_name', 'unknown') for file_info in similar_conversations[:2]]

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'target_user': target_user,
            'is_first_message': is_first_message,
            'similar_conversations_count': len(similar_conversations),
            'user_conversations_count': len(user_conversations),
            'json_files_sent_to_gemini': json_files_sent,  # Track which files were sent
            'similar_conversations': similar_conversations,  # Complete data
            'user_conversations': user_conversations  # Complete data
        }

        # Log to file
        self.context_logger.info(f"JSON_CONTEXT: {json.dumps(log_entry, indent=2)}")

        # Console output
        print(f"\n📝 JSON Context for query: '{query}...'")
        print(f"   Target user: {target_user['name']}")
        print(f"   Similar conversations: {len(similar_conversations)}")
        print(f"   User conversations: {len(user_conversations)}")
        print(f"   First message: {is_first_message}")

        if similar_conversations:
            print("   Top similar conversations (RAW JSON FILES):")
            for i, file_info in enumerate(similar_conversations[:3], 1):
                score = file_info.get('similarity_score', 0)
                file_name = file_info.get('file_name', 'unknown')
                json_path = file_info.get('json_path', 'unknown')
                print(f"     {i}. [{score:.3f}] File: {file_name}")
                print(f"        📄 Path: {json_path}")

    def test_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Hello, this is a test message."
            )
            return bool(response.text)
        except Exception as e:
            print(f"Gemini API test failed: {e}")
            return False


def main():
    """Test the JSON Gemini client"""
    print("🧪 Testing JSON Gemini Client")

    try:
        client = JSONGeminiClient()

        # Test connection
        if client.test_connection():
            print("✅ Gemini API connection successful")
        else:
            print("❌ Gemini API connection failed")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
