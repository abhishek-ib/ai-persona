"""
JSON-Based Gemini Client for AI Persona Project

This client works with complete conversation JSON files instead of fragmented context,
providing much better conversation understanding for Gemini.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai
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
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            raise ValueError("Google AI API key not provided. Set GOOGLE_AI_API_KEY environment variable.")

        self.model_name = model_name
        self.enable_logging = enable_logging

        # Setup logging
        if enable_logging:
            self._setup_logging()

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

        # # Generation config
        self.generation_config = genai.types.GenerationConfig(
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

    def get_or_create_chat_session(self, user_name: str) -> Any:
        """Get or create a chat session with helpful coworker setup"""

        if user_name not in self.chat_sessions:
            # Create new chat session with helpful coworker context
            chat = self.model.start_chat()

            # Send initial helpful coworker setup message
            initial_prompt = self._create_initial_prompt()

            try:
                # Initialize the chat with helpful coworker context
                initial_response = chat.send_message(initial_prompt)

                self.chat_sessions[user_name] = chat

            except Exception as e:
                print(f"❌ Failed to initialize chat session: {e}")
                return None

        return self.chat_sessions[user_name]

    def generate_response(self, target_user: Dict[str, Any], query: str,
                         similar_conversations: List[Dict[str, Any]],
                         user_conversations: List[Dict[str, Any]],
                         is_first_message: bool = False) -> Dict[str, Any]:
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
                chat = self.get_or_create_chat_session(user_name)
                if not chat:
                    raise Exception("Failed to create chat session")
            else:
                chat = self.chat_sessions[user_name]

            # Create message with attached relevant conversations (1-2 most relevant)
            message_with_context = self._create_message_with_attachments(
                query, similar_conversations
            )
            print(f"   [GeminiClient] Query: {message_with_context}")

            # Send the query with attached conversations to the chat session
            response = chat.send_message(
                message_with_context,
                generation_config=self.generation_config
            )

            # Check if response has valid content
            if hasattr(response, 'text') and response.text:
                result = {
                    'success': True,
                    'response': response.text.strip(),
                    'user_name': 'Helpful Coworker',
                    'user_id': target_user['id'],
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'model_used': self.model_name,
                    'context_conversations': len(similar_conversations),
                    'user_conversations': len(user_conversations),
                    'has_session_context': user_name in self.chat_sessions
                }

                # Log successful response
                if self.enable_logging:
                    self.response_logger.info(f"JSON_RESPONSE: {json.dumps(result, indent=2)}")

                return result
            else:
                # Handle cases where response is blocked or empty
                finish_reason = getattr(response, 'finish_reason', 'unknown')
                raise Exception(f"No valid response text (finish_reason: {finish_reason})")

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

    def _create_initial_prompt(self) -> str:
        """Create initial helpful coworker prompt for chat session"""

        prompt = """Reply like a helpful coworker. For all questions, you will be given attached files of relevant conversations from Slack. Look through these conversations to see if you can find anything helpful and respond accordingly.

If you can find relevant information in the conversations, provide a helpful answer based on what you found.
If you can't find information that answers the question with high degree of certainty, politely say you can't find information on that topic. Each message should also have a timestamp, prefer the latest information over the oldest information, if you feel that the timestamp is old, mention that the information might be outdated. don't mention the timestamp itself, just that the information might be outdated.

Keep responses conversational, short and concise, helpful, and reference the conversations when relevant.

Respond with 'Ready' to confirm you understand."""

        return prompt

    def _create_message_with_attachments(self, query: str,
                                       relevant_conversations: List[Dict[str, Any]]) -> str:
        """Create message with attached relevant conversations as raw JSON"""

        message_parts = [
            f"{query}",
            "",
            ""
        ]

        if relevant_conversations:
            # Dump the entire JSON structure directly
            import json
            json_dump = json.dumps(relevant_conversations, indent=4)
            message_parts.append(json_dump)
        else:
            message_parts.append("No relevant conversations found.")

        return "\n".join(message_parts)

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
        json_files_sent = [conv.get('id', 'unknown') + '.json' for conv in similar_conversations[:2]]

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
            print("   Top similar conversations (COMPLETE DATA):")
            for i, conv in enumerate(similar_conversations[:3], 1):
                score = conv.get('similarity_score', 0)
                conv_type = conv.get('conversation_type', 'single')
                channel_name = conv.get('channel_name', 'DM')
                message_count = len(conv.get('messages', []))
                participants = ', '.join(conv.get('participants', [])[:3])
                json_file = conv.get('id', 'unknown') + '.json'
                print(f"     {i}. [{score:.3f}] {channel_name} - {conv_type} ({message_count} msgs) - {participants}")
                print(f"        📄 JSON file: {json_file}")

    def test_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            response = self.model.generate_content("Hello, this is a test message.")
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
