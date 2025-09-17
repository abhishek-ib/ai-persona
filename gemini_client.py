"""
Google Gemini Integration for AI Persona Project

This module handles communication with Google Gemini API for generating
natural language responses based on conversation context.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from datetime import datetime


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", enable_logging: bool = True):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google AI API key (if None, will try to get from environment)
            model_name: Name of the Gemini model to use
            enable_logging: Whether to enable detailed logging
        """
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            raise ValueError("Google AI API key not provided. Set GOOGLE_AI_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name = model_name
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            self._setup_logging()
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config for consistent responses
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=512,
            stop_sequences=["User:", "Assistant:", "\n\n---"]
        )
    
    def _setup_logging(self):
        """Setup logging for context and responses"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Setup logger for context logging
        self.context_logger = logging.getLogger('gemini_context')
        self.context_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.context_logger.handlers.clear()
        
        # Create file handler for context logs
        context_handler = logging.FileHandler('logs/gemini_context.log')
        context_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        context_handler.setFormatter(formatter)
        
        self.context_logger.addHandler(context_handler)
        
        # Setup logger for responses
        self.response_logger = logging.getLogger('gemini_responses')
        self.response_logger.setLevel(logging.INFO)
        self.response_logger.handlers.clear()
        
        response_handler = logging.FileHandler('logs/gemini_responses.log')
        response_handler.setLevel(logging.INFO)
        response_handler.setFormatter(formatter)
        
        self.response_logger.addHandler(response_handler)
    
    def _log_context(self, context: Dict[str, Any], query: str):
        """Log the context sent to Gemini"""
        if not self.enable_logging:
            return
        
        # Handle both legacy and new threaded context formats
        if 'target_user' in context and isinstance(context['target_user'], dict):
            # New threaded format
            target_user_name = context['target_user']['name']
            target_user_id = context['target_user']['id']
            similar_messages_count = len(context.get('similar_conversations', []))
        else:
            # Legacy format
            target_user_name = context['target_user_name']
            target_user_id = context['target_user_id']
            similar_messages_count = len(context['similar_messages'])
            
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'target_user': target_user_name,
            'target_user_id': target_user_id,
            'similar_messages_count': similar_messages_count,
            'similar_messages': [],
            'user_recent_messages': []
        }
        
        # Handle different context formats for logging
        if 'target_user' in context and isinstance(context['target_user'], dict):
            # New threaded format
            similar_messages = context.get('similar_conversations', [])
            user_messages = context.get('user_conversation_history', [])
            log_entry['user_recent_messages_count'] = len(user_messages)
            
            # Log similar threaded conversations
            for i, conv in enumerate(similar_messages):
                conv_entry = {
                    'rank': i + 1,
                    'similarity_score': conv.get('similarity_score', 0),
                    'author': conv.get('author', 'Unknown'),
                    'participants': conv.get('participants', []),
                    'conversation_type': conv.get('conversation_type', 'Single message'),
                    'content': conv.get('content', ''),  # Full content for logging
                    'timestamp': conv.get('timestamp', '')
                }
                log_entry['similar_messages'].append(conv_entry)
            
            # Log user's recent conversations
            for i, conv in enumerate(user_messages[:5]):
                conv_entry = {
                    'rank': i + 1,
                    'author': conv.get('author', 'Unknown'),
                    'participants': conv.get('participants', []),
                    'conversation_type': conv.get('conversation_type', 'Single message'),
                    'content': conv.get('content', ''),  # Full content for logging
                    'timestamp': conv.get('timestamp', '')
                }
                log_entry['user_recent_messages'].append(conv_entry)
        else:
            # Legacy format
            similar_messages = context.get('similar_messages', [])
            user_messages = context.get('user_recent_messages', [])
            log_entry['user_recent_messages_count'] = len(user_messages)
            
            # Log similar messages (closest neighbors)
            for i, msg in enumerate(similar_messages):
                msg_entry = {
                    'rank': i + 1,
                    'similarity_score': msg.get('similarity_score', 0),
                    'user_name': msg.get('user_name', 'Unknown'),
                    'user_id': msg.get('user_id', 'Unknown'),
                    'content': msg.get('content', ''),  # Full content for logging
                    'timestamp': msg.get('timestamp', 0),
                    'datetime': msg.get('datetime', '')
                }
                log_entry['similar_messages'].append(msg_entry)
            
            # Log user's recent messages
            for i, msg in enumerate(user_messages[:5]):
                msg_entry = {
                    'rank': i + 1,
                    'user_name': msg.get('user_name', 'Unknown'),
                    'content': msg.get('content', ''),  # Full content for logging
                    'timestamp': msg.get('timestamp', 0),
                    'datetime': msg.get('datetime', '')
                }
                log_entry['user_recent_messages'].append(msg_entry)
        
        # Log to file
        self.context_logger.info(f"CONTEXT: {json.dumps(log_entry, indent=2)}")
        
        # Also print to console for immediate feedback
        print(f"\n📝 Context logged for query: '{query[:50]}...'")
        print(f"   Target user: {target_user_name}")
        print(f"   Similar messages: {similar_messages_count}")
        print(f"   User context messages: {log_entry['user_recent_messages_count']}")
        
        # Handle different formats for console output
        if 'target_user' in context and isinstance(context['target_user'], dict):
            # New threaded format
            similar_conversations = context.get('similar_conversations', [])
            if similar_conversations:
                print("   Top similar conversations:")
                for i, conv in enumerate(similar_conversations[:3], 1):
                    score = conv.get('similarity_score', 0)
                    author = conv.get('author', 'Unknown')
                    conv_type = conv.get('conversation_type', 'Single message')
                    content = conv.get('content', '')
                    print(f"     {i}. [{score:.3f}] {author} ({conv_type}): {content}")
        else:
            # Legacy format
            similar_messages = context.get('similar_messages', [])
            if similar_messages:
                print("   Top similar messages:")
                for i, msg in enumerate(similar_messages[:3], 1):
                    score = msg.get('similarity_score', 0)
                    user = msg.get('user_name', 'Unknown')
                    content = msg.get('content', '')
                    print(f"     {i}. [{score:.3f}] {user}: {content}")
        
    def create_persona_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create a prompt for persona-based response generation with threaded conversation support
        
        Args:
            context: Context dictionary from vector store
            
        Returns:
            Formatted prompt string
        """
        # Handle both old and new context formats
        if 'target_user' in context and isinstance(context['target_user'], dict):
            # New threaded conversation format
            return self._create_threaded_conversation_prompt(context)
        else:
            # Legacy format
            return self._create_legacy_prompt(context)
    
    def _create_threaded_conversation_prompt(self, context: Dict[str, Any]) -> str:
        """Create a prompt optimized for threaded conversation context"""
        query = context['query']
        target_user = context['target_user']
        similar_conversations = context.get('similar_conversations', [])
        user_conversations = context.get('user_conversation_history', [])
        
        user_name = target_user['name']
        session_context = context.get('session_context', '')
        
        prompt_parts = [
            f"You are an AI assistant that responds exactly as {user_name} would respond.",
            f"You have access to {user_name}'s threaded conversation history and similar conversations for context.",
        ]
        
        # Add session context if available
        if session_context:
            prompt_parts.extend([
                "",
                "## Recent Conversation Context:",
                session_context,
            ])
        
        prompt_parts.extend([
            "",
            "## Context from Similar Threaded Conversations:",
        ])
        
        # Add similar threaded conversations
        if similar_conversations:
            for i, conv in enumerate(similar_conversations[:3], 1):
                score = conv.get('similarity_score', 0)
                conv_type = conv.get('conversation_type', 'Single message')
                author = conv.get('author', 'Unknown')
                participants = conv.get('participants', [])
                timestamp = conv.get('timestamp', '')[:10] if conv.get('timestamp') else 'Unknown'
                
                prompt_parts.append(f"\n{i}. [{score:.3f}] {conv_type} - Started by {author} [{timestamp}]")
                if len(participants) > 1:
                    prompt_parts.append(f"   Participants: {', '.join(participants)}")
                
                # Format the conversation content
                content = conv.get('content', '')
                if len(content) > 600:
                    content = content[:600] + "..."
                
                # Add the conversation with indentation
                for line in content.split('\n'):
                    if line.strip():
                        prompt_parts.append(f"   {line}")
        else:
            prompt_parts.append("No similar conversations found.")
        
        prompt_parts.extend([
            "",
            f"## Recent Conversations from {user_name}:",
        ])
        
        # Add user's conversation history
        if user_conversations:
            for i, conv in enumerate(user_conversations[:3], 1):
                conv_type = conv.get('conversation_type', 'Single message')
                participants = conv.get('participants', [])
                timestamp = conv.get('timestamp', '')[:10] if conv.get('timestamp') else 'Unknown'
                
                prompt_parts.append(f"\n{i}. {conv_type} [{timestamp}]")
                if len(participants) > 1:
                    prompt_parts.append(f"   Participants: {', '.join(participants)}")
                
                content = conv.get('content', '')
                if len(content) > 500:
                    content = content[:500] + "..."
                
                for line in content.split('\n'):
                    if line.strip():
                        prompt_parts.append(f"   {line}")
        else:
            prompt_parts.append(f"No recent conversations from {user_name} found.")
        
        prompt_parts.extend([
            "",
            "## Instructions:",
            f"- Respond exactly as {user_name} would, using their communication style and patterns",
            "- Pay attention to how they participate in threaded conversations",
            "- Notice their tone, vocabulary, and typical response patterns",
            "- Consider whether this might start a discussion thread or be a simple response",
            "- Use the conversation context above to inform your response style",
            "- Keep the response natural and authentic to their personality",
            "- Don't directly reference the context - just use it to guide your response style",
            "",
            f"## Current Message to Respond To:",
            f'"{query}"',
            "",
            f"## Response as {user_name}:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _create_legacy_prompt(self, context: Dict[str, Any]) -> str:
        """Create the original prompt format for backward compatibility"""
        target_user_name = context['target_user_name']
        query = context['query']
        similar_messages = context['similar_messages']
        user_recent_messages = context['user_recent_messages']
        
        prompt_parts = [
            f"You are an AI assistant that responds as {target_user_name} would respond.",
            f"You have access to {target_user_name}'s conversation history and similar conversations for context.",
            "",
            "## Context from Similar Conversations:",
        ]
        
        # Add similar messages for context
        if similar_messages:
            for i, msg in enumerate(similar_messages[:5], 1):
                timestamp = datetime.fromisoformat(msg['datetime']).strftime('%Y-%m-%d %H:%M') if msg['datetime'] else 'Unknown time'
                prompt_parts.append(f"{i}. [{timestamp}] {msg['user_name']}: {msg['content']}")
        else:
            prompt_parts.append("No similar conversations found.")
        
        prompt_parts.extend([
            "",
            f"## Recent Messages from {target_user_name}:",
        ])
        
        # Add user's recent messages for personality context
        if user_recent_messages:
            for i, msg in enumerate(user_recent_messages[:5], 1):
                timestamp = datetime.fromisoformat(msg['datetime']).strftime('%Y-%m-%d %H:%M') if msg['datetime'] else 'Unknown time'
                prompt_parts.append(f"{i}. [{timestamp}] {msg['content']}")
        else:
            prompt_parts.append(f"No recent messages from {target_user_name} found.")
        
        prompt_parts.extend([
            "",
            "## Instructions:",
            f"- Respond as {target_user_name} would, using their typical communication style",
            "- Use the context above to inform your response but don't directly reference it",
            "- Keep the response natural and conversational",
            "- Match the tone and style of the user's previous messages",
            "- Be helpful and engaging while staying in character",
            "- If you don't have enough context, respond in a friendly, general way",
            "",
            f"## Current Message to Respond To:",
            f'"{query}"',
            "",
            f"## Response as {target_user_name}:"
        ])
        
        return "\n".join(prompt_parts)
    
    def generate_response(self, context: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate a response using Gemini
        
        Args:
            context: Context dictionary from vector store
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        # Log the context before generating response
        self._log_context(context, context['query'])
        
        prompt = self.create_persona_prompt(context)
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                
                if response.text:
                    # Handle both context formats
                    if 'target_user' in context and isinstance(context['target_user'], dict):
                        user_name = context['target_user']['name']
                        user_id = context['target_user']['id']
                    else:
                        user_name = context['target_user_name']
                        user_id = context['target_user_id']
                    
                    result = {
                        'success': True,
                        'response': response.text.strip(),
                        'user_name': user_name,
                        'user_id': user_id,
                        'query': context['query'],
                        'timestamp': datetime.now().isoformat(),
                        'model_used': self.model_name,
                        'attempt': attempt + 1
                    }
                    
                    # Log the response
                    if self.enable_logging:
                        self.response_logger.info(f"RESPONSE: {json.dumps(result, indent=2)}")
                        print(f"✅ Response generated and logged")
                    
                    return result
                else:
                    print(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"Error generating response (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Handle both context formats
                    if 'target_user' in context and isinstance(context['target_user'], dict):
                        user_name = context['target_user']['name']
                        user_id = context['target_user']['id']
                    else:
                        user_name = context['target_user_name']
                        user_id = context['target_user_id']
                    
                    return {
                        'success': False,
                        'error': str(e),
                        'user_name': user_name,
                        'user_id': user_id,
                        'query': context['query'],
                        'timestamp': datetime.now().isoformat(),
                        'model_used': self.model_name,
                        'attempts': max_retries
                    }
        
        # Handle both context formats for final fallback
        if 'target_user' in context and isinstance(context['target_user'], dict):
            user_name = context['target_user']['name']
            user_id = context['target_user']['id']
        else:
            user_name = context['target_user_name']
            user_id = context['target_user_id']
        
        return {
            'success': False,
            'error': 'Max retries exceeded',
            'user_name': user_name,
            'user_id': user_id,
            'query': context['query'],
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model_name,
            'attempts': max_retries
        }
    
    def generate_simple_response(self, query: str, user_name: str = "Assistant") -> str:
        """
        Generate a simple response without persona context
        
        Args:
            query: Input query
            user_name: Name to use in response
            
        Returns:
            Generated response text
        """
        prompt = f"""You are {user_name}. Respond to the following message in a natural, conversational way:

"{query}"

Response:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text.strip() if response.text else "I'm not sure how to respond to that."
        except Exception as e:
            print(f"Error generating simple response: {e}")
            return "Sorry, I'm having trouble generating a response right now."
    
    def test_connection(self) -> bool:
        """Test if the Gemini API connection is working"""
        try:
            response = self.model.generate_content("Hello, this is a test message.")
            return bool(response.text)
        except Exception as e:
            print(f"Gemini API test failed: {e}")
            return False


class ResponseCache:
    """Simple cache for storing generated responses"""
    
    def __init__(self, cache_file: str = "response_cache.json"):
        self.cache_file = cache_file
        self.cache = self.load_cache()
    
    def load_cache(self) -> Dict[str, Any]:
        """Load cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
        return {}
    
    def save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_cache_key(self, query: str, user_id: str) -> str:
        """Generate cache key for query and user"""
        return f"{user_id}:{hash(query)}"
    
    def get_cached_response(self, query: str, user_id: str) -> Optional[str]:
        """Get cached response if available"""
        key = self.get_cache_key(query, user_id)
        return self.cache.get(key)
    
    def cache_response(self, query: str, user_id: str, response: str):
        """Cache a response"""
        key = self.get_cache_key(query, user_id)
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        self.save_cache()


if __name__ == "__main__":
    # Test the Gemini client
    try:
        client = GeminiClient()
        if client.test_connection():
            print("✓ Gemini API connection successful!")
            
            # Test simple response
            test_response = client.generate_simple_response("Hello, how are you?")
            print(f"Test response: {test_response}")
        else:
            print("✗ Gemini API connection failed!")
    except Exception as e:
        print(f"Error testing Gemini client: {e}")
        print("Make sure to set your GOOGLE_AI_API_KEY environment variable.")
