#!/usr/bin/env python3
"""
AI Persona Bot - Enhanced Threaded Version

This version uses the threaded data normalizer to provide better context
by understanding conversation threads and replies.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data_normalizer_threaded import ThreadedDataNormalizer
from vector_store_pinecone import PineconeVectorStore
from gemini_client import GeminiClient, ResponseCache
from chat_session import session_manager


class AIPersonaBotThreaded:
    def __init__(self, data_dir: str = "data", rebuild_index: bool = False, pinecone_api_key: Optional[str] = None):
        """
        Initialize the AI Persona Bot with threaded conversation support
        
        Args:
            data_dir: Directory containing Slack message data
            rebuild_index: Whether to rebuild the vector index from scratch
            pinecone_api_key: Optional Pinecone API key
        """
        self.data_dir = data_dir
        self.normalizer = ThreadedDataNormalizer(data_dir)
        self.vector_store = None
        self.gemini_client = None
        self.response_cache = ResponseCache()
        
        # Available users (will be populated after data loading)
        self.available_users = {}
        
        # Initialize the bot
        self._initialize(rebuild_index, pinecone_api_key)
    
    def _initialize(self, rebuild_index: bool = False, pinecone_api_key: Optional[str] = None):
        """Initialize the bot by loading/building necessary components"""
        print("🤖 Initializing AI Persona Bot with Threaded Conversations...")
        print("🚀 Using Pinecone for scalable vector storage with thread support")
        
        # Initialize Pinecone vector store
        try:
            self.vector_store = PineconeVectorStore(
                api_key=pinecone_api_key,
                index_name="ai-persona-threaded",
                enable_logging=True
            )
            print("✅ Connected to Pinecone")
        except Exception as e:
            print(f"❌ Failed to connect to Pinecone: {e}")
            print("Make sure PINECONE_API_KEY is set or provide the API key")
            raise e
        
        # Check if we need to rebuild the index
        normalized_data_exists = os.path.exists("normalized_messages_threaded.json")
        
        # Check if Pinecone index is empty
        stats = self.vector_store.get_index_stats()
        index_vector_count = stats.get('total_vector_count', 0) if stats else 0
        
        if rebuild_index or not normalized_data_exists or index_vector_count == 0:
            if index_vector_count == 0 and normalized_data_exists:
                print("📊 Pinecone index is empty, building from existing threaded data...")
                print("📚 Loading existing normalized threaded data...")
                with open("normalized_messages_threaded.json", 'r') as f:
                    messages = json.load(f)
            else:
                print("📊 Normalizing message data with thread support...")
                messages = self.normalizer.normalize_all_data()
                self.normalizer.save_normalized_data(messages)
            
            print("🔍 Building Pinecone vector index with threaded conversations...")
            self.vector_store.build_index(messages)
        else:
            print("📚 Using existing Pinecone threaded index...")
            print(f"Index contains {index_vector_count} conversation blocks")
        
        # Load available users
        self._load_available_users()
        
        print("✅ AI Persona Bot with Threaded Conversations initialized successfully!")
    
    def _load_available_users(self):
        """Load available users from the normalized threaded data"""
        try:
            with open("normalized_messages_threaded.json", 'r') as f:
                messages = json.load(f)
            
            # Count messages per user
            user_counts = {}
            for message in messages:
                user_id = message['user_id']
                user_name = message['user_name']
                if user_id not in user_counts:
                    user_counts[user_id] = {
                        'name': user_name,
                        'count': 0,
                        'thread_count': 0
                    }
                user_counts[user_id]['count'] += 1
                if message['message_type'] == 'thread':
                    user_counts[user_id]['thread_count'] += 1
            
            # Only include users with significant message history
            self.available_users = {
                uid: info for uid, info in user_counts.items()
                if info['count'] >= 5  # At least 5 conversation blocks
            }
            
            print(f"📋 Found {len(self.available_users)} users with sufficient conversation history:")
            for uid, info in sorted(self.available_users.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
                print(f"  - {info['name']}: {info['count']} blocks ({info['thread_count']} threaded)")
                
        except Exception as e:
            print(f"Error loading users: {e}")
            self.available_users = {}
    
    def initialize_gemini(self, api_key: Optional[str] = None):
        """Initialize Gemini client with API key"""
        try:
            self.gemini_client = GeminiClient(api_key, enable_logging=True)
            if self.gemini_client.test_connection():
                print("✅ Gemini API connected successfully!")
                return True
            else:
                print("❌ Gemini API connection failed!")
                return False
        except Exception as e:
            print(f"❌ Error initializing Gemini: {e}")
            return False
    
    def list_available_users(self) -> List[Dict[str, Any]]:
        """Get list of available users for persona selection"""
        return [
            {
                'user_id': uid,
                'name': info['name'],
                'message_count': info['count'],
                'thread_count': info['thread_count']
            }
            for uid, info in sorted(self.available_users.items(), key=lambda x: x[1]['count'], reverse=True)
        ]
    
    def chat_as_user(self, user_identifier: str, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate a response as a specific user using threaded conversation context
        
        Args:
            user_identifier: User ID or name to respond as
            query: Input message/query
            use_cache: Whether to use cached responses
            
        Returns:
            Response dictionary
        """
        if not self.gemini_client:
            return {
                'success': False,
                'error': 'Gemini API not initialized. Call initialize_gemini() first.',
                'query': query
            }
        
        # Find user ID
        target_user_id = self._find_user_id(user_identifier)
        if not target_user_id:
            return {
                'success': False,
                'error': f'User "{user_identifier}" not found. Use list_available_users() to see available users.',
                'query': query,
                'available_users': list(self.available_users.keys())
            }
        
        # Check cache first
        if use_cache:
            cached_response = self.response_cache.get_cached_response(query, target_user_id)
            if cached_response:
                return {
                    'success': True,
                    'response': cached_response['response'],
                    'user_name': self.available_users[target_user_id]['name'],
                    'user_id': target_user_id,
                    'query': query,
                    'from_cache': True,
                    'timestamp': cached_response['timestamp']
                }
        
        # Get threaded conversation context
        context = self.get_threaded_conversation_context(query, target_user_id)
        
        # Generate response with enhanced context
        result = self.gemini_client.generate_response(context)
        
        # Cache successful responses
        if result['success'] and use_cache:
            self.response_cache.cache_response(query, target_user_id, result['response'])
        
        return result
    
    def get_threaded_conversation_context(self, query: str, target_user_id: str) -> Dict[str, Any]:
        """
        Get conversation context with threaded message support
        
        Args:
            query: The input query
            target_user_id: Target user to respond as
            
        Returns:
            Context dictionary for Gemini
        """
        # Get similar threaded conversations
        similar_conversations = self.vector_store.search_similar(query, k=5)
        
        # Get user's recent threaded conversations
        user_conversations = self.vector_store.get_user_message_context(target_user_id, limit=10)
        
        # Get user info
        user_info = self.available_users.get(target_user_id, {})
        user_name = user_info.get('name', 'Unknown User')
        
        # Build enhanced context for Gemini
        context = {
            'query': query,
            'target_user': {
                'id': target_user_id,
                'name': user_name,
                'message_count': user_info.get('count', 0),
                'thread_count': user_info.get('thread_count', 0)
            },
            'similar_conversations': self._format_conversations_for_gemini(similar_conversations),
            'user_conversation_history': self._format_conversations_for_gemini(user_conversations),
            'context_type': 'threaded_conversations'
        }
        
        return context
    
    def _format_conversations_for_gemini(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format threaded conversations for Gemini understanding
        
        Args:
            conversations: List of conversation blocks
            
        Returns:
            Formatted conversations for Gemini
        """
        formatted = []
        
        for conv in conversations:
            # Parse the conversation content to understand the thread structure
            content = conv.get('content', '')
            participants = conv.get('participants', [conv.get('user_name', 'Unknown')])
            thread_length = conv.get('thread_length', 1)
            message_type = conv.get('message_type', 'message')
            
            formatted_conv = {
                'content': content,
                'author': conv.get('user_name', 'Unknown'),
                'participants': participants,
                'thread_length': thread_length,
                'message_type': message_type,
                'timestamp': conv.get('datetime', ''),
                'similarity_score': conv.get('similarity_score', 0.0)
            }
            
            # Add conversation analysis
            if message_type == 'thread' and thread_length > 1:
                formatted_conv['conversation_type'] = f"Thread with {thread_length} messages from {len(participants)} participants"
            else:
                formatted_conv['conversation_type'] = "Single message"
            
            formatted.append(formatted_conv)
        
        return formatted
    
    def chat_as_user(self, user_identifier: str, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate a response as a specific user with session context
        
        Args:
            user_identifier: User name or ID to respond as
            query: Input query
            use_cache: Whether to use cached responses
            
        Returns:
            Response dictionary
        """
        # Find user ID
        target_user_id = self._find_user_id(user_identifier)
        if not target_user_id:
            return {
                'success': False,
                'error': f'User "{user_identifier}" not found',
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        
        # Expand query with session context
        expanded_query = session_manager.expand_query(target_user_id, query)
        
        # Check cache first
        if use_cache and hasattr(self, 'response_cache'):
            cached_response = self.response_cache.get_cached_response(expanded_query, target_user_id)
            if cached_response:
                return {
                    'success': True,
                    'response': cached_response,
                    'user_name': self.available_users.get(target_user_id, {}).get('name', 'Unknown'),
                    'user_id': target_user_id,
                    'query': query,
                    'expanded_query': expanded_query,
                    'timestamp': datetime.now().isoformat(),
                    'from_cache': True
                }
        
        # Get threaded conversation context
        context = self.get_threaded_conversation_context(expanded_query, target_user_id)
        
        # Add session conversation context to the prompt
        session_context = session_manager.get_conversation_context(target_user_id)
        if session_context:
            context['session_context'] = session_context
        
        # Generate response with enhanced context
        result = self.gemini_client.generate_response(context)
        
        # Add session tracking to result
        if result['success']:
            result['expanded_query'] = expanded_query
            result['original_query'] = query
            
            # Store exchange in session
            session_manager.add_exchange(target_user_id, query, result['response'], context)
        
        # Cache successful responses
        if result['success'] and use_cache and hasattr(self, 'response_cache'):
            self.response_cache.cache_response(expanded_query, target_user_id, result['response'])
        
        return result
    
    def interactive_chat(self):
        """Start interactive chat with session management"""
        print("🤖 AI Persona Bot - Interactive Chat with Session Memory")
        print("=" * 60)
        print("Commands:")
        print("  'users' - Show available users")
        print("  'switch <user>' - Switch to different user")
        print("  'session' - Show session info")
        print("  'clear' - Clear session history")
        print("  'quit' - Exit")
        print("=" * 60)
        
        current_user = None
        
        while True:
            # Get current user if not set
            if not current_user:
                print("\nAvailable users:")
                for user_id, info in list(self.available_users.items())[:10]:
                    name = info.get('name', 'Unknown')
                    count = info.get('count', 0)
                    threads = info.get('thread_count', 0)
                    print(f"  - {name}: {count} messages ({threads} threads)")
                
                user_input = input("\nEnter user name to chat as: ").strip()
                if user_input.lower() == 'quit':
                    break
                
                user_id = self._find_user_id(user_input)
                if user_id:
                    current_user = user_id
                    user_name = self.available_users[user_id]['name']
                    print(f"\n💬 Chatting as: {user_name}")
                    print("   (Session memory is active - follow-up questions will remember context)")
                else:
                    print(f"❌ User '{user_input}' not found")
                    continue
            
            # Get user input
            try:
                user_input = input(f"\n[{self.available_users[current_user]['name']}] > ").strip()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'users':
                current_user = None
                continue
            elif user_input.lower().startswith('switch '):
                new_user = user_input[7:].strip()
                user_id = self._find_user_id(new_user)
                if user_id:
                    current_user = user_id
                    print(f"💬 Switched to: {self.available_users[user_id]['name']}")
                else:
                    print(f"❌ User '{new_user}' not found")
                continue
            elif user_input.lower() == 'session':
                session = session_manager.get_or_create_session(current_user)
                print(f"\n📊 Session Info:")
                print(f"   Session ID: {session.session_id}")
                print(f"   Exchanges: {len(session.conversation_history)}")
                print(f"   Entities tracked: {len(session.context_entities)}")
                if session.context_entities:
                    entities = list(session.context_entities.keys())[:5]
                    print(f"   Recent entities: {', '.join(entities)}")
                continue
            elif user_input.lower() == 'clear':
                if current_user in session_manager.sessions:
                    del session_manager.sessions[current_user]
                print("🗑️  Session history cleared")
                continue
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  'users' - Show available users")
                print("  'switch <user>' - Switch to different user")
                print("  'session' - Show session info")
                print("  'clear' - Clear session history")
                print("  'quit' - Exit")
                continue
            
            # Generate response
            print("🤔 Thinking...")
            result = self.chat_as_user(current_user, user_input)
            
            if result['success']:
                response = result['response']
                print(f"\n💬 {response}")
                
                # Show expansion info if query was expanded
                if result.get('expanded_query') != result.get('original_query'):
                    print(f"   💡 (Used context: {result['expanded_query']})")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    def _find_user_id(self, user_identifier: str) -> Optional[str]:
        """Find user ID by ID or name"""
        # Direct ID match
        if user_identifier in self.available_users:
            return user_identifier
        
        # Name match (case insensitive)
        for uid, info in self.available_users.items():
            if info['name'].lower() == user_identifier.lower():
                return uid
        
        # Partial name match
        for uid, info in self.available_users.items():
            if user_identifier.lower() in info['name'].lower():
                return uid
        
        return None
    
    def search_similar_conversations(self, query: str, k: int = 5, user_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar threaded conversations in the database
        
        Args:
            query: Search query
            k: Number of results to return
            user_filter: Optional user to filter by
            
        Returns:
            List of similar conversation blocks
        """
        return self.vector_store.search_similar(query, k, user_filter)
    
    def get_user_conversation_history(self, user_identifier: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent conversation blocks from a specific user
        
        Args:
            user_identifier: User ID or name
            limit: Number of conversation blocks to return
            
        Returns:
            List of user's recent conversation blocks
        """
        user_id = self._find_user_id(user_identifier)
        if not user_id:
            return []
        
        return self.vector_store.get_user_message_context(user_id, limit)
    
    def interactive_chat(self):
        """Start an interactive chat session with threaded conversation support"""
        print("\n🤖 AI Persona Bot - Interactive Mode (Threaded Conversations)")
        print("🧵 Now supports threaded conversation context!")
        print("Type 'help' for commands, 'quit' to exit")
        
        # Get index stats for display
        stats = self.vector_store.get_index_stats()
        vector_count = stats.get('total_vector_count', 0) if stats else 0
        print(f"📊 Pinecone index contains {vector_count} conversation blocks")
        
        current_user = None
        
        while True:
            try:
                if current_user:
                    user_name = self.available_users[current_user]['name']
                    prompt = f"\n[{user_name}] > "
                else:
                    prompt = "\n[No persona selected] > "
                
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._print_help()
                
                elif user_input.lower() == 'users':
                    self._print_available_users()
                
                elif user_input.startswith('persona '):
                    user_name = user_input[8:].strip()
                    user_id = self._find_user_id(user_name)
                    if user_id:
                        current_user = user_id
                        user_info = self.available_users[user_id]
                        print(f"✅ Now responding as: {user_info['name']}")
                        print(f"   📊 {user_info['count']} conversation blocks ({user_info['thread_count']} threaded)")
                    else:
                        print(f"❌ User '{user_name}' not found")
                
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    results = self.search_similar_conversations(query)
                    self._print_search_results(results)
                
                elif current_user:
                    # Generate response as current user
                    result = self.chat_as_user(current_user, user_input)
                    if result['success']:
                        print(f"\n💬 {result['response']}")
                        if result.get('from_cache'):
                            print("   (from cache)")
                    else:
                        print(f"❌ Error: {result['error']}")
                
                else:
                    print("❌ Please select a persona first using 'persona <username>'")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _print_help(self):
        """Print help information"""
        print("""
📖 Available Commands:
  help                 - Show this help message
  users               - List available users/personas
  persona <name>      - Select a persona to respond as
  search <query>      - Search for similar conversation blocks
  quit/exit/q         - Exit the bot
  
💬 Chat:
  After selecting a persona, just type your message and the bot will respond as that user.
  
🧵 Threaded Conversation Support:
  - The bot now understands conversation threads and replies
  - Context includes full conversation flows with multiple participants
  - Responses are generated based on threaded conversation patterns
        """)
    
    def _print_available_users(self):
        """Print available users with threaded conversation stats"""
        print("\n👥 Available Users/Personas:")
        users = self.list_available_users()
        for user in users[:15]:  # Show top 15
            print(f"  - {user['name']}: {user['message_count']} blocks ({user['thread_count']} threaded)")
    
    def _print_search_results(self, results: List[Dict[str, Any]]):
        """Print search results with threaded conversation info"""
        print(f"\n🔍 Found {len(results)} similar conversation blocks:")
        for i, result in enumerate(results, 1):
            timestamp = datetime.fromisoformat(result['datetime']).strftime('%Y-%m-%d %H:%M') if result['datetime'] else 'Unknown'
            thread_info = ""
            if result.get('message_type') == 'thread':
                participants = result.get('participants', [])
                thread_info = f" [Thread: {result.get('thread_length', 1)} msgs, {len(participants)} people]"
            
            # Show first line of content
            content_preview = result['content'].split('\n')[0][:100]
            print(f"{i}. [{timestamp}] {result['user_name']}: {content_preview}...{thread_info}")
            print(f"   Similarity: {result['similarity_score']:.3f}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="AI Persona Bot with Threaded Conversations")
    parser.add_argument("--data-dir", default="data", help="Directory containing message data")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild vector index from scratch")
    parser.add_argument("--pinecone-api-key", help="Pinecone API key")
    parser.add_argument("--gemini-api-key", help="Google AI API key")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--user", help="User to respond as")
    parser.add_argument("--query", help="Query to respond to")
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = AIPersonaBotThreaded(args.data_dir, args.rebuild_index, args.pinecone_api_key)
    
    # Initialize Gemini if API key provided or available in environment
    if args.gemini_api_key or os.getenv('GOOGLE_AI_API_KEY'):
        bot.initialize_gemini(args.gemini_api_key)
    
    if args.interactive:
        # Interactive mode
        bot.interactive_chat()
    elif args.user and args.query:
        # Single query mode
        result = bot.chat_as_user(args.user, args.query)
        if result['success']:
            print(f"{result['user_name']}: {result['response']}")
        else:
            print(f"Error: {result['error']}")
    else:
        # Show available users and basic info
        print("🤖 AI Persona Bot with Threaded Conversations")
        print(f"Data directory: {args.data_dir}")
        print(f"Available users: {len(bot.available_users)}")
        print("\nUse --interactive for chat mode or --help for more options.")


if __name__ == "__main__":
    main()
