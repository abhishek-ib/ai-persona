#!/usr/bin/env python3
"""
AI Persona Bot - Minimal Version

This is a lightweight version that works without PyTorch/sentence-transformers.
Uses TF-IDF instead of neural embeddings.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

from data_normalizer import DataNormalizer
from vector_store_minimal import MinimalVectorStore
from gemini_client import GeminiClient, ResponseCache


class AIPersonaBotMinimal:
    def __init__(self, data_dir: str = "data", rebuild_index: bool = False):
        """
        Initialize the AI Persona Bot (Minimal Version)
        
        Args:
            data_dir: Directory containing Slack message data
            rebuild_index: Whether to rebuild the vector index from scratch
        """
        self.data_dir = data_dir
        self.normalizer = DataNormalizer(data_dir)
        self.vector_store = MinimalVectorStore()
        self.gemini_client = None
        self.response_cache = ResponseCache()
        
        # Available users (will be populated after data loading)
        self.available_users = {}
        
        # Initialize the bot
        self._initialize(rebuild_index)
    
    def _initialize(self, rebuild_index: bool = False):
        """Initialize the bot by loading/building necessary components"""
        print("🤖 Initializing AI Persona Bot (Minimal Version)...")
        print("📝 Using TF-IDF embeddings instead of neural embeddings")
        
        # Check if we need to rebuild the index
        index_exists = os.path.exists("message_index_minimal.pkl") and os.path.exists("message_metadata_minimal.pkl")
        normalized_data_exists = os.path.exists("normalized_messages.json")
        
        if rebuild_index or not index_exists or not normalized_data_exists:
            print("📊 Normalizing message data...")
            messages = self.normalizer.normalize_all_data()
            self.normalizer.save_normalized_data(messages)
            
            print("🔍 Building TF-IDF index...")
            self.vector_store.build_index(messages)
            self.vector_store.save_index()
        else:
            print("📚 Loading existing TF-IDF index...")
            self.vector_store.load_index()
        
        # Load available users
        self._load_available_users()
        
        print("✅ AI Persona Bot (Minimal) initialized successfully!")
    
    def _load_available_users(self):
        """Load available users from the normalized data"""
        try:
            with open("normalized_messages.json", 'r') as f:
                messages = json.load(f)
            
            # Count messages per user
            user_counts = {}
            for message in messages:
                user_id = message['user_id']
                user_name = message['user_name']
                if user_id not in user_counts:
                    user_counts[user_id] = {
                        'name': user_name,
                        'count': 0
                    }
                user_counts[user_id]['count'] += 1
            
            # Only include users with significant message history
            self.available_users = {
                uid: info for uid, info in user_counts.items()
                if info['count'] >= 5  # At least 5 messages
            }
            
            print(f"📋 Found {len(self.available_users)} users with sufficient message history:")
            for uid, info in sorted(self.available_users.items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"  - {info['name']}: {info['count']} messages")
                
        except Exception as e:
            print(f"Error loading users: {e}")
            self.available_users = {}
    
    def initialize_gemini(self, api_key: Optional[str] = None):
        """Initialize Gemini client with API key"""
        try:
            self.gemini_client = GeminiClient(api_key)
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
                'message_count': info['count']
            }
            for uid, info in sorted(self.available_users.items(), key=lambda x: x[1]['count'], reverse=True)
        ]
    
    def chat_as_user(self, user_identifier: str, query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Generate a response as a specific user
        
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
        
        # Get conversation context
        context = self.vector_store.get_conversation_context(query, target_user_id)
        
        # Generate response
        result = self.gemini_client.generate_response(context)
        
        # Cache successful responses
        if result['success'] and use_cache:
            self.response_cache.cache_response(query, target_user_id, result['response'])
        
        return result
    
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
    
    def search_similar_messages(self, query: str, k: int = 5, user_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar messages in the database
        
        Args:
            query: Search query
            k: Number of results to return
            user_filter: Optional user to filter by
            
        Returns:
            List of similar messages
        """
        return self.vector_store.search_similar(query, k, user_filter)
    
    def get_user_context(self, user_identifier: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent messages from a specific user
        
        Args:
            user_identifier: User ID or name
            limit: Number of messages to return
            
        Returns:
            List of user's recent messages
        """
        user_id = self._find_user_id(user_identifier)
        if not user_id:
            return []
        
        return self.vector_store.get_user_message_context(user_id, limit)
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n🤖 AI Persona Bot - Interactive Mode (Minimal Version)")
        print("📝 Using TF-IDF for similarity search (no PyTorch required)")
        print("Type 'help' for commands, 'quit' to exit")
        
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
                        print(f"✅ Now responding as: {self.available_users[user_id]['name']}")
                    else:
                        print(f"❌ User '{user_name}' not found")
                
                elif user_input.startswith('search '):
                    query = user_input[7:].strip()
                    results = self.search_similar_messages(query)
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
  search <query>      - Search for similar messages (using TF-IDF)
  quit/exit/q         - Exit the bot
  
💬 Chat:
  After selecting a persona, just type your message and the bot will respond as that user.
  
📝 Note: This minimal version uses TF-IDF instead of neural embeddings.
        """)
    
    def _print_available_users(self):
        """Print available users"""
        print("\n👥 Available Users/Personas:")
        users = self.list_available_users()
        for user in users:
            print(f"  - {user['name']} ({user['message_count']} messages)")
    
    def _print_search_results(self, results: List[Dict[str, Any]]):
        """Print search results"""
        print(f"\n🔍 Found {len(results)} similar messages (TF-IDF similarity):")
        for i, result in enumerate(results, 1):
            timestamp = datetime.fromisoformat(result['datetime']).strftime('%Y-%m-%d %H:%M') if result['datetime'] else 'Unknown'
            print(f"{i}. [{timestamp}] {result['user_name']}: {result['content'][:100]}...")
            print(f"   Similarity: {result['similarity_score']:.3f}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="AI Persona Bot (Minimal Version)")
    parser.add_argument("--data-dir", default="data", help="Directory containing message data")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild TF-IDF index from scratch")
    parser.add_argument("--api-key", help="Google AI API key")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--user", help="User to respond as")
    parser.add_argument("--query", help="Query to respond to")
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = AIPersonaBotMinimal(args.data_dir, args.rebuild_index)
    
    # Initialize Gemini if API key provided or available in environment
    if args.api_key or os.getenv('GOOGLE_AI_API_KEY'):
        bot.initialize_gemini(args.api_key)
    
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
        print("🤖 AI Persona Bot (Minimal Version)")
        print("📝 Uses TF-IDF embeddings (no PyTorch required)")
        print(f"Data directory: {args.data_dir}")
        print(f"Available users: {len(bot.available_users)}")
        print("\nUse --interactive for chat mode or --help for more options.")


if __name__ == "__main__":
    main()
