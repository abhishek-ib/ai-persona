#!/usr/bin/env python3
"""
JSON-Based AI Persona Bot

This version uses individual JSON files for each conversation, providing
complete, clean context to Gemini for much better responses.

Key improvements:
1. Complete conversation context (no fragmentation)
2. Clean sender/message format
3. Efficient JSON file retrieval
4. Enhanced session memory
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data_normalizer_json import JSONDataNormalizer
from vector_store_json import JSONVectorStore
from gemini_client_json import JSONGeminiClient


class AIPersonaBotJSON:
    def __init__(self, data_dir: str = "data", rebuild_index: bool = False, 
                 pinecone_api_key: Optional[str] = None):
        """
        Initialize JSON-based AI Persona Bot
        
        Args:
            data_dir: Directory containing Slack dump folders
            rebuild_index: Whether to rebuild the index from scratch
            pinecone_api_key: Pinecone API key
        """
        self.data_dir = data_dir
        self.json_dir = "generated"
        self.gemini_client = None
        
        print("🤖 Initializing JSON-based AI Persona Bot...")
        print("🎯 Using complete conversation JSON files for better context")
        
        # Initialize components
        self._initialize(rebuild_index, pinecone_api_key)
        
        print("✅ JSON-based AI Persona Bot initialized successfully!")
    
    def _initialize(self, rebuild_index: bool, pinecone_api_key: Optional[str]):
        """Initialize all components"""
        
        # Check if we need to normalize data
        if rebuild_index or not os.path.exists(os.path.join(self.json_dir, 'conversation_index.json')):
            print("🔄 Normalizing data into JSON files...")
            normalizer = JSONDataNormalizer(self.data_dir, self.json_dir)
            conversations = normalizer.normalize_all_data()
            print(f"✅ Created {len(conversations)} conversation JSON files")
        
        # Initialize vector store
        print("🔍 Initializing JSON vector store...")
        self.vector_store = JSONVectorStore(
            api_key=pinecone_api_key,
            json_dir=self.json_dir
        )
        
        # Build index if needed
        if rebuild_index:
            print("🔄 Building Pinecone index...")
            self.vector_store.build_index()
        
        # Load available users
        self._load_available_users()
    
    def _load_available_users(self):
        """Load available users from conversation index"""
        try:
            conversations = self.vector_store.load_conversation_index()
            
            # Count conversations per user
            user_stats = {}
            for conv in conversations:
                for participant in conv.get('participants', []):
                    if participant not in user_stats:
                        user_stats[participant] = {
                            'name': participant,
                            'conversation_count': 0,
                            'thread_count': 0,
                            'dm_count': 0,
                            'channel_count': 0
                        }
                    
                    user_stats[participant]['conversation_count'] += 1
                    
                    if conv.get('type') == 'dm':
                        user_stats[participant]['dm_count'] += 1
                    elif conv.get('type') == 'channel':
                        user_stats[participant]['channel_count'] += 1
                    
                    if conv.get('conversation_type') == 'thread':
                        user_stats[participant]['thread_count'] += 1
            
            # Filter users with sufficient data (at least 5 conversations)
            self.available_users = {
                user: stats for user, stats in user_stats.items()
                if stats['conversation_count'] >= 5
            }
            
            print(f"📋 Found {len(self.available_users)} users with sufficient conversation history:")
            
            # Show top users
            sorted_users = sorted(
                self.available_users.items(),
                key=lambda x: x[1]['conversation_count'],
                reverse=True
            )
            
            for user, stats in sorted_users[:10]:
                conv_count = stats['conversation_count']
                thread_count = stats['thread_count']
                dm_count = stats['dm_count']
                channel_count = stats['channel_count']
                print(f"  - {user}: {conv_count} conversations ({thread_count} threads, {dm_count} DMs, {channel_count} channels)")
            
        except Exception as e:
            print(f"Warning: Could not load user statistics: {e}")
            self.available_users = {}
    
    def initialize_gemini(self, api_key: Optional[str] = None):
        """Initialize Gemini client"""
        try:
            self.gemini_client = JSONGeminiClient(api_key=api_key)
            print("✅ Gemini API connected successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            print("   Set GOOGLE_AI_API_KEY environment variable or provide api_key")
    
    def chat(self, query: str, session_id: str = "default", is_first_message: bool = False) -> Dict[str, Any]:
        """
        Generate a helpful coworker response with relevant conversation context
        
        Args:
            query: Input query
            session_id: Chat session identifier
            is_first_message: Whether this is the first message (triggers setup)
            
        Returns:
            Response dictionary
        """
        # Check if Gemini is initialized
        if not self.gemini_client:
            return {
                'success': False,
                'error': 'Gemini API not initialized. Please provide Google AI API key.',
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        
        # Get similar conversations (complete JSON data) - only 1-2 most relevant
        similar_conversations = self.vector_store.search_similar_conversations(query, k=20)
        
        # Prepare dummy target user info for logging
        target_user = {
            'name': 'Helpful Coworker',
            'id': session_id
        }
        
        
        # Log the query and similar conversations before sending to Gemini
        print(f"\n📝 Bot Query Log:")
        print(f"   Query: {query}")
        print(f"   Similar conversations found: {len(similar_conversations)}")
        
        if similar_conversations:
            print(f"   Top similar conversation files:")
            for i, file_info in enumerate(similar_conversations[:3], 1):
                score = file_info.get('similarity_score', 0)
                file_name = file_info.get('file_name', 'unknown')
                print(f"     {i}. [{score:.3f}] {file_name}")
        
        # Generate response using Gemini chat session with attached conversations
        result = self.gemini_client.generate_response(
            target_user=target_user,
            query=query,
            similar_conversations=similar_conversations,
            user_conversations=[],  # Not needed for helpful coworker mode
            is_first_message=is_first_message
        )
        
        return result
    
    def search_mode(self):
        """Start search mode for finding relevant conversations"""
        print("🔍 Conversation Search Mode")
        print("=" * 60)
        print("🎯 Search through Slack conversations to find relevant information")
        print("📚 Get detailed results with conversation context and sources")
        print("Commands:")
        print("  'quit' - Exit search mode")
        print("  'help' - Show this help")
        print("=" * 60)
        
        print(f"\n🔍 Search through {len(self.vector_store.load_conversation_index())} conversations")
        print(f"   💡 Enter your search query to find relevant discussions")
        print(f"   📄 Results will show conversation files and similarity scores")
        
        while True:
            # Get user input
            try:
                user_input = input(f"\n[Search] > ").strip()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("\nSearch Commands:")
                print("  'quit' - Exit search mode")
                print("  'help' - Show this help")
                print("\nTips:")
                print("  - Use specific keywords for better results")
                print("  - Try different phrasings if you don't find what you need")
                print("  - Results show similarity scores (higher = more relevant)")
                continue
            
            # Perform search
            print(f"🔍 Searching for: '{user_input}'...")
            
            # Get similar conversations with higher limit for search mode
            similar_conversations = self.vector_store.search_similar_conversations(user_input, k=10)
            
            if similar_conversations:
                print(f"\n📋 Found {len(similar_conversations)} relevant conversations:")
                print("=" * 60)
                
                for i, file_info in enumerate(similar_conversations, 1):
                    score = file_info.get('similarity_score', 0)
                    file_name = file_info.get('file_name', 'unknown')
                    
                    # Try to load the JSON to get more details
                    try:
                        import json
                        raw_json = file_info.get('raw_json_content', '')
                        if raw_json:
                            conv_data = json.loads(raw_json)
                            channel_name = conv_data.get('channel_name', 'Unknown')
                            conv_type = conv_data.get('conversation_type', 'single')
                            participants = conv_data.get('participants', [])
                            message_count = len(conv_data.get('messages', []))
                            
                            # Get first message preview
                            messages = conv_data.get('messages', [])
                            first_msg_preview = ""
                            if messages:
                                first_msg = messages[0].get('message', '')
                                first_msg_preview = (first_msg[:100] + '...') if len(first_msg) > 100 else first_msg
                            
                            print(f"\n{i}. [{score:.3f}] {channel_name} - {conv_type}")
                            print(f"   👥 Participants: {', '.join(participants[:3])}")
                            print(f"   💬 Messages: {message_count}")
                            print(f"   📄 File: {file_name}")
                            if first_msg_preview:
                                print(f"   📝 Preview: {first_msg_preview}")
                    except:
                        # Fallback to basic info if JSON parsing fails
                        print(f"\n{i}. [{score:.3f}] {file_name}")
                
                print("\n" + "=" * 60)
                print(f"💡 Tip: Use these results to ask more specific questions!")
            else:
                print(f"\n❌ No conversations found for: '{user_input}'")
                print("💡 Try:")
                print("   - Different keywords or phrases")
                print("   - More general terms")
                print("   - Check spelling")

    def interactive_chat(self):
        """Start interactive chat as helpful coworker"""
        print("🤖 Helpful Coworker AI - Interactive Chat")
        print("=" * 60)
        print("🎯 I'll help answer questions using relevant Slack conversations!")
        print("🧠 Gemini 2.5 Flash automatically maintains conversation context")
        print("📎 Each question gets 1-2 most relevant conversation attachments")
        print("Commands:")
        print("  'clear' - Clear chat session")
        print("  'quit' - Exit")
        print("=" * 60)
        
        session_id = f"session_{int(datetime.now().timestamp())}"
        is_first_message = True
        
        print(f"\n🤖 Hi! I'm your helpful coworker AI.")
        print(f"   📚 I have access to {len(self.vector_store.load_conversation_index())} conversations")
        print(f"   💬 Ask me anything and I'll find relevant discussions to help!")
        print(f"   🚀 Powered by Gemini 2.5 Flash with complete conversation context")
        
        while True:
            # Get user input
            try:
                user_input = input(f"\n[You] > ").strip()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                if self.gemini_client:
                    self.gemini_client.clear_chat_session(session_id)
                    is_first_message = True  # Next message will reinitialize
                    print("🗑️  Chat session cleared")
                continue
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  'clear' - Clear chat session")
                print("  'quit' - Exit")
                continue
            
            # Generate response
            if is_first_message:
                print("🧠 Setting up helpful coworker mode...")
            else:
                print("🤔 Looking for relevant conversations...")
            
            result = self.chat(user_input, session_id, is_first_message)
            
            if result['success']:
                response = result['response']
                print(f"\n🤖 {response}")
                
                # Show context info
                context_count = result.get('context_conversations', 0)
                
                if is_first_message:
                    print(f"   🎯 (Helpful coworker mode initialized)")
                else:
                    print(f"   📎 (Found {context_count} relevant conversations)")
                
                is_first_message = False  # Subsequent messages use existing session
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="JSON-based AI Persona Bot")
    parser.add_argument("--data-dir", default="data", help="Directory containing Slack dumps")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild index from scratch")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")
    parser.add_argument("--search", action="store_true", help="Start search mode")
    parser.add_argument("--user", help="User to respond as")
    parser.add_argument("--query", help="Query to respond to")
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = AIPersonaBotJSON(args.data_dir, args.rebuild_index)
    
    # Initialize Gemini (only if not in search-only mode)
    if not args.search:
        bot.initialize_gemini()
    
    if args.interactive:
        bot.interactive_chat()
    elif args.search:
        bot.search_mode()
    elif args.query:
        result = bot.chat(args.query, is_first_message=True)
        if result['success']:
            print(f"🤖 {result['response']}")
        else:
            print(f"Error: {result['error']}")
    else:
        print("🤖 Helpful Coworker AI")
        print(f"📚 {len(bot.vector_store.load_conversation_index())} conversations available")
        print("\nUse --interactive for chat mode, --search for search mode, or --help for more options.")


if __name__ == "__main__":
    main()
