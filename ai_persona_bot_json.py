
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
    
    def search_with_gemini(self, query: str, session_id: str = "search_session") -> Dict[str, Any]:
        """
        Search and generate structured response with references using Gemini chat sessions
        
        Args:
            query: Search query
            session_id: Chat session identifier for search mode
            
        Returns:
            Structured response with answer and references
        """
        # Check if Gemini is initialized
        if not self.gemini_client:
            return {
                'success': False,
                'error': 'Gemini API not initialized. Please provide Google AI API key.',
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        
        # Get similar conversations (complete JSON data)
        similar_conversations = self.vector_store.search_similar_conversations(query, k=10)
        
        if not similar_conversations:
            return {
                'success': True,
                'response': f"I couldn't find any relevant conversations about '{query}'. Try different keywords or check the spelling.",
                'references': [],
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        
        # Prepare target user info for search session
        target_user = {
            'name': 'Search Assistant',
            'id': session_id
        }
        
        # Log the query and similar conversations before sending to Gemini
        print(f"\n📝 Search Query Log:")
        print(f"   Query: {query}")
        print(f"   Similar conversations found: {len(similar_conversations)}")
        
        if similar_conversations:
            print(f"   Top similar conversation files:")
            for i, file_info in enumerate(similar_conversations[:5], 1):
                score = file_info.get('similarity_score', 0)
                file_name = file_info.get('file_name', 'unknown')
                print(f"     {i}. [{score:.3f}] {file_name}")
        
        # Use Gemini chat session with search mode for structured responses
        result = self.gemini_client.generate_response(
            target_user=target_user,
            query=query,
            similar_conversations=similar_conversations,
            user_conversations=[],  # Not needed for search mode
            is_first_message=session_id not in self.gemini_client.chat_sessions,
            mode="search"
        )
        
        if result['success']:
            # Parse the structured response from chat session
            parsed_response = self._parse_structured_response(result['response'], similar_conversations)
            return {
                'success': True,
                'response': parsed_response['response'],
                'references': parsed_response['references'],
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'context_conversations': len(similar_conversations)
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error from Gemini'),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    
    def _parse_structured_response(self, response_text: str, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse Gemini's structured response and create references with full details
        """
        import json
        import re
        
        try:
            # Try to extract JSON from the response - look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group().strip()
                parsed = json.loads(json_str)
                
                # Validate that we have the expected structure
                if 'response' in parsed and 'references' in parsed:
                    # Build detailed references
                    detailed_references = []
                    referenced_ids = parsed.get('references', [])
                    
                    # Only include conversations that were actually referenced by Gemini
                    for conv in conversations:
                        if conv.get('message_details'):
                            details = conv['message_details']
                            conv_id = details.get('id', '')
                            
                            if conv_id in referenced_ids:
                                detailed_references.append({
                                    'id': details.get('id'),
                                    'text': details.get('text'),
                                    'user': details.get('user'),
                                    'timestamp': details.get('timestamp'),
                                    'type': details.get('type'),
                                    'channel_name': details.get('channel_name')
                                })
                    
                    return {
                        'response': parsed.get('response', 'No response found'),
                        'references': detailed_references
                    }
                else:
                    print("Warning: JSON response missing expected 'response' or 'references' fields")
            
        except (json.JSONDecodeError, AttributeError) as e:
            # Fallback if JSON parsing fails - return response with empty references
            print(f"Warning: Could not parse JSON response from Gemini: {e}")
            print(f"Raw response: {response_text[:200]}...")
        
        # Fallback: return the raw response with empty references since we couldn't parse which ones were actually used
        return {
            'response': response_text,
            'references': []
        }
    
    def search_mode(self):
        """Start search mode for finding relevant conversations"""
        # Check if Gemini is initialized
        if not self.gemini_client:
            print("❌ Gemini API not initialized. Please provide Google AI API key.")
            return
        
        print("🔍 Conversation Search Mode (AI-Powered)")
        print("=" * 60)
        print("🎯 Search through Slack conversations with AI analysis")
        print("🤖 Get structured responses with relevant references")
        print("🧠 Chat session maintains context across multiple searches")
        print("Commands:")
        print("  'clear' - Clear search session context")
        print("  'quit' - Exit search mode")
        print("  'help' - Show this help")
        print("=" * 60)
        
        print(f"\n🔍 Search through {len(self.vector_store.load_conversation_index())} conversations")
        print(f"   🤖 AI-powered responses using Gemini LLM")
        print(f"   📚 Structured output with response and references")
        print(f"   🔍 Hybrid search: exact keyword matches + semantic similarity")
        print(f"   🧠 Chat session provides context continuity across searches")
        
        # Use consistent session ID for search mode
        search_session_id = f"search_{int(datetime.now().timestamp())}"
        
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
            elif user_input.lower() == 'clear':
                if self.gemini_client:
                    self.gemini_client.clear_chat_session(search_session_id)
                    search_session_id = f"search_{int(datetime.now().timestamp())}"  # New session ID
                    print("🗑️  Search session cleared")
                continue
            elif user_input.lower() == 'help':
                print("\nSearch Commands:")
                print("  'clear' - Clear search session context")
                print("  'quit' - Exit search mode")
                print("  'help' - Show this help")
                print("\nTips:")
                print("  - Use specific keywords for better results")
                print("  - Try different phrasings if you don't find what you need")
                print("  - Results show similarity scores (higher = more relevant)")
                print("  - Search session maintains context across multiple queries")
                continue
            
            # Perform search with Gemini response
            print(f"🔍 Searching and analyzing: '{user_input}'...")
            
            # Use Gemini to generate structured response with references using chat session
            result = self.search_with_gemini(user_input, search_session_id)
            
            if result['success']:
                print(f"\n📝 Structured Response:")
                print("=" * 60)
                
                # Display the response
                print(f"\n🤖 AI Response:")
                print(f"{result['response']}")
                
                # Display references
                references = result.get('references', [])
                if references:
                    print(f"\n📚 References ({len(references)} conversations):")
                    print("-" * 60)
                    
                    for i, ref in enumerate(references, 1):
                        print(f"\n{i}. Reference:")
                        print(f"   🆔 ID: {ref.get('id', 'N/A')}")
                        print(f"   👤 User: {ref.get('user', 'N/A')}")
                        print(f"   ⏰ Timestamp: {ref.get('timestamp', 'N/A')}")
                        print(f"   📂 Type: {ref.get('type', 'N/A')}")
                        print(f"   📺 Channel: {ref.get('channel_name', 'N/A')}")
                        
                        # Show text preview
                        text = ref.get('text', '')
                        if text:
                            preview = (text[:150] + '...') if len(text) > 150 else text
                            print(f"   📝 Text: {preview}")
                
                # Show complete JSON output
                print(f"\n📋 Complete JSON Output:")
                print("-" * 60)
                import json
                output_json = {
                    'response': result['response'],
                    'references': references
                }
                print(json.dumps(output_json, indent=2))
                
            else:
                print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                print("💡 Make sure Gemini API is initialized and try again.")

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
    
    # Initialize Gemini for interactive and search modes
    if args.interactive or args.search or args.query:
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
