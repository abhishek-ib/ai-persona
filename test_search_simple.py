#!/usr/bin/env python3
"""
Simple test script for AI Persona Bot search functionality

This script provides a clean interface to test the search and clear functions
from the AI Persona Bot without the full interactive mode.
"""

import os
from ai_persona_bot_json import AIPersonaBotJSON

def test_search_functionality():
    """Test the search functionality with a simple interface"""
    
    print("🧪 Testing AI Persona Bot Search Functionality")
    print("=" * 60)
    
    # Initialize the bot
    print("🤖 Initializing AI Persona Bot...")
    bot = AIPersonaBotJSON("data")
    
    # Initialize Gemini
    print("🔑 Initializing Gemini API...")
    bot.initialize_gemini()
    
    if not bot.gemini_client:
        print("❌ Failed to initialize Gemini. Please check your API key.")
        return
    
    print("✅ Bot initialized successfully!")
    print(f"📚 Available conversations: {len(bot.vector_store.load_conversation_index())}")
    
    # Test search function
    def search_query(query: str):
        """Search for a query and return results"""
        print(f"\n🔍 Searching: '{query}'")
        print("-" * 40)
        
        result = bot.search_with_gemini(query)
        
        if result['success']:
            print(f"✅ Search successful!")
            print(f"📝 Response: {result['response']}")
            
            references = result.get('references', [])
            if references:
                print(f"📚 References found: {len(references)}")
                for i, ref in enumerate(references, 1):
                    print(f"  {i}. {ref}")
            else:
                print("⚠️  No references found")
                
            return result
        else:
            print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
            return None
    
    # Test clear function
    def clear_session():
        """Clear the search session"""
        print("\n🗑️  Clearing search session...")
        if bot.gemini_client:
            # Clear all chat sessions
            for session_id in list(bot.gemini_client.chat_sessions.keys()):
                bot.gemini_client.clear_chat_session(session_id)
            print("✅ Search session cleared")
        else:
            print("⚠️  No Gemini client to clear")
    
    # Interactive test loop
    print("\n🎯 Interactive Test Mode")
    print("Commands:")
    print("  'search <query>' - Search for a query")
    print("  'clear' - Clear search session")
    print("  'quit' - Exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n[Test] > ").strip()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            clear_session()
        elif user_input.lower().startswith('search '):
            query = user_input[7:].strip()  # Remove 'search ' prefix
            if query:
                search_query(query)
            else:
                print("❌ Please provide a search query after 'search '")
        elif user_input.lower() == 'help':
            print("\nCommands:")
            print("  'search <query>' - Search for a query")
            print("  'clear' - Clear search session")
            print("  'quit' - Exit")
            print("  'help' - Show this help")
        else:
            # Treat as direct search query
            search_query(user_input)

def run_predefined_tests():
    """Run some predefined test queries"""
    
    print("🧪 Running Predefined Search Tests")
    print("=" * 60)
    
    # Initialize the bot
    bot = AIPersonaBotJSON("data")
    bot.initialize_gemini()
    
    if not bot.gemini_client:
        print("❌ Failed to initialize Gemini. Please check your API key.")
        return
    
    # Test queries
    test_queries = [
        "crafting documentation",
        "API authentication",
        "deployment issues",
        "database problems"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        result = bot.search_with_gemini(query)
        
        if result['success']:
            print(f"✅ Success!")
            print(f"📝 Response: {result['response'][:200]}...")
            print(f"📚 References: {len(result.get('references', []))}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Clear session between tests
        if bot.gemini_client:
            for session_id in list(bot.gemini_client.chat_sessions.keys()):
                bot.gemini_client.clear_chat_session(session_id)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--predefined":
        run_predefined_tests()
    else:
        test_search_functionality()



