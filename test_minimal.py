#!/usr/bin/env python3
"""
Minimal test script for AI Persona Bot search and clear functions

This script demonstrates how to use the search and clear functions
from the AI Persona Bot programmatically.
"""

from ai_persona_bot_json import AIPersonaBotJSON

def main():
    """Demonstrate search and clear functionality"""
    
    print("🧪 Minimal AI Persona Bot Test")
    print("=" * 50)
    
    # Step 1: Initialize the bot
    print("1. Initializing bot...")
    bot = AIPersonaBotJSON("data")
    bot.initialize_gemini()
    
    if not bot.gemini_client:
        print("❌ Failed to initialize Gemini")
        return
    
    print("✅ Bot initialized")
    
    # Step 2: Test search function
    print("\n2. Testing search function...")
    query = "How to get started with the crafting?"
    result = bot.search_with_gemini(query)
    
    if result['success']:
        print(f"✅ Search successful!")
        print(f"📝 Response:")
        print("-" * 80)
        print(result['response'])
        print("-" * 80)
        print(f"📚 References: {len(result.get('references', []))}")
    else:
        print(f"❌ Search failed: {result.get('error')}")
    
    # Step 3: Test clear function
    print("\n3. Testing clear function...")
    print(f"   Sessions before clear: {len(bot.gemini_client.chat_sessions)}")
    
    # Clear all sessions
    for session_id in list(bot.gemini_client.chat_sessions.keys()):
        bot.gemini_client.clear_chat_session(session_id)
    
    print(f"   Sessions after clear: {len(bot.gemini_client.chat_sessions)}")
    print("✅ Clear function works")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()



