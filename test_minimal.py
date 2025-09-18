#!/usr/bin/env python3
"""
Minimal test script for AI Persona Bot search and clear functions

This script demonstrates how to use the search and clear functions
from the AI Persona Bot programmatically.
"""

import uuid
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
    session_id = str(uuid.uuid4())
    print(f"🆔 Generated random session ID: {session_id}")
    result = bot.search_with_gemini(query, session_id=session_id)
    
    if result['success']:
        print(f"✅ Search successful!")
        print(f"📝 Response:")
        print("-" * 80)
        print(result['response'])
        print("-" * 80)
        print(f"📚 References: {len(result.get('references', []))}")
        print(f"📚 References: {result.get('references', [])}")
    else:
        print(f"❌ Search failed: {result.get('error')}")
    
    # Step 3: Test clear function
    print("\n3. Testing clear function...")
    print(f"   Sessions before clear: {len(bot.gemini_client.chat_sessions)}")
    
    # Clear the specific session we created
    print(f"🧹 Cleaning up session: {session_id}")
    bot.gemini_client.clear_chat_session(session_id)
    
    print(f"   Sessions after clear: {len(bot.gemini_client.chat_sessions)}")
    print("✅ Session cleaned up successfully")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()



