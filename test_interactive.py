#!/usr/bin/env python3
"""
Interactive test script for AI Persona Bot with Paul's persona

This script demonstrates how to use the search_with_gemini function
with mode="interactive" to get responses in Paul Hoang's voice.
"""

from ai_persona_bot_json import AIPersonaBotJSON

def main():
    """Demonstrate interactive query functionality with Paul's persona"""
    
    print("🧪 Interactive AI Persona Bot Test (Paul's Voice)")
    print("=" * 60)
    
    # Step 1: Initialize the bot
    print("1. Initializing bot...")
    bot = AIPersonaBotJSON("data")
    bot.initialize_gemini()
    
    if not bot.gemini_client:
        print("❌ Failed to initialize Gemini")
        return
    
    print("✅ Bot initialized")
    
    # Step 2: Test interactive queries with Paul's persona
    print("\n2. Testing interactive queries with Paul's persona...")
    
    test_queries = [
        "What's your favorite Pokemon game?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        result = bot.search_with_gemini(query, mode="interactive")
        
        if result['success']:
            print(f"✅ Paul's Response:")
            print("-" * 80)
            print(result['response'])
            print("-" * 80)
            print(f"📚 References: {len(result.get('references', []))}")
        else:
            print(f"❌ Query failed: {result.get('error')}")
    
    # Step 3: Test session continuity
    print("\n3. Testing session continuity...")
    follow_up_query = "Can you tell me more about that?"
    print(f"Follow-up query: {follow_up_query}")
    
    result = bot.search_with_gemini(follow_up_query, mode="interactive")
    
    if result['success']:
        print(f"✅ Paul's Follow-up Response:")
        print("-" * 80)
        print(result['response'])
        print("-" * 80)
    else:
        print(f"❌ Follow-up failed: {result.get('error')}")
    
    # Step 4: Test clear function
    print("\n4. Testing clear function...")
    print(f"   Sessions before clear: {len(bot.gemini_client.chat_sessions)}")
    
    # Clear all sessions
    for session_id in list(bot.gemini_client.chat_sessions.keys()):
        bot.gemini_client.clear_chat_session(session_id)
    
    print(f"   Sessions after clear: {len(bot.gemini_client.chat_sessions)}")
    print("✅ Clear function works")
    

if __name__ == "__main__":
    main()
