#!/usr/bin/env python3
"""
Test script for the new summarize functionality
"""

import json
from ai_persona_bot_json import AIPersonaBotJSON

def test_summarize_functionality():
    """Test the summarize functionality with sample data"""
    
    # Sample conversation data matching the format you provided
    sample_conversation = {
        "id": "1757963391.139329",
        "text": "Honestly, I'm surprised there's no meme teams this year",
        "user": {
            "id": "U011UPRAPNV",
            "displayName": "Paul Hoang",
            "avatarUrl": "https://avatars.slack-edge.com/2020-04-10/1053764284707_a6d73dc101b681df17db_original.jpg"
        },
        "timestamp": "1757963391.139329",
        "isAi": False,
        "thread": [
            {
                "id": "1757963546.954369",
                "text": "the vibes are so down in the dumps nobodys having funnnnnnnn",
                "user": {
                    "id": "U03JW1JAW1G",
                    "displayName": "Natasha Puthukudy",
                    "avatarUrl": "https://avatars.slack-edge.com/2024-07-12/7416761356226_24f05f3b1209cb36766c_original.jpg"
                },
                "timestamp": "1757963546.954369",
                "isAi": False
            },
            {
                "id": "1757963559.514649",
                "text": "True true",
                "user": {
                    "id": "U011UPRAPNV",
                    "displayName": "Paul Hoang",
                    "avatarUrl": "https://avatars.slack-edge.com/2020-04-10/1053764284707_a6d73dc101b681df17db_original.jpg"
                },
                "timestamp": "1757963559.514649",
                "isAi": False
            }
        ]
    }
    
    print("🧪 Testing Summarize Functionality")
    print("=" * 50)
    
    # Initialize bot
    try:
        bot = AIPersonaBotJSON()
        bot.initialize_gemini()
        
        print("✅ Bot initialized successfully")
        
        # Test summarize_chat method
        print("\n📝 Testing summarize_chat method...")
        result = bot.summarize_chat(sample_conversation)
        
        if result['success']:
            print("✅ Summarize successful!")
            print(f"\n📝 Summary:")
            print(f"{result['summary']}")
            
            print(f"\n📋 Details:")
            print(f"   🆔 Conversation ID: {result['conversation_id']}")
            print(f"   💬 Main Message: {result['main_message']}")
            print(f"   🧵 Thread Replies: {result['thread_count']}")
            print(f"   ⏰ Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Summarize failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure you have:")
        print("   - GOOGLE_AI_API_KEY environment variable set")
        print("   - Required dependencies installed")
        print("   - Data directory with conversations available")

if __name__ == "__main__":
    test_summarize_functionality()
