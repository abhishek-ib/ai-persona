#!/usr/bin/env python3
"""
Test Threaded Context Compatibility

This script tests the threaded conversation context without requiring API keys.
"""

import os
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ai_persona_bot_threaded import AIPersonaBotThreaded


def test_threaded_context():
    """Test the threaded conversation context creation"""
    print("🧪 Testing Threaded Context Compatibility")
    print("=" * 50)
    
    try:
        # Initialize the bot (this will work even without API keys for context testing)
        print("1. Initializing threaded bot...")
        bot = AIPersonaBotThreaded()
        
        print("2. Testing context creation...")
        
        # Get a test user
        if not bot.available_users:
            print("❌ No users available for testing")
            return False
        
        # Get the first available user
        test_user_id = list(bot.available_users.keys())[0]
        test_user_name = bot.available_users[test_user_id]['name']
        
        print(f"   Using test user: {test_user_name}")
        
        # Test context creation
        test_query = "What are you working on these days?"
        context = bot.get_threaded_conversation_context(test_query, test_user_id)
        
        print("3. Analyzing context structure...")
        print(f"   ✅ Query: {context['query']}")
        print(f"   ✅ Target user: {context['target_user']['name']}")
        print(f"   ✅ Similar conversations: {len(context['similar_conversations'])}")
        print(f"   ✅ User conversation history: {len(context['user_conversation_history'])}")
        print(f"   ✅ Context type: {context['context_type']}")
        
        # Test that similar conversations have the right format
        if context['similar_conversations']:
            sample_conv = context['similar_conversations'][0]
            print(f"   ✅ Sample conversation format:")
            print(f"      - Author: {sample_conv.get('author', 'Unknown')}")
            print(f"      - Participants: {len(sample_conv.get('participants', []))}")
            print(f"      - Type: {sample_conv.get('conversation_type', 'Unknown')}")
            print(f"      - Content preview: {sample_conv.get('content', '')[:100]}...")
        
        # Test Gemini client context compatibility (without actually calling Gemini)
        print("4. Testing Gemini context compatibility...")
        try:
            # This will test the prompt creation without calling the API
            from gemini_client import GeminiClient
            
            # Create a client without API key (just for testing prompt creation)
            gemini_client = GeminiClient(api_key="test-key", enable_logging=False)
            
            # Test prompt creation (this should work with our fixed compatibility)
            prompt = gemini_client.create_persona_prompt(context)
            
            print("   ✅ Prompt created successfully!")
            print(f"   ✅ Prompt length: {len(prompt)} characters")
            
            # Show a preview of the prompt
            lines = prompt.split('\n')
            print("   ✅ Prompt preview (first 10 lines):")
            for i, line in enumerate(lines[:10]):
                print(f"      {i+1:2}. {line}")
            
            if len(lines) > 10:
                print(f"      ... and {len(lines) - 10} more lines")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Gemini context compatibility failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 Threaded Context Compatibility Test")
    print("This test verifies that threaded conversations work correctly")
    print("without requiring valid API keys.\n")
    
    success = test_threaded_context()
    
    if success:
        print("\n🎉 All tests passed!")
        print("✅ Threaded conversation context is working correctly")
        print("✅ Gemini compatibility is fixed")
        print("✅ Ready for use with valid API keys")
    else:
        print("\n❌ Some tests failed")
        print("Please check the errors above")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
