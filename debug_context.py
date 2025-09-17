#!/usr/bin/env python3
"""
Debug Context Format

This script helps debug what context format is being generated.
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


def debug_context_format():
    """Debug the context format being generated"""
    print("🔧 Debugging Context Format")
    print("=" * 40)
    
    try:
        # Initialize the bot
        print("1. Initializing threaded bot...")
        bot = AIPersonaBotThreaded()
        
        # Get a test user
        test_user_id = list(bot.available_users.keys())[0]
        test_user_name = bot.available_users[test_user_id]['name']
        print(f"2. Using test user: {test_user_name} ({test_user_id})")
        
        # Test context creation
        test_query = "what are you upto?"
        print(f"3. Creating context for query: '{test_query}'")
        
        context = bot.get_threaded_conversation_context(test_query, test_user_id)
        
        print("4. Context structure analysis:")
        print(f"   Keys in context: {list(context.keys())}")
        
        if 'target_user' in context:
            print(f"   target_user type: {type(context['target_user'])}")
            print(f"   target_user content: {context['target_user']}")
        
        if 'target_user_name' in context:
            print(f"   target_user_name: {context['target_user_name']}")
        
        print(f"   context_type: {context.get('context_type', 'Not specified')}")
        
        # Test the detection logic
        print("5. Testing format detection:")
        has_target_user = 'target_user' in context
        is_dict = isinstance(context.get('target_user'), dict) if has_target_user else False
        
        print(f"   'target_user' in context: {has_target_user}")
        print(f"   isinstance(target_user, dict): {is_dict}")
        print(f"   Detection result: {'Threaded format' if (has_target_user and is_dict) else 'Legacy format'}")
        
        # Test prompt creation
        print("6. Testing prompt creation...")
        try:
            from gemini_client import GeminiClient
            gemini_client = GeminiClient(api_key="test-key", enable_logging=False)
            
            prompt = gemini_client.create_persona_prompt(context)
            print(f"   ✅ Prompt created successfully! Length: {len(prompt)} chars")
            
            # Show first few lines
            lines = prompt.split('\n')[:5]
            print("   First few lines:")
            for i, line in enumerate(lines, 1):
                print(f"     {i}. {line}")
            
        except Exception as e:
            print(f"   ❌ Prompt creation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    debug_context_format()
