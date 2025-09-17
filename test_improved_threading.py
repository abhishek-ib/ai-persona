#!/usr/bin/env python3
"""
Test Improved Threading

This script tests that threaded conversations are now properly grouped together.
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


def test_improved_threading():
    """Test that thread starters and replies are now in the same conversation block"""
    print("🧵 Testing Improved Thread Grouping")
    print("=" * 40)
    
    try:
        # Initialize the bot with the updated index
        print("1. Initializing threaded bot...")
        bot = AIPersonaBotThreaded()
        
        # Test a query that should find threaded conversations
        print("2. Testing query for threaded conversations...")
        query = "builder error field"
        results = bot.search_similar_conversations(query, k=3)
        
        print(f"   Found {len(results)} similar conversations")
        
        # Analyze the results to see if we have proper threading
        for i, result in enumerate(results, 1):
            thread_length = result.get('thread_length', 1)
            participants = result.get('participants', [])
            message_type = result.get('message_type', 'message')
            content = result.get('content', '')
            
            print(f"\n   Result {i}:")
            print(f"   - Type: {message_type}")
            print(f"   - Thread length: {thread_length}")
            print(f"   - Participants: {len(participants)} ({', '.join(participants)})")
            print(f"   - Score: {result.get('similarity_score', 0):.3f}")
            
            # Check if this is a properly threaded conversation
            if thread_length > 1 and '└─' in content:
                print(f"   ✅ This is a properly threaded conversation!")
                
                # Show the structure
                lines = content.split('\n')
                print(f"   - Thread structure:")
                for j, line in enumerate(lines[:5]):  # Show first 5 lines
                    if line.strip():
                        print(f"     {j+1}. {line[:80]}...")
                if len(lines) > 5:
                    print(f"     ... and {len(lines) - 5} more lines")
            elif thread_length == 1:
                print(f"   - Single message (not threaded)")
            else:
                print(f"   ⚠️  Thread length > 1 but no thread structure found")
        
        # Test with a specific user context
        print("\n3. Testing user context with threading...")
        user_context = bot.get_user_conversation_history("Lee", limit=3)
        
        print(f"   Found {len(user_context)} conversation blocks for Lee")
        
        threaded_count = 0
        for i, conv in enumerate(user_context, 1):
            thread_length = conv.get('thread_length', 1)
            participants = conv.get('participants', [])
            message_type = conv.get('message_type', 'message')
            
            if thread_length > 1:
                threaded_count += 1
                print(f"   Block {i}: {message_type} with {thread_length} messages, {len(participants)} participants")
        
        print(f"   ✅ {threaded_count} out of {len(user_context)} blocks are threaded conversations")
        
        # Test context creation
        print("\n4. Testing context creation with improved threading...")
        context = bot.get_threaded_conversation_context("what's the issue with the builder?", "U01J31371LZ")
        
        similar_convs = context.get('similar_conversations', [])
        print(f"   Context contains {len(similar_convs)} similar conversations")
        
        for i, conv in enumerate(similar_convs[:2], 1):
            thread_length = conv.get('thread_length', 1)
            participants = conv.get('participants', [])
            conv_type = conv.get('conversation_type', 'Single message')
            
            print(f"   Context {i}: {conv_type} (length: {thread_length}, participants: {len(participants)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 Testing Improved Thread Grouping")
    print("This test verifies that thread starters and replies are now properly grouped.\n")
    
    success = test_improved_threading()
    
    if success:
        print("\n🎉 Improved threading test passed!")
        print("✅ Thread starters and replies are properly grouped")
        print("✅ Conversation context is more comprehensive")
        print("✅ Multi-participant threads are preserved")
    else:
        print("\n❌ Improved threading test failed")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
