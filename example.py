#!/usr/bin/env python3
"""
Example usage of AI Persona Bot

This script demonstrates how to use the AI Persona Bot programmatically.
"""

import os
from ai_persona_bot import AIPersonaBot


def main():
    """Example usage of the AI Persona Bot"""
    print("🤖 AI Persona Bot Example")
    print("=" * 40)
    
    # Initialize the bot
    print("Initializing bot...")
    bot = AIPersonaBot(data_dir="data")
    
    # Initialize Gemini (make sure GOOGLE_AI_API_KEY is set)
    if not bot.initialize_gemini():
        print("❌ Failed to initialize Gemini API")
        print("Make sure GOOGLE_AI_API_KEY environment variable is set")
        return
    
    # List available users
    print("\n👥 Available Users:")
    users = bot.list_available_users()
    for i, user in enumerate(users[:5], 1):  # Show top 5 users
        print(f"{i}. {user['name']} ({user['message_count']} messages)")
    
    if not users:
        print("No users found with sufficient message history")
        return
    
    # Use the first user as an example
    example_user = users[0]['name']
    print(f"\n🎭 Using {example_user} as example persona")
    
    # Example queries
    example_queries = [
        "How's the project going?",
        "Any updates on the deployment?",
        "What do you think about the new feature?",
        "Need any help with anything?",
        "How was your day?"
    ]
    
    print(f"\n💬 Example conversations with {example_user}:")
    print("-" * 50)
    
    for query in example_queries:
        print(f"\n👤 You: {query}")
        
        # Generate response
        result = bot.chat_as_user(example_user, query)
        
        if result['success']:
            print(f"🤖 {example_user}: {result['response']}")
            if result.get('from_cache'):
                print("   📝 (cached response)")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Demonstrate search functionality
    print(f"\n🔍 Searching for similar messages about 'deployment':")
    search_results = bot.search_similar_messages("deployment", k=3)
    
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result['user_name']}: {result['content'][:80]}...")
        print(f"   Similarity: {result['similarity_score']:.3f}")
    
    # Show user context
    print(f"\n📚 Recent messages from {example_user}:")
    user_context = bot.get_user_context(example_user, limit=3)
    
    for i, msg in enumerate(user_context, 1):
        print(f"{i}. {msg['content'][:80]}...")
    
    print("\n✅ Example completed!")
    print("Run 'python ai_persona_bot.py --interactive' for interactive mode")


if __name__ == "__main__":
    main()
