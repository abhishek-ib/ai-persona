#!/usr/bin/env python3
"""
Example usage of AI Persona Bot with Pinecone

This script demonstrates how to use the Pinecone-powered AI Persona Bot.
"""

import os
from ai_persona_bot_pinecone import AIPersonaBotPinecone


def main():
    """Example usage of the AI Persona Bot with Pinecone"""
    print("🚀 AI Persona Bot with Pinecone - Example")
    print("=" * 50)
    
    # Initialize the bot
    print("Initializing bot with Pinecone...")
    try:
        bot = AIPersonaBotPinecone(data_dir="data")
    except Exception as e:
        print(f"❌ Failed to initialize Pinecone bot: {e}")
        print("Make sure PINECONE_API_KEY is set in your environment")
        return
    
    # Initialize Gemini (make sure GOOGLE_AI_API_KEY is set)
    if not bot.initialize_gemini():
        print("❌ Failed to initialize Gemini API")
        print("Make sure GOOGLE_AI_API_KEY environment variable is set")
        return
    
    # Show Pinecone statistics
    print("\n📊 Pinecone Index Statistics:")
    stats = bot.get_pinecone_stats()
    if stats:
        print(f"  Total vectors: {stats.get('total_vector_count', 0):,}")
        print(f"  Dimension: {stats.get('dimension', 'Unknown')}")
    else:
        print("  Could not retrieve statistics")
    
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
        content = result.get('content', '')[:80]
        user_name = result.get('user_name', 'Unknown')
        print(f"{i}. {user_name}: {content}...")
        print(f"   Similarity: {result['similarity_score']:.3f}")
    
    # Show user context
    print(f"\n📚 Recent messages from {example_user}:")
    user_context = bot.get_user_context(example_user, limit=3)
    
    for i, msg in enumerate(user_context, 1):
        content = msg.get('content', '')[:80]
        print(f"{i}. {content}...")
    
    print("\n✅ Pinecone example completed!")
    print("Benefits of Pinecone version:")
    print("  🚀 Scales to millions of messages")
    print("  ⚡ Fast similarity search")
    print("  🌐 Cloud-based storage")
    print("  🔄 No local index files needed")
    print("\nRun 'python ai_persona_bot_pinecone.py --interactive' for interactive mode")


if __name__ == "__main__":
    main()
