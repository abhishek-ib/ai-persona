#!/usr/bin/env python3
"""
Test script to verify the search functionality improvements
"""

from ai_persona_bot_json import AIPersonaBotJSON

def test_search():
    print("🧪 Testing improved search functionality...")
    
    # Initialize the bot
    bot = AIPersonaBotJSON("data")
    bot.initialize_gemini()
    
    # Test search query
    query = "crafting documentation"
    print(f"\n🔍 Testing query: '{query}'")
    
    result = bot.search_with_gemini(query)
    
    if result['success']:
        print("✅ Search successful!")
        print(f"📝 Response: {result['response'][:200]}...")
        print(f"📚 References found: {len(result.get('references', []))}")
        
        # Show references
        references = result.get('references', [])
        if references:
            print("\n📋 Reference IDs:")
            for i, ref in enumerate(references, 1):
                ref_id = ref.get('id', 'unknown')
                print(f"  {i}. {ref_id}")
        else:
            print("⚠️  No references found - this is the issue we're trying to fix")
    else:
        print(f"❌ Search failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_search()
