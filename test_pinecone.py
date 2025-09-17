#!/usr/bin/env python3
"""
Test Pinecone Connection

This script tests the Pinecone API connection and setup.
"""

import os
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vector_store_pinecone import PineconeVectorStore


def get_sample_messages():
    """Create sample messages for testing based on actual AIHub feedback patterns"""
    return [
        {
            'user_id': 'U001',
            'user_name': 'Sarah Chen',
            'user_display_name': 'Sarah',
            'content': 'In Build, after I added documents to the project, I created my first class. When I re-classify, my grouping from before is gone. Is this on purpose?',
            'timestamp': 1694865000.0,
            'datetime': '2023-09-16T10:30:00',
            'message_type': 'message'
        },
        {
            'user_id': 'U002',
            'user_name': 'Mike Rodriguez',
            'user_display_name': 'Mike',
            'content': 'Sometimes it takes 15+ minutes to upload documents into Converse. We tested this on documents of 20-140 pages. The latency is concerning.',
            'timestamp': 1694865300.0,
            'datetime': '2023-09-16T10:35:00',
            'message_type': 'message'
        },
        {
            'user_id': 'U003',
            'user_name': 'Jennifer Kim',
            'user_display_name': 'Jen',
            'content': 'Found a bug in Build - when I create a field and use selection mode, the field name suggested contains letters that are not allowed.',
            'timestamp': 1694865600.0,
            'datetime': '2023-09-16T10:40:00',
            'message_type': 'message'
        },
        {
            'user_id': 'U004',
            'user_name': 'Alex Thompson',
            'user_display_name': 'Alex',
            'content': 'Getting streaming error for this chatbot when I access it as a shared user. Error: Unable to receive response while streaming.',
            'timestamp': 1694865900.0,
            'datetime': '2023-09-16T10:45:00',
            'message_type': 'message'
        },
        {
            'user_id': 'U005',
            'user_name': 'Lisa Wang',
            'user_display_name': 'Lisa',
            'content': 'The AIHub feature flags should be enabled at an org level. Being able to turn off location tracking would be useful for demos.',
            'timestamp': 1694866200.0,
            'datetime': '2023-09-16T10:50:00',
            'message_type': 'message'
        },
        {
            'user_id': 'U006',
            'user_name': 'David Martinez',
            'user_display_name': 'David',
            'content': 'All Excel/CSV files are not shown properly in Build projects after the latest deployment. Multiple people marked this as working in UAT though.',
            'timestamp': 1694866500.0,
            'datetime': '2023-09-16T10:55:00',
            'message_type': 'message'
        },
        {
            'user_id': 'U007',
            'user_name': 'Rachel Green',
            'user_display_name': 'Rachel',
            'content': 'URL scraping should crawl sitemap or follow links within website. This is a Q3 roadmap candidate - we are behind industry standard on this feature.',
            'timestamp': 1694866800.0,
            'datetime': '2023-09-16T11:00:00',
            'message_type': 'message'
        }
    ]


def test_pinecone_connection_and_indexing():
    """Test Pinecone API connection and basic indexing/querying"""
    print("🧪 Testing Pinecone Connection and Indexing")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set")
        print("Get your API key from: https://app.pinecone.io/")
        
        api_key = input("\nEnter your Pinecone API key to test: ").strip()
        if not api_key:
            print("❌ No API key provided")
            return False
    
    try:
        # Step 1: Connect to Pinecone
        print("🔌 Step 1: Connecting to Pinecone...")
        vector_store = PineconeVectorStore(
            api_key=api_key, 
            index_name="ai-persona-test",  # Use test index
            enable_logging=True
        )
        print("✅ Successfully connected to Pinecone!")
        
        # Step 2: Check initial stats
        print("\n📊 Step 2: Checking initial index stats...")
        stats = vector_store.get_index_stats()
        if stats:
            initial_count = stats.get('total_vector_count', 0)
            print(f"  - Initial vectors: {initial_count:,}")
            print(f"  - Dimension: {stats.get('dimension', 'Unknown')}")
        else:
            print("  - Index is empty (ready for data)")
            initial_count = 0
        
        # Step 3: Index sample data
        print("\n📝 Step 3: Indexing sample messages...")
        sample_messages = get_sample_messages()
        print(f"  - Indexing {len(sample_messages)} sample messages")
        
        vector_store.build_index(sample_messages, batch_size=10)
        
        # Step 4: Verify indexing with retry
        print("\n🔍 Step 4: Verifying indexing...")
        import time
        
        # Wait a bit for Pinecone to process the vectors
        print("  - Waiting for Pinecone to process vectors...")
        time.sleep(5)
        
        # Try multiple times to get accurate stats
        for attempt in range(3):
            stats = vector_store.get_index_stats()
            if stats:
                new_count = stats.get('total_vector_count', 0)
                print(f"  - Attempt {attempt + 1}: Vectors in index: {new_count:,}")
                if new_count > initial_count:
                    print("✅ Indexing successful!")
                    break
                elif attempt < 2:
                    print("  - Waiting a bit more...")
                    time.sleep(3)
                else:
                    print("⚠️  Vector count still shows 0 - but this might be a Pinecone stats delay")
                    print("  - Proceeding with search test to verify actual functionality...")
        
        # Step 5: Test similarity search (more comprehensive)
        print("\n🔎 Step 5: Testing similarity search...")
        test_queries = [
            "Build project classification issues",
            "Bug report for field validation", 
            "AIHub feedback and user experience",
            "Document upload latency problems"
        ]
        
        search_success = False
        for i, query in enumerate(test_queries, 1):
            print(f"\n  Test Query {i}: '{query}'")
            
            # Try the search multiple times if needed
            results = None
            for search_attempt in range(2):
                results = vector_store.search_similar(query, k=3)
                if results:
                    break
                elif search_attempt == 0:
                    print("    - No results on first try, waiting and retrying...")
                    time.sleep(2)
            
            if results:
                print(f"    ✅ Found {len(results)} similar messages:")
                for j, result in enumerate(results, 1):
                    score = result.get('similarity_score', 0)
                    user = result.get('user_name', 'Unknown')
                    content = result.get('content', '')[:80]
                    print(f"      {j}. [{score:.3f}] {user}: {content}...")
                search_success = True
            else:
                print("    ❌ No results found even after retry")
        
        if not search_success:
            print("\n❌ All search queries failed - there may be an indexing issue")
            
            # Debug: Try to query the index directly
            print("\n🔧 Debug: Checking if any vectors exist...")
            try:
                # Try a very broad search
                debug_results = vector_store.search_similar("test", k=10)
                if debug_results:
                    print(f"  ✅ Debug search found {len(debug_results)} results")
                else:
                    print("  ❌ Debug search also returned no results")
                    
                # Check if the index has the right dimensions
                stats = vector_store.get_index_stats()
                if stats:
                    print(f"  - Index dimension: {stats.get('dimension', 'Unknown')}")
                    print(f"  - Expected dimension: {vector_store.embedding_dim}")
                    
            except Exception as debug_e:
                print(f"  ❌ Debug search failed: {debug_e}")
            
            return False
        
        # Step 6: Test user context search
        print("\n👤 Step 6: Testing user context search...")
        user_context = vector_store.get_user_message_context('U001', limit=5)
        if user_context:
            print(f"    ✅ Found {len(user_context)} messages from Sarah Chen:")
            for j, msg in enumerate(user_context, 1):
                content = msg.get('content', '')[:80]
                print(f"      {j}. {content}...")
        else:
            print("    ❌ No user context found")
            return False
        
        print("\n🎉 All tests passed! Pinecone is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        
        # Common error messages and solutions
        error_str = str(e).lower()
        if "unauthorized" in error_str or "api key" in error_str:
            print("💡 Solution: Check your API key is correct")
        elif "not found" in error_str:
            print("💡 Solution: Index will be created automatically")
        elif "quota" in error_str or "limit" in error_str:
            print("💡 Solution: Check your Pinecone plan limits")
        elif "dimension" in error_str:
            print("💡 Solution: Existing index has different dimensions - delete and recreate")
        else:
            print("💡 Solution: Check your internet connection and Pinecone service status")
        
        return False


def test_with_real_data():
    """Test with real normalized data if available"""
    print("\n🔍 Testing with Real Data")
    print("=" * 30)
    
    normalized_file = "normalized_messages.json"
    if not os.path.exists(normalized_file):
        print("❌ No normalized_messages.json found")
        print("Run the data normalizer first to create this file")
        return False
    
    try:
        print("📂 Loading normalized messages...")
        with open(normalized_file, 'r') as f:
            messages = json.load(f)
        
        print(f"✅ Found {len(messages):,} messages")
        
        # Get API key
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            print("❌ PINECONE_API_KEY not set")
            return False
        
        # Test with first 100 messages for speed
        test_messages = messages[:100] if len(messages) > 100 else messages
        print(f"🧪 Testing with {len(test_messages)} messages...")
        
        # Create vector store
        vector_store = PineconeVectorStore(
            api_key=api_key,
            index_name="ai-persona-real-test",
            enable_logging=True
        )
        
        # Index the data
        vector_store.build_index(test_messages)
        
        # Test search with AIHub-relevant queries
        test_queries = [
            "Build project classification problems",
            "document upload latency issues",
            "field validation errors"
        ]
        
        for query in test_queries:
            print(f"\n  Testing query: '{query}'")
            results = vector_store.search_similar(query, k=3)
            
            if results:
                print(f"    ✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    score = result.get('similarity_score', 0)
                    user = result.get('user_name', 'Unknown')
                    content = result.get('content', '')[:80]
                    print(f"      {i}. [{score:.3f}] {user}: {content}...")
            else:
                print(f"    ❌ No results for '{query}'")
        
        # Test with one specific query for return value
        test_query = "Build project issues"
        results = vector_store.search_similar(test_query, k=5)
        
        if results:
            print(f"✅ Query '{test_query}' found {len(results)} results:")
            for i, result in enumerate(results, 1):
                score = result.get('similarity_score', 0)
                user = result.get('user_name', 'Unknown')
                content = result.get('content', '')[:80]
                print(f"  {i}. [{score:.3f}] {user}: {content}...")
            return True
        else:
            print(f"❌ Query '{test_query}' found no results")
            return False
            
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 AI Persona Bot - Pinecone Testing Suite")
    print("=" * 60)
    
    # Test 1: Connection and basic indexing with sample data
    print("\n🧪 TEST 1: Connection and Sample Data Indexing")
    success1 = test_pinecone_connection_and_indexing()
    
    # Test 2: Real data testing (if available)
    success2 = True  # Default to True if no real data
    if os.path.exists("normalized_messages.json"):
        print("\n🧪 TEST 2: Real Data Testing")
        success2 = test_with_real_data()
    else:
        print("\n⏭️  TEST 2: Skipped (no normalized_messages.json found)")
        print("   Run data normalization first to enable real data testing")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"  Sample Data Test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"  Real Data Test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    overall_success = success1 and success2
    
    if overall_success:
        print("\n🎉 All tests passed! Pinecone is ready to use!")
        print("You can now run: python quickstart.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        print("You can still try the minimal version: python ai_persona_bot_minimal.py")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit(main())
