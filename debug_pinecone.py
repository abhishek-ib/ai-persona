#!/usr/bin/env python3
"""
Debug Pinecone Issues

This script helps debug why Pinecone indexing/querying might not be working.
"""

import os
import time

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vector_store_pinecone import PineconeVectorStore


def debug_pinecone_issue():
    """Debug the Pinecone indexing issue step by step"""
    print("🔧 Debugging Pinecone Indexing Issue")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY not set")
        return False
    
    try:
        # Step 1: Create vector store with debug index
        print("Step 1: Creating vector store...")
        vector_store = PineconeVectorStore(
            api_key=api_key,
            index_name="debug-test",
            enable_logging=True
        )
        
        # Step 2: Create a single simple message
        print("\nStep 2: Creating simple test message...")
        simple_message = {
            'user_id': 'TEST001',
            'user_name': 'Test User',
            'user_display_name': 'Test',
            'content': 'This is a simple test message for debugging',
            'timestamp': 1694865000.0,
            'datetime': '2023-09-16T10:30:00',
            'message_type': 'message'
        }
        
        # Step 3: Create embedding manually
        print("\nStep 3: Creating embedding manually...")
        embedding = vector_store.model.encode([f"User {simple_message['user_name']}: {simple_message['content']}"])
        print(f"  - Embedding shape: {embedding.shape}")
        print(f"  - Embedding dimension: {vector_store.embedding_dim}")
        
        # Step 4: Create vector ID
        print("\nStep 4: Creating vector ID...")
        vector_id = vector_store._create_message_id(simple_message)
        print(f"  - Vector ID: {vector_id}")
        
        # Step 5: Prepare vector for upsert
        print("\nStep 5: Preparing vector...")
        vector = {
            'id': vector_id,
            'values': embedding[0].tolist(),
            'metadata': {
                'user_id': simple_message['user_id'],
                'user_name': simple_message['user_name'],
                'content': simple_message['content'],
                'timestamp': simple_message['timestamp']
            }
        }
        print(f"  - Vector prepared with {len(vector['values'])} dimensions")
        
        # Step 6: Upsert single vector
        print("\nStep 6: Upserting single vector...")
        result = vector_store.index.upsert(vectors=[vector])
        print(f"  - Upsert result: {result}")
        
        # Step 7: Wait and check stats multiple times
        print("\nStep 7: Checking index stats...")
        for i in range(5):
            time.sleep(2)
            stats = vector_store.get_index_stats()
            count = stats.get('total_vector_count', 0) if stats else 0
            print(f"  - Attempt {i+1}: {count} vectors in index")
            if count > 0:
                break
        
        # Step 8: Try to query
        print("\nStep 8: Testing query...")
        time.sleep(3)  # Extra wait
        
        # Try exact match first
        results = vector_store.search_similar("This is a simple test message", k=5)
        print(f"  - Exact match results: {len(results)}")
        
        # Try broader match
        results2 = vector_store.search_similar("test message", k=5)
        print(f"  - Broad match results: {len(results2)}")
        
        # Try very broad match
        results3 = vector_store.search_similar("debugging", k=5)
        print(f"  - Very broad match results: {len(results3)}")
        
        if results or results2 or results3:
            print("✅ At least one query returned results!")
            for i, result in enumerate((results or results2 or results3), 1):
                score = result.get('similarity_score', 0)
                content = result.get('content', '')
                print(f"    {i}. [{score:.3f}] {content}")
            return True
        else:
            print("❌ No queries returned results")
            
            # Step 9: Debug - list all vectors in index
            print("\nStep 9: Attempting to list vectors...")
            try:
                # Try to fetch by ID
                fetch_result = vector_store.index.fetch(ids=[vector_id])
                print(f"  - Fetch by ID result: {fetch_result}")
                
                if 'vectors' in fetch_result and fetch_result['vectors']:
                    print("  ✅ Vector exists in index!")
                    vector_data = fetch_result['vectors'][vector_id]
                    print(f"  - Metadata: {vector_data.get('metadata', {})}")
                else:
                    print("  ❌ Vector not found by ID")
                    
            except Exception as e:
                print(f"  - Fetch error: {e}")
            
            return False
    
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main debug function"""
    success = debug_pinecone_issue()
    
    if success:
        print("\n🎉 Debug successful - Pinecone is working!")
    else:
        print("\n❌ Debug failed - there's an issue with Pinecone setup")
        print("\nPossible issues:")
        print("1. API key permissions")
        print("2. Index configuration")
        print("3. Network connectivity")
        print("4. Pinecone service delays")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
