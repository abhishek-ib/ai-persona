#!/usr/bin/env python3
"""
Test script to verify AI Persona Bot installation

This script tests all components to ensure everything is working correctly.
"""

import os
import sys
import traceback
from pathlib import Path


def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        ('numpy', 'numpy'),
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-cpu'),
        ('google.generativeai', 'google-generativeai'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('dateutil', 'python-dateutil'),
        ('tqdm', 'tqdm')
    ]
    
    failed_imports = []
    
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name} - {e}")
            failed_imports.append(package_name)
    
    if failed_imports:
        print(f"\n❌ Missing packages: {', '.join(failed_imports)}")
        print("Install with: pip install " + " ".join(failed_imports))
        return False
    
    print("✅ All required modules imported successfully")
    return True


def test_project_modules():
    """Test if project modules can be imported"""
    print("\n🧪 Testing project modules...")
    
    project_modules = [
        'data_normalizer',
        'vector_store', 
        'gemini_client',
        'ai_persona_bot'
    ]
    
    failed_modules = []
    
    for module_name in project_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except Exception as e:
            print(f"  ❌ {module_name} - {e}")
            failed_modules.append(module_name)
    
    if failed_modules:
        print(f"\n❌ Failed to import: {', '.join(failed_modules)}")
        return False
    
    print("✅ All project modules imported successfully")
    return True


def test_data_directory():
    """Test data directory structure"""
    print("\n🧪 Testing data directory...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    # Check for users.json
    users_file = data_dir / "users.json"
    if users_file.exists():
        print("  ✅ users.json found")
    else:
        print("  ⚠️  users.json not found")
    
    # Check for message directories
    message_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if message_dirs:
        print(f"  ✅ Found {len(message_dirs)} message directories")
        
        # Check first directory for JSON files
        first_dir = message_dirs[0]
        json_files = list(first_dir.glob("*.json"))
        if json_files:
            print(f"  ✅ Found {len(json_files)} JSON files in {first_dir.name}")
        else:
            print(f"  ⚠️  No JSON files found in {first_dir.name}")
    else:
        print("  ⚠️  No message directories found")
    
    print("✅ Data directory structure checked")
    return True


def test_data_normalization():
    """Test data normalization"""
    print("\n🧪 Testing data normalization...")
    
    try:
        from data_normalizer import DataNormalizer
        
        normalizer = DataNormalizer("data")
        users = normalizer.load_users()
        
        if users:
            print(f"  ✅ Loaded {len(users)} users")
            
            # Test processing a small sample
            sample_messages = normalizer.normalize_all_data()
            if sample_messages:
                print(f"  ✅ Normalized {len(sample_messages)} messages")
            else:
                print("  ⚠️  No messages were normalized")
        else:
            print("  ⚠️  No users loaded")
        
        print("✅ Data normalization test completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Data normalization failed: {e}")
        traceback.print_exc()
        return False


def test_vector_store():
    """Test vector store creation"""
    print("\n🧪 Testing vector store...")
    
    try:
        from vector_store import VectorStore
        
        # Create a small test
        vector_store = VectorStore()
        print("  ✅ Vector store initialized")
        
        # Test embedding creation with sample data
        sample_messages = [
            {
                'user_id': 'test_user',
                'user_name': 'Test User',
                'user_display_name': 'Test',
                'content': 'This is a test message',
                'timestamp': 1694865000.0,
                'datetime': '2023-09-16T10:30:00',
                'message_type': 'message'
            }
        ]
        
        embeddings = vector_store.create_embeddings(sample_messages)
        print(f"  ✅ Created embeddings with shape: {embeddings.shape}")
        
        print("✅ Vector store test completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Vector store test failed: {e}")
        traceback.print_exc()
        return False


def test_gemini_connection():
    """Test Gemini API connection"""
    print("\n🧪 Testing Gemini API connection...")
    
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        print("  ⚠️  GOOGLE_AI_API_KEY not set - skipping Gemini test")
        return True
    
    try:
        from gemini_client import GeminiClient
        
        client = GeminiClient(api_key)
        if client.test_connection():
            print("  ✅ Gemini API connection successful")
            
            # Test simple response
            response = client.generate_simple_response("Hello, this is a test")
            if response:
                print(f"  ✅ Generated test response: {response[:50]}...")
            
            return True
        else:
            print("  ❌ Gemini API connection failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Gemini test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 AI Persona Bot Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_project_modules,
        test_data_directory,
        test_data_normalization,
        test_vector_store,
        test_gemini_connection
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation looks good.")
        print("\nNext steps:")
        print("1. Set GOOGLE_AI_API_KEY environment variable")
        print("2. Run: python ai_persona_bot.py --interactive")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
