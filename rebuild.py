#!/usr/bin/env python3
"""
Rebuild Script for AI Persona Bot

This script forces a complete rebuild of all indexes when you have new data.
"""

import os
import sys
from pathlib import Path


def clean_old_files():
    """Remove old index files"""
    files_to_remove = [
        "normalized_messages.json",
        "message_index.faiss",
        "message_metadata.pkl",
        "message_index_minimal.pkl",
        "message_metadata_minimal.pkl",
        "response_cache.json",
        "data_hash.txt"
    ]
    
    removed = []
    for file_path in files_to_remove:
        if Path(file_path).exists():
            try:
                os.remove(file_path)
                removed.append(file_path)
            except Exception as e:
                print(f"⚠️  Could not remove {file_path}: {e}")
    
    if removed:
        print(f"🗑️  Removed old files: {', '.join(removed)}")
    else:
        print("📁 No old files to remove")


def count_messages():
    """Count total messages in normalized data"""
    try:
        with open("normalized_messages.json", 'r') as f:
            # Count lines instead of loading all data
            line_count = sum(1 for _ in f)
        return line_count - 2  # Subtract opening and closing brackets
    except Exception:
        return 0

def rebuild_indexes():
    """Rebuild indexes from scratch with smart version selection"""
    print("🔄 Rebuilding all indexes from data...")
    
    # Check dataset size
    message_count = count_messages()
    print(f"📊 Dataset contains approximately {message_count:,} messages")
    
    # Smart version selection based on dataset size
    if message_count > 100000:
        print("🧠 Large dataset detected - using minimal version (TF-IDF) for stability")
        try:
            from ai_persona_bot_minimal import AIPersonaBotMinimal
            bot = AIPersonaBotMinimal(rebuild_index=True)
            print("✅ Minimal version rebuilt successfully!")
            return True, "minimal"
        except Exception as e:
            print(f"❌ Minimal version failed: {e}")
            return False, None
    
    elif message_count > 50000:
        print("⚡ Medium dataset - trying chunked neural embeddings...")
        try:
            from vector_store_chunked import build_chunked_vector_store_from_normalized_data
            build_chunked_vector_store_from_normalized_data(max_messages=50000)
            print("✅ Chunked version rebuilt successfully!")
            return True, "chunked"
        except Exception as e:
            print(f"❌ Chunked version failed: {e}")
            print("🔄 Falling back to minimal version...")
            try:
                from ai_persona_bot_minimal import AIPersonaBotMinimal
                bot = AIPersonaBotMinimal(rebuild_index=True)
                print("✅ Minimal version rebuilt successfully!")
                return True, "minimal"
            except Exception as e2:
                print(f"❌ Minimal version also failed: {e2}")
                return False, None
    
    else:
        print("📊 Small dataset - trying full version (neural embeddings)...")
        try:
            from ai_persona_bot import AIPersonaBot
            bot = AIPersonaBot(rebuild_index=True)
            print("✅ Full version rebuilt successfully!")
            return True, "full"
        except Exception as e:
            print(f"❌ Full version failed: {e}")
            print("🔄 Trying minimal version...")
            try:
                from ai_persona_bot_minimal import AIPersonaBotMinimal
                bot = AIPersonaBotMinimal(rebuild_index=True)
                print("✅ Minimal version rebuilt successfully!")
                return True, "minimal"
            except Exception as e2:
                print(f"❌ Both versions failed: {e2}")
                return False, None


def main():
    """Main rebuild function"""
    print("🔄 AI Persona Bot - Force Rebuild")
    print("=" * 40)
    
    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        print("Please place your Slack export data in a 'data' directory")
        return 1
    
    # Count data files
    json_files = list(data_dir.rglob("*.json"))
    print(f"📁 Found {len(json_files)} JSON files in data directory")
    
    if not json_files:
        print("❌ No JSON files found in data directory")
        return 1
    
    # Confirm rebuild
    print("\n⚠️  This will delete all existing indexes and rebuild from scratch.")
    if not input("Continue? (y/n): ").lower().startswith('y'):
        print("❌ Rebuild cancelled")
        return 0
    
    # Clean old files
    clean_old_files()
    
    # Rebuild indexes
    success, version = rebuild_indexes()
    
    if success:
        # Save data hash to track changes
        try:
            import hashlib
            hash_md5 = hashlib.md5()
            for json_file in json_files:
                with open(json_file, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
            
            with open("data_hash.txt", 'w') as f:
                f.write(hash_md5.hexdigest())
            print("📝 Saved data hash for future change detection")
        except Exception as e:
            print(f"⚠️  Could not save data hash: {e}")
        
        print(f"\n🎉 Rebuild complete using {version} version!")
        print("You can now run: python quickstart.py")
        return 0
    else:
        print("\n❌ Rebuild failed!")
        print("Check the error messages above and ensure all dependencies are installed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
