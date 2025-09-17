#!/usr/bin/env python3
"""
Quick Start Script for AI Persona Bot

This script provides a simple way to get started with the AI Persona Bot.
"""

import os
import sys
import time
import hashlib
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    pass


def get_data_hash():
    """Calculate hash of data directory to detect changes"""
    data_dir = Path("data")
    if not data_dir.exists():
        return None
    
    hash_md5 = hashlib.md5()
    
    # Hash all JSON files in data directory
    for json_file in data_dir.rglob("*.json"):
        try:
            with open(json_file, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except Exception:
            continue
    
    return hash_md5.hexdigest()

def needs_rebuild():
    """Check if indexes need to be rebuilt due to data changes"""
    # Check if index files exist
    index_files = [
        "normalized_messages.json",
        "message_index.faiss",
        "message_metadata.pkl"
    ]
    
    if not all(Path(f).exists() for f in index_files):
        return True, "Index files missing"
    
    # Check if data has changed
    current_hash = get_data_hash()
    if current_hash is None:
        return True, "Data directory not found"
    
    hash_file = Path("data_hash.txt")
    if hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            if stored_hash != current_hash:
                return True, "Data has changed"
        except Exception:
            return True, "Cannot read data hash"
    else:
        return True, "First time setup"
    
    return False, "Indexes are up to date"

def save_data_hash():
    """Save current data hash"""
    current_hash = get_data_hash()
    if current_hash:
        with open("data_hash.txt", 'w') as f:
            f.write(current_hash)

def rebuild_indexes():
    """Rebuild all indexes from scratch"""
    print("🔄 Rebuilding indexes from new data...")
    
    # Try Threaded Pinecone version first (recommended for best conversation context)
    try:
        from ai_persona_bot_threaded import AIPersonaBotThreaded
        print("🚀 Using Threaded Pinecone version with conversation context...")
        bot = AIPersonaBotThreaded(rebuild_index=True)
        print("✅ Threaded Pinecone version indexes rebuilt successfully!")
        return True, "threaded"
    except Exception as e:
        print(f"Threaded version failed: {e}")
        print("🔄 Trying standard Pinecone version...")
        
        try:
            from ai_persona_bot_pinecone import AIPersonaBotPinecone
            print("🚀 Using standard Pinecone version...")
            bot = AIPersonaBotPinecone(rebuild_index=True)
            print("✅ Pinecone version indexes rebuilt successfully!")
            return True, "pinecone"
        except Exception as e2:
            print(f"Standard Pinecone failed: {e2}")
            print("🔄 Trying minimal version...")
            
            try:
                from ai_persona_bot_minimal import AIPersonaBotMinimal
                print("📊 Using minimal version with TF-IDF...")
                bot = AIPersonaBotMinimal(rebuild_index=True)
                print("✅ Minimal version indexes rebuilt successfully!")
                return True, "minimal"
            except Exception as e3:
                print(f"❌ All versions failed: {e3}")
                return False, None

def main():
    """Quick start guide"""
    print("🚀 AI Persona Bot - Quick Start")
    print("=" * 40)
    
    print("Welcome! Let's get you started with the AI Persona Bot.")
    print("This bot learns from your Slack messages to create AI personas.\n")
    
    # Check if data directory exists
    if not Path("data").exists():
        print("❌ Data directory not found!")
        print("Please place your Slack export data in a 'data' directory")
        return 1
    
    # Check if rebuild is needed
    rebuild_needed, reason = needs_rebuild()
    if rebuild_needed:
        print(f"🔄 Rebuild needed: {reason}")
        
        if input("Rebuild indexes now? (y/n): ").lower().startswith('y'):
            success, version = rebuild_indexes()
            if success:
                save_data_hash()
                print(f"✅ Indexes rebuilt successfully using {version} version!")
            else:
                print("❌ Failed to rebuild indexes")
                return 1
        else:
            print("⚠️  Continuing with existing indexes (may be outdated)")
    else:
        print(f"✅ {reason}")
    
    # Check API keys
    print("\n🔑 API Key Setup:")
    
    # Pinecone API Key
    if not os.getenv('PINECONE_API_KEY'):
        print("You need a Pinecone API key for scalable vector storage.")
        print("Get one at: https://app.pinecone.io/")
        
        pinecone_key = input("\nEnter your Pinecone API key (or press Enter to skip): ").strip()
        if pinecone_key:
            with open(".env", "a") as f:
                f.write(f"\nPINECONE_API_KEY={pinecone_key}\n")
            print("✅ Pinecone API key saved to .env file")
            os.environ['PINECONE_API_KEY'] = pinecone_key
        else:
            print("⚠️  You'll need to set PINECONE_API_KEY later")
    
    # Google AI API Key
    if not os.getenv('GOOGLE_AI_API_KEY'):
        print("\nYou need a Google AI API key for response generation.")
        print("Get one at: https://makersuite.google.com/app/apikey")
        
        gemini_key = input("\nEnter your Google AI API key (or press Enter to skip): ").strip()
        if gemini_key:
            with open(".env", "a") as f:
                f.write(f"\nGOOGLE_AI_API_KEY={gemini_key}\n")
            print("✅ Google AI API key saved to .env file")
            os.environ['GOOGLE_AI_API_KEY'] = gemini_key
        else:
            print("⚠️  You'll need to set GOOGLE_AI_API_KEY later")
    
    # Show available options
    print("\n🎯 What would you like to do?")
    print("1. Start interactive chat")
    print("2. Run example demonstration")
    print("3. Test installation")
    print("4. Show available users")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🤖 Starting interactive chat...")
            print("Use 'help' for commands, 'users' to see personas, 'quit' to exit")
            
            # Try Threaded Pinecone version first (recommended for best conversation context)
            try:
                from ai_persona_bot_threaded import AIPersonaBotThreaded
                print("🚀 Using Threaded Pinecone version for enhanced conversation context...")
                bot = AIPersonaBotThreaded()
                bot.initialize_gemini()
                bot.interactive_chat()
            except Exception as e:
                print(f"Threaded version failed: {e}")
                print("🔄 Trying standard Pinecone version...")
                try:
                    from ai_persona_bot_pinecone import AIPersonaBotPinecone
                    print("🚀 Using standard Pinecone version...")
                    bot = AIPersonaBotPinecone()
                    bot.initialize_gemini()
                    bot.interactive_chat()
                except Exception as e2:
                    print(f"Standard Pinecone failed: {e2}")
                    print("🔄 Trying minimal version (TF-IDF, local storage)...")
                    try:
                        from ai_persona_bot_minimal import AIPersonaBotMinimal
                        bot = AIPersonaBotMinimal()
                        bot.initialize_gemini()
                        bot.interactive_chat()
                    except Exception as e3:
                        print(f"All versions failed: {e3}")
            break
            
        elif choice == "2":
            print("\n🎭 Running example demonstration...")
            try:
                import example
                example.main()
            except Exception as e:
                print(f"Error running example: {e}")
            break
            
        elif choice == "3":
            print("\n🧪 Testing installation...")
            try:
                import test_installation
                test_installation.main()
            except Exception as e:
                print(f"Error running tests: {e}")
            break
            
        elif choice == "4":
            print("\n👥 Loading available users...")
            try:
                from ai_persona_bot import AIPersonaBot
                bot = AIPersonaBot()
                users = bot.list_available_users()
                
                if users:
                    print(f"Found {len(users)} users with message history:")
                    for i, user in enumerate(users, 1):
                        print(f"{i:2}. {user['name']:20} ({user['message_count']:3} messages)")
                else:
                    print("No users found with sufficient message history")
            except Exception as e:
                print(f"Error loading users: {e}")
            break
            
        elif choice == "5":
            print("👋 Goodbye!")
            break
            
        else:
            print("Please enter a number between 1-5")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
