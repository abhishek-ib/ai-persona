#!/usr/bin/env python3
"""
Setup script for AI Persona Bot

This script helps users set up the AI Persona Bot environment and dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def check_data_directory():
    """Check if data directory exists and has content"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        print("   Please place your Slack export data in a 'data' directory")
        return False
    
    # Check for users.json
    if not (data_dir / "users.json").exists():
        print("⚠️  users.json not found in data directory")
        print("   This file is needed for user information")
        return False
    
    # Check for message directories
    message_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not message_dirs:
        print("⚠️  No message directories found in data directory")
        print("   Expected directories with JSON message files")
        return False
    
    print(f"✅ Data directory found with {len(message_dirs)} message directories")
    return True


def check_api_key():
    """Check if Google AI API key is configured"""
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        print("⚠️  GOOGLE_AI_API_KEY environment variable not set")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        print("   Then set it with: export GOOGLE_AI_API_KEY='your-key-here'")
        return False
    
    print("✅ Google AI API key configured")
    return True


def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    api_key = input("Enter your Google AI API key (or press Enter to skip): ").strip()
    if api_key:
        with open(".env", "w") as f:
            f.write(f"GOOGLE_AI_API_KEY={api_key}\n")
        print("✅ .env file created")
    else:
        print("⚠️  Skipped .env file creation")


def run_initial_setup():
    """Run initial data processing"""
    print("🔄 Running initial data processing...")
    try:
        # Import and run data normalization
        from data_normalizer import DataNormalizer
        
        normalizer = DataNormalizer()
        messages = normalizer.normalize_all_data()
        normalizer.save_normalized_data(messages)
        
        print("✅ Data normalization completed")
        return True
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False


def build_vector_index():
    """Build the vector index"""
    print("🔍 Building vector index...")
    try:
        from vector_store import build_vector_store_from_normalized_data
        build_vector_store_from_normalized_data()
        print("✅ Vector index built successfully")
        return True
    except Exception as e:
        print(f"❌ Vector index building failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🤖 AI Persona Bot Setup")
    print("=" * 40)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install requirements
    if success and not install_requirements():
        success = False
    
    # Check data directory
    if success and not check_data_directory():
        success = False
    
    # Create .env file
    if success:
        create_env_file()
    
    # Check API key
    if success:
        check_api_key()
    
    # Run initial setup
    if success and input("\nRun initial data processing? (y/n): ").lower().startswith('y'):
        if run_initial_setup():
            build_vector_index()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Setup completed!")
        print("\nNext steps:")
        print("1. Set your GOOGLE_AI_API_KEY environment variable")
        print("2. Run: python ai_persona_bot.py --interactive")
        print("3. Try: persona <username> then chat!")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
