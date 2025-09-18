#!/usr/bin/env python3
"""
Script to parse messages from sherif and weiming data directories
and extract all messages from users named Paul.
"""

import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any

# Paul user IDs found in the data
PAUL_USER_IDS = {
    'UEDF3JGN8',      # paul (Paul Nakata)
    'U011UPRAPNV',    # hoangpaul (Paul Hoang)
    'U0303UC6HDX',    # paul.chicos (Paul Chicos)
    'U03AB9GJ4JZ',    # paul.gilbert (Paul Gilbert)
    'U03T9DVM3QC',    # Paul Gill
}

def load_users(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load users.json and create a mapping of user_id to user info."""
    users_file = os.path.join(data_dir, 'users.json')
    if not os.path.exists(users_file):
        print(f"Warning: {users_file} not found")
        return {}
    
    with open(users_file, 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    return {user['id']: user for user in users}

def contains_url_or_link(text: str) -> bool:
    """Check if message text contains URLs or links."""
    if not text:
        return False
    
    # Patterns to match URLs and links
    url_patterns = [
        r'https?://[^\s]+',  # http/https URLs
        r'www\.[^\s]+',      # www URLs
        r'<https?://[^>]+>', # Slack formatted links
        r'github\.com/[^\s]+', # GitHub links
        r'instabase\.atlassian\.net/[^\s]+', # JIRA links
        r'atlassian\.net/[^\s]+', # Atlassian links
        r'pull/\d+',         # Pull request references
        r'FLOW-\d+',         # JIRA ticket references
        r'[A-Z]+-\d+',       # General ticket references
        r'<[^>]*github[^>]*>', # GitHub links in angle brackets
        r'<[^>]*pull[^>]*>',   # Pull request links in angle brackets
    ]
    
    for pattern in url_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False

def parse_messages_from_file(file_path: str, users: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse messages from a single JSON file and return Paul's messages."""
    paul_messages = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        
        for message in messages:
            if isinstance(message, dict) and message.get('user') in PAUL_USER_IDS:
                text = message.get('text', '')
                
                # Skip messages with URLs or links
                if contains_url_or_link(text):
                    continue
                
                # Get user info
                user_id = message['user']
                user_info = users.get(user_id, {})
                
                # Format the message
                formatted_message = {
                    'timestamp': message.get('ts', ''),
                    'user_id': user_id,
                    'user_name': user_info.get('name', 'unknown'),
                    'real_name': user_info.get('real_name', 'unknown'),
                    'text': text,
                    'file_path': file_path
                }
                
                paul_messages.append(formatted_message)
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return paul_messages

def parse_directory(data_dir: str) -> List[Dict[str, Any]]:
    """Parse all message files in a data directory."""
    print(f"Parsing directory: {data_dir}")
    
    # Load users
    users = load_users(data_dir)
    print(f"Loaded {len(users)} users")
    
    all_paul_messages = []
    
    # Find all subdirectories that contain JSON files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json') and file != 'users.json':
                file_path = os.path.join(root, file)
                messages = parse_messages_from_file(file_path, users)
                all_paul_messages.extend(messages)
                if messages:
                    print(f"Found {len(messages)} Paul messages in {file}")
    
    return all_paul_messages

def format_timestamp(ts: str) -> str:
    """Convert Slack timestamp to readable format."""
    try:
        if '.' in ts:
            ts = ts.split('.')[0]
        timestamp = int(ts)
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ts

def main():
    """Main function to parse messages and create output file."""
    base_dir = '/Users/abhishek/ai-persona/data'
    sherif_dir = os.path.join(base_dir, 'sherif')
    weiming_dir = os.path.join(base_dir, 'weiming')
    
    all_messages = []
    
    # Parse sherif directory
    if os.path.exists(sherif_dir):
        sherif_messages = parse_directory(sherif_dir)
        all_messages.extend(sherif_messages)
        print(f"Total Paul messages from sherif: {len(sherif_messages)}")
    else:
        print(f"Directory not found: {sherif_dir}")
    
    # Parse weiming directory
    if os.path.exists(weiming_dir):
        weiming_messages = parse_directory(weiming_dir)
        all_messages.extend(weiming_messages)
        print(f"Total Paul messages from weiming: {len(weiming_messages)}")
    else:
        print(f"Directory not found: {weiming_dir}")
    
    # Sort messages by timestamp
    all_messages.sort(key=lambda x: x['timestamp'])
    
    # Write to output file
    output_file = '/Users/abhishek/ai-persona/paul_messages.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for msg in all_messages:
            f.write(f"{msg['text']}\n")
    
    print(f"\nOutput written to: {output_file}")
    print(f"Total Paul messages found: {len(all_messages)}")
    
    # Print summary by user
    user_counts = {}
    for msg in all_messages:
        real_name = msg['real_name']
        user_counts[real_name] = user_counts.get(real_name, 0) + 1
    
    print("\nMessages by Paul user:")
    for real_name, count in sorted(user_counts.items()):
        print(f"  {real_name}: {count} messages")

if __name__ == "__main__":
    main()
