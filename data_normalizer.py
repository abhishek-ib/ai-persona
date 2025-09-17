"""
Data Normalizer for AI Persona Project

This module normalizes Slack message data for training an AI persona.
It extracts and cleans messages, user information, and metadata.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import re


class DataNormalizer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.users = {}
        self.normalized_messages = []
        
    def load_users(self) -> Dict[str, Any]:
        """Load user information from users.json"""
        users_file = os.path.join(self.data_dir, "users.json")
        if os.path.exists(users_file):
            with open(users_file, 'r') as f:
                users_list = json.load(f)
                # Convert to dict for easier lookup
                self.users = {user['id']: user for user in users_list}
        return self.users
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize message text"""
        if not text:
            return ""
        
        # Remove Slack-specific formatting
        text = re.sub(r'<@[A-Z0-9]+>', '', text)  # Remove user mentions
        text = re.sub(r'<#[A-Z0-9]+\|[^>]+>', '', text)  # Remove channel mentions
        text = re.sub(r'<https?://[^>]+\|([^>]+)>', r'\1', text)  # Extract link text
        text = re.sub(r'<https?://[^>]+>', '', text)  # Remove plain links
        text = re.sub(r':[\w+-]+:', '', text)  # Remove emoji codes
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_message_content(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract the main content from a message"""
        content_parts = []
        
        # Main text
        if message.get('text'):
            content_parts.append(self.clean_text(message['text']))
        
        # Attachments text
        if message.get('attachments'):
            for attachment in message['attachments']:
                if attachment.get('text'):
                    content_parts.append(self.clean_text(attachment['text']))
        
        # Thread replies (if any)
        if message.get('replies'):
            for reply in message['replies']:
                if reply.get('text'):
                    content_parts.append(self.clean_text(reply['text']))
        
        content = ' '.join(content_parts).strip()
        return content if content else None
    
    def normalize_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a single message"""
        # Skip bot messages and system messages
        if message.get('subtype') in ['bot_message', 'channel_join', 'channel_leave']:
            return None
            
        user_id = message.get('user')
        if not user_id or user_id not in self.users:
            return None
        
        content = self.extract_message_content(message)
        if not content or len(content.strip()) < 3:  # Skip very short messages
            return None
        
        user_info = self.users[user_id]
        timestamp = float(message.get('ts', 0))
        
        normalized = {
            'user_id': user_id,
            'user_name': user_info.get('real_name', user_info.get('name', 'Unknown')),
            'user_display_name': user_info.get('profile', {}).get('display_name', ''),
            'content': content,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat() if timestamp else None,
            'message_type': message.get('type', 'message'),
            'thread_ts': message.get('thread_ts'),
            'original_message': message  # Keep original for reference
        }
        
        return normalized
    
    def process_channel_files(self, channel_dir: str) -> List[Dict[str, Any]]:
        """Process all message files in a channel directory"""
        messages = []
        
        if not os.path.exists(channel_dir):
            print(f"Channel directory not found: {channel_dir}")
            return messages
        
        # Get all JSON files in the channel directory
        json_files = [f for f in os.listdir(channel_dir) if f.endswith('.json')]
        json_files.sort()  # Process in chronological order
        
        for filename in json_files:
            filepath = os.path.join(channel_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    daily_messages = json.load(f)
                    
                for message in daily_messages:
                    normalized = self.normalize_message(message)
                    if normalized:
                        messages.append(normalized)
                        
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        return messages
    
    def normalize_all_data(self) -> List[Dict[str, Any]]:
        """Normalize all message data"""
        print("Loading users...")
        self.load_users()
        print(f"Loaded {len(self.users)} users")
        
        all_messages = []
        
        # Process each channel directory
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                print(f"Processing channel: {item}")
                channel_messages = self.process_channel_files(item_path)
                all_messages.extend(channel_messages)
                print(f"  Found {len(channel_messages)} messages")
        
        # Sort by timestamp
        all_messages.sort(key=lambda x: x['timestamp'])
        
        print(f"Total normalized messages: {len(all_messages)}")
        return all_messages
    
    def save_normalized_data(self, messages: List[Dict[str, Any]], output_file: str = "normalized_messages.json"):
        """Save normalized messages to file"""
        with open(output_file, 'w') as f:
            json.dump(messages, f, indent=2, default=str)
        print(f"Saved normalized data to {output_file}")
    
    def get_user_messages(self, user_id: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get all messages from a specific user"""
        return [msg for msg in messages if msg['user_id'] == user_id]
    
    def get_user_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about users and their messages"""
        user_stats = {}
        
        for message in messages:
            user_id = message['user_id']
            user_name = message['user_name']
            
            if user_id not in user_stats:
                user_stats[user_id] = {
                    'user_name': user_name,
                    'message_count': 0,
                    'total_characters': 0,
                    'avg_message_length': 0
                }
            
            user_stats[user_id]['message_count'] += 1
            user_stats[user_id]['total_characters'] += len(message['content'])
        
        # Calculate averages
        for user_id, stats in user_stats.items():
            if stats['message_count'] > 0:
                stats['avg_message_length'] = stats['total_characters'] / stats['message_count']
        
        return user_stats


if __name__ == "__main__":
    normalizer = DataNormalizer()
    messages = normalizer.normalize_all_data()
    normalizer.save_normalized_data(messages)
    
    # Print user statistics
    stats = normalizer.get_user_stats(messages)
    print("\nUser Statistics:")
    for user_id, stat in sorted(stats.items(), key=lambda x: x[1]['message_count'], reverse=True):
        print(f"{stat['user_name']}: {stat['message_count']} messages, avg length: {stat['avg_message_length']:.1f} chars")
