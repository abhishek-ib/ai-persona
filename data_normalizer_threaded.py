"""
Enhanced Data Normalizer for AI Persona Project with Thread Support

This module normalizes Slack message data with proper thread handling.
It groups threaded conversations together as contextual blocks.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import defaultdict


class ThreadedDataNormalizer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.users = {}
        self.normalized_messages = []
        self.thread_groups = defaultdict(list)  # thread_ts -> list of messages
        
    def load_users(self) -> Dict[str, Any]:
        """Load user information from users.json"""
        users_file = os.path.join(self.data_dir, "users.json")
        if os.path.exists(users_file):
            with open(users_file, 'r') as f:
                users_list = json.load(f)
                # Convert to dict for easier lookup
                self.users = {user['id']: user for user in users_list}
        return self.users
    
    def get_user_info(self, user_id: str) -> Dict[str, str]:
        """Get user display name and real name"""
        if user_id in self.users:
            user = self.users[user_id]
            return {
                'name': user.get('name', user_id),
                'display_name': user.get('profile', {}).get('display_name', ''),
                'real_name': user.get('profile', {}).get('real_name', '')
            }
        return {'name': user_id, 'display_name': '', 'real_name': ''}
    
    def clean_message_text(self, text: str) -> str:
        """Clean and normalize message text"""
        if not text:
            return ""
        
        # Remove Slack formatting
        text = re.sub(r'<@[^>]+>', lambda m: self._resolve_user_mention(m.group()), text)
        text = re.sub(r'<#[^>]+\|([^>]+)>', r'#\1', text)  # Channel mentions
        text = re.sub(r'<([^>|]+)>', r'\1', text)  # URLs
        text = re.sub(r':([a-zA-Z0-9_+-]+):', r'[\1]', text)  # Emojis
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _resolve_user_mention(self, mention: str) -> str:
        """Resolve user mention to display name"""
        user_id = mention.strip('<@>')
        user_info = self.get_user_info(user_id)
        display_name = user_info['display_name'] or user_info['real_name'] or user_info['name']
        return f"@{display_name}"
    
    def extract_message_content(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract clean text content from a message"""
        # Try different text fields
        text = message.get('text', '')
        
        # Handle different message types
        if message.get('subtype') == 'bot_message':
            # Bot messages might have attachments
            attachments = message.get('attachments', [])
            if attachments and not text:
                text = attachments[0].get('text', '') or attachments[0].get('fallback', '')
        
        # Handle file uploads
        if message.get('subtype') == 'file_share':
            files = message.get('files', [])
            if files:
                file_info = files[0]
                text = f"[Shared file: {file_info.get('name', 'file')}] {text}"
        
        return self.clean_message_text(text) if text else None
    
    def group_messages_by_thread(self, messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group messages by thread_ts and create conversational threads from sequential messages
        
        This handles:
        1. Explicit Slack threads (messages with thread_ts)
        2. Conversational threads (sequential messages that form Q&A patterns)
        """
        thread_groups = defaultdict(list)
        standalone_messages = []
        
        # First pass: identify all explicit thread relationships
        thread_starters = set()
        threaded_messages = {}  # message_ts -> thread_ts
        
        for message in messages:
            thread_ts = message.get('thread_ts')
            message_ts = message.get('ts')
            
            if thread_ts:
                # This is a reply in a thread
                thread_starters.add(thread_ts)
                threaded_messages[message_ts] = thread_ts
        
        # Second pass: group explicit threads
        for message in messages:
            thread_ts = message.get('thread_ts')
            message_ts = message.get('ts')
            
            if thread_ts:
                # This is a reply - add to the thread group using thread_ts as key
                thread_groups[thread_ts].append(message)
            elif message_ts in thread_starters:
                # This message starts a thread - add it to its own thread group
                thread_groups[message_ts].append(message)
            else:
                # This is a standalone message - will be processed for conversational grouping
                standalone_messages.append(message)
        
        # Third pass: create conversational threads from standalone messages
        conversational_groups = self.create_conversational_threads(standalone_messages)
        
        # Merge conversational groups with explicit thread groups
        for group_id, group_messages in conversational_groups.items():
            thread_groups[group_id] = group_messages
        
        return dict(thread_groups)
    
    def create_conversational_threads(self, standalone_messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create conversational threads from standalone messages based on:
        1. Time proximity (messages within conversation_window)
        2. User interaction patterns (different users responding)
        3. Content similarity and question-answer patterns
        """
        if not standalone_messages:
            return {}
        
        # Sort messages by timestamp
        sorted_messages = sorted(standalone_messages, key=lambda x: float(x.get('ts', 0)))
        
        conversational_groups = {}
        current_group = []
        conversation_window = 300  # 5 minutes in seconds
        min_conversation_length = 2  # Minimum messages to form a conversation
        
        for i, message in enumerate(sorted_messages):
            message_time = float(message.get('ts', 0))
            message_user = message.get('user')
            
            # Check if this message should start a new conversation or continue the current one
            should_start_new = True
            
            if current_group:
                last_message = current_group[-1]
                last_time = float(last_message.get('ts', 0))
                last_user = last_message.get('user')
                time_diff = message_time - last_time
                
                # Continue conversation if:
                # 1. Within time window
                # 2. Different user (indicates interaction)
                # 3. Or same user but short time gap (continuation)
                if (time_diff <= conversation_window and 
                    (message_user != last_user or time_diff <= 60)):  # 1 minute for same user
                    should_start_new = False
            
            if should_start_new and current_group:
                # Save current group if it has enough messages
                if len(current_group) >= min_conversation_length:
                    group_id = f"conv_{current_group[0]['ts']}"
                    conversational_groups[group_id] = current_group[:]
                else:
                    # Add single messages as standalone
                    for msg in current_group:
                        conversational_groups[msg['ts']] = [msg]
                
                current_group = []
            
            current_group.append(message)
        
        # Handle the last group
        if current_group:
            if len(current_group) >= min_conversation_length:
                group_id = f"conv_{current_group[0]['ts']}"
                conversational_groups[group_id] = current_group
            else:
                # Add single messages as standalone
                for msg in current_group:
                    conversational_groups[msg['ts']] = [msg]
        
        return conversational_groups
    
    def _determine_message_type(self, thread_messages: List[Dict[str, Any]], is_explicit_thread: bool) -> str:
        """Determine the type of message block"""
        if len(thread_messages) == 1:
            return 'message'
        elif is_explicit_thread:
            return 'thread'
        else:
            return 'conversation'
    
    def create_threaded_message_block(self, thread_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a single message block from a thread of messages (explicit or conversational)"""
        # Sort messages by timestamp
        thread_messages.sort(key=lambda x: float(x.get('ts', 0)))
        
        # Get the main message (first in thread)
        main_message = thread_messages[0]
        
        # Determine if this is an explicit thread or conversational thread
        is_explicit_thread = any(msg.get('thread_ts') for msg in thread_messages)
        
        # Build the conversation text
        conversation_parts = []
        participants = set()
        
        for i, message in enumerate(thread_messages):
            user_id = message.get('user', 'Unknown')
            user_info = self.get_user_info(user_id)
            user_name = user_info['display_name'] or user_info['real_name'] or user_info['name']
            participants.add(user_name)
            
            content = self.extract_message_content(message)
            if content:
                timestamp = datetime.fromtimestamp(float(message.get('ts', 0)))
                time_str = timestamp.strftime('%H:%M')
                
                if is_explicit_thread:
                    # Explicit Slack thread formatting
                    if i == 0:
                        # Main thread message
                        conversation_parts.append(f"[{time_str}] {user_name}: {content}")
                    else:
                        # Reply in thread
                        conversation_parts.append(f"  └─ [{time_str}] {user_name}: {content}")
                else:
                    # Conversational thread formatting (all messages at same level)
                    conversation_parts.append(f"[{time_str}] {user_name}: {content}")
        
        # Create the combined message
        full_conversation = "\n".join(conversation_parts)
        
        # Use the main message's metadata
        main_user_id = main_message.get('user', 'Unknown')
        main_user_info = self.get_user_info(main_user_id)
        
        return {
            'user_id': main_user_id,
            'user_name': main_user_info['display_name'] or main_user_info['real_name'] or main_user_info['name'],
            'user_display_name': main_user_info['display_name'],
            'content': full_conversation,
            'timestamp': float(main_message.get('ts', 0)),
            'datetime': datetime.fromtimestamp(float(main_message.get('ts', 0))).isoformat(),
            'message_type': self._determine_message_type(thread_messages, is_explicit_thread),
            'thread_length': len(thread_messages),
            'participants': list(participants),
            'thread_ts': main_message.get('ts'),
            'raw_content': main_message.get('text', ''),  # Keep original for reference
        }
    
    def normalize_channel_messages(self, channel_file: str) -> List[Dict[str, Any]]:
        """Normalize messages from a single channel file with thread support"""
        print(f"Processing {channel_file}...")
        
        try:
            with open(channel_file, 'r', encoding='utf-8') as f:
                messages = json.load(f)
        except Exception as e:
            print(f"Error loading {channel_file}: {e}")
            return []
        
        # Filter valid messages
        valid_messages = []
        for message in messages:
            # Skip if message is not a dict (malformed data)
            if not isinstance(message, dict):
                continue
                
            if (message.get('type') == 'message' and 
                message.get('user') and 
                not message.get('subtype') in ['channel_join', 'channel_leave']):
                valid_messages.append(message)
        
        print(f"  Found {len(valid_messages)} valid messages")
        
        # Group messages by thread
        thread_groups = self.group_messages_by_thread(valid_messages)
        print(f"  Organized into {len(thread_groups)} conversation blocks")
        
        # Create normalized message blocks
        normalized = []
        for thread_ts, thread_messages in thread_groups.items():
            try:
                message_block = self.create_threaded_message_block(thread_messages)
                if message_block['content']:  # Only include if there's actual content
                    normalized.append(message_block)
            except Exception as e:
                print(f"  Error processing thread {thread_ts}: {e}")
                continue
        
        print(f"  Created {len(normalized)} normalized message blocks")
        return normalized
    
    def discover_dump_folders(self) -> List[str]:
        """Discover all Slack dump folders in the data directory"""
        dump_folders = []
        
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            
            # Skip non-directories and system files
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue
            
            # Check if this looks like a Slack dump folder (has users.json and channels.json)
            users_file = os.path.join(item_path, 'users.json')
            channels_file = os.path.join(item_path, 'channels.json')
            
            if os.path.exists(users_file) and os.path.exists(channels_file):
                dump_folders.append(item_path)
                print(f"📦 Found Slack dump: {item}")
        
        return dump_folders
    
    def load_users_from_dump(self, dump_folder: str):
        """Load users from a specific dump folder"""
        users_file = os.path.join(dump_folder, 'users.json')
        
        try:
            with open(users_file, 'r', encoding='utf-8') as f:
                dump_users = json.load(f)
            
            # Merge users (avoid duplicates by user ID)
            for user in dump_users:
                user_id = user.get('id')
                if user_id and user_id not in self.users:
                    self.users[user_id] = user
                    
        except Exception as e:
            print(f"   Warning: Could not load users from {users_file}: {e}")
    
    def process_dump_folder(self, dump_folder: str) -> List[Dict[str, Any]]:
        """Process all messages from a single dump folder"""
        dump_name = os.path.basename(dump_folder)
        print(f"\n🗂️  Processing dump folder: {dump_name}")
        
        dump_messages = []
        
        # Process each channel directory in this dump
        for item in os.listdir(dump_folder):
            item_path = os.path.join(dump_folder, item)
            
            # Skip non-directories, system files, and known non-message directories
            if (not os.path.isdir(item_path) or 
                item.startswith('.') or 
                item in ['attachments'] or
                item.endswith('.json')):
                continue
            
            print(f"  📁 Processing channel directory: {item}")
            
            # Process all JSON files in the directory
            try:
                json_files = [f for f in os.listdir(item_path) if f.endswith('.json')]
                
                for json_file in sorted(json_files):
                    file_path = os.path.join(item_path, json_file)
                    channel_messages = self.normalize_channel_messages(file_path)
                    dump_messages.extend(channel_messages)
                    
            except Exception as e:
                print(f"  ⚠️  Error processing {item}: {e}")
                continue
        
        print(f"  ✅ Processed {len(dump_messages)} message blocks from {dump_name}")
        return dump_messages
    
    def normalize_all_data(self) -> List[Dict[str, Any]]:
        """Normalize all message data with thread support from multiple dump folders"""
        print("🔄 Starting recursive threaded data normalization...")
        
        # Discover all dump folders
        dump_folders = self.discover_dump_folders()
        
        if not dump_folders:
            print("❌ No Slack dump folders found in data directory!")
            print("   Expected folders with users.json and channels.json files")
            return []
        
        print(f"📦 Found {len(dump_folders)} dump folder(s)")
        
        # Load users from all dumps
        print("\n📋 Loading users from all dumps...")
        self.users = {}  # Reset users dict
        
        for dump_folder in dump_folders:
            self.load_users_from_dump(dump_folder)
        
        print(f"   Loaded {len(self.users)} unique users across all dumps")
        
        # Process all dump folders
        all_messages = []
        
        for dump_folder in dump_folders:
            dump_messages = self.process_dump_folder(dump_folder)
            all_messages.extend(dump_messages)
        
        # Sort all messages by timestamp
        all_messages.sort(key=lambda x: x['timestamp'])
        
        # Calculate statistics
        threaded_count = sum(1 for msg in all_messages if msg['message_type'] == 'thread')
        single_count = sum(1 for msg in all_messages if msg['message_type'] == 'message')
        
        print(f"\n✅ Recursive normalization complete!")
        print(f"   Total message blocks: {len(all_messages)}")
        print(f"   Threaded conversations: {threaded_count}")
        print(f"   Single messages: {single_count}")
        print(f"   Processed from {len(dump_folders)} dump folder(s)")
        
        return all_messages
    
    def save_normalized_data(self, messages: List[Dict[str, Any]], filename: str = "normalized_messages_threaded.json"):
        """Save normalized messages to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(messages)} normalized message blocks to {filename}")
    
    def get_statistics(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the normalized data"""
        if not messages:
            return {}
        
        users = {}
        thread_lengths = []
        total_participants = set()
        
        for message in messages:
            user_id = message['user_id']
            if user_id not in users:
                users[user_id] = {
                    'name': message['user_name'],
                    'message_count': 0,
                    'thread_count': 0
                }
            
            users[user_id]['message_count'] += 1
            if message['message_type'] == 'thread':
                users[user_id]['thread_count'] += 1
                thread_lengths.append(message['thread_length'])
            
            total_participants.update(message.get('participants', [message['user_name']]))
        
        return {
            'total_messages': len(messages),
            'total_users': len(users),
            'total_participants': len(total_participants),
            'threaded_conversations': sum(1 for msg in messages if msg['message_type'] == 'thread'),
            'single_messages': sum(1 for msg in messages if msg['message_type'] == 'message'),
            'average_thread_length': sum(thread_lengths) / len(thread_lengths) if thread_lengths else 0,
            'max_thread_length': max(thread_lengths) if thread_lengths else 0,
            'users': dict(sorted(users.items(), key=lambda x: x[1]['message_count'], reverse=True))
        }


def main():
    """Test the threaded data normalizer"""
    normalizer = ThreadedDataNormalizer("data")
    messages = normalizer.normalize_all_data()
    normalizer.save_normalized_data(messages)
    
    # Print statistics
    stats = normalizer.get_statistics(messages)
    print(f"\n📊 Statistics:")
    print(f"   Total message blocks: {stats['total_messages']}")
    print(f"   Threaded conversations: {stats['threaded_conversations']}")
    print(f"   Single messages: {stats['single_messages']}")
    print(f"   Average thread length: {stats['average_thread_length']:.1f}")
    print(f"   Max thread length: {stats['max_thread_length']}")
    print(f"   Top users:")
    for i, (user_id, info) in enumerate(list(stats['users'].items())[:10]):
        print(f"     {i+1}. {info['name']}: {info['message_count']} blocks ({info['thread_count']} threads)")


if __name__ == "__main__":
    main()
