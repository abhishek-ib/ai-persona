#!/usr/bin/env python3
"""
JSON-Based Data Normalizer for AI Persona Project

This normalizer creates individual JSON files for each conversation block,
providing clean, complete context for Gemini without fragmentation.

Key improvements:
1. Individual JSON files per conversation block
2. Simple sender/message format
3. Complete context preservation
4. Clean separation between indexing and retrieval
"""

import os
import json
import shutil
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class JSONDataNormalizer:
    def __init__(self, data_dir: str = "data", output_dir: str = "generated"):
        """
        Initialize JSON-based data normalizer
        
        Args:
            data_dir: Directory containing Slack dump folders
            output_dir: Directory to store individual JSON files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.users = {}
        self.conversation_index = []  # Track all conversation files
        
        # Ensure output directory exists and is clean
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Setup clean output directory"""
        if os.path.exists(self.output_dir):
            print(f"🗑️  Clearing existing {self.output_dir}/ directory...")
            shutil.rmtree(self.output_dir)
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Created clean {self.output_dir}/ directory")
    
    def discover_dump_folders(self) -> List[str]:
        """Discover all Slack dump folders"""
        dump_folders = []
        
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            
            if not os.path.isdir(item_path) or item.startswith('.'):
                continue
            
            # Check if this looks like a Slack dump folder
            users_file = os.path.join(item_path, 'users.json')
            if os.path.exists(users_file):
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
    
    def get_user_info(self, user_id: str) -> Dict[str, str]:
        """Get clean user information"""
        if user_id in self.users:
            user = self.users[user_id]
            profile = user.get('profile', {})
            return {
                'id': user_id,
                'name': user.get('name', user_id),
                'display_name': profile.get('display_name', ''),
                'real_name': profile.get('real_name', ''),
                'clean_name': (
                    profile.get('display_name') or 
                    profile.get('real_name') or 
                    user.get('name', user_id)
                )
            }
        return {
            'id': user_id,
            'name': user_id,
            'display_name': '',
            'real_name': '',
            'clean_name': user_id
        }
    
    def clean_message_text(self, text: str) -> str:
        """Clean message text"""
        if not text:
            return ""
        
        # Remove Slack formatting but keep readability
        text = text.replace('&gt;', '>')
        text = text.replace('&lt;', '<')
        text = text.replace('&amp;', '&')
        
        # Clean up user mentions - convert <@U123> to readable names
        import re
        def replace_user_mention(match):
            user_id = match.group(1)
            user_info = self.get_user_info(user_id)
            return f"@{user_info['clean_name']}"
        
        text = re.sub(r'<@([A-Z0-9]+)>', replace_user_mention, text)
        
        # Clean up channel mentions
        text = re.sub(r'<#[A-Z0-9]+\|([^>]+)>', r'#\1', text)
        
        # Clean up links
        text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
        text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
        
        return text.strip()
    
    def process_dm_conversations(self, dump_folder: str) -> List[Dict[str, Any]]:
        """Process DM conversations from a dump folder"""
        dms_file = os.path.join(dump_folder, 'dms.json')
        conversations = []
        
        if not os.path.exists(dms_file):
            return conversations
        
        try:
            with open(dms_file, 'r', encoding='utf-8') as f:
                dms_data = json.load(f)
        except Exception as e:
            print(f"   Error loading {dms_file}: {e}")
            return conversations
        
        dump_name = os.path.basename(dump_folder)
        print(f"  📱 Processing DMs from {dump_name}")
        
        for dm in dms_data:
            dm_id = dm.get('id')
            if not dm_id:
                continue
            
            dm_folder = os.path.join(dump_folder, dm_id)
            if not os.path.exists(dm_folder):
                continue
            
            # Process all messages in this DM
            dm_conversations = self._process_dm_folder(dm_folder, dm_id, dump_name)
            conversations.extend(dm_conversations)
        
        return conversations
    
    def _process_dm_folder(self, dm_folder: str, dm_id: str, dump_name: str) -> List[Dict[str, Any]]:
        """Process messages from a single DM folder"""
        conversations = []
        
        # Get all JSON files in the DM folder
        json_files = [f for f in os.listdir(dm_folder) if f.endswith('.json')]
        
        for json_file in sorted(json_files):
            file_path = os.path.join(dm_folder, json_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")
                continue
            
            # Group messages into conversations
            grouped_conversations = self._group_dm_messages(messages, dm_id, json_file)
            conversations.extend(grouped_conversations)
        
        return conversations
    
    def _group_dm_messages(self, messages: List[Dict], dm_id: str, date_file: str) -> List[Dict[str, Any]]:
        """Group DM messages into conversation blocks"""
        conversations = []
        current_conversation = []
        conversation_window = 1800  # 30 minutes
        
        # Sort messages by timestamp
        valid_messages = [
            msg for msg in messages 
            if isinstance(msg, dict) and msg.get('type') == 'message' 
            and msg.get('user') and msg.get('text')
        ]
        
        valid_messages.sort(key=lambda x: float(x.get('ts', 0)))
        
        for message in valid_messages:
            message_time = float(message.get('ts', 0))
            
            # Check if this starts a new conversation
            if current_conversation:
                last_time = float(current_conversation[-1].get('ts', 0))
                if message_time - last_time > conversation_window:
                    # Save current conversation
                    if len(current_conversation) >= 1:
                        conv_data = self._create_dm_conversation_json(
                            current_conversation, dm_id, date_file
                        )
                        conversations.append(conv_data)
                    current_conversation = []
            
            current_conversation.append(message)
        
        # Handle last conversation
        if current_conversation and len(current_conversation) >= 1:
            conv_data = self._create_dm_conversation_json(
                current_conversation, dm_id, date_file
            )
            conversations.append(conv_data)
        
        return conversations
    
    def _create_dm_conversation_json(self, messages: List[Dict], dm_id: str, date_file: str) -> Dict[str, Any]:
        """Create a JSON file for a DM conversation"""
        # Generate unique conversation ID
        first_ts = messages[0].get('ts', '0')
        conversation_id = f"dm_{dm_id}_{date_file.replace('.json', '')}_{first_ts}"
        
        # Build clean conversation data
        conversation_data = {
            'id': conversation_id,
            'type': 'dm',
            'participants': [],
            'messages': [],
            'metadata': {
                'dm_id': dm_id,
                'date_file': date_file,
                'message_count': len(messages),
                'start_time': messages[0].get('ts'),
                'end_time': messages[-1].get('ts')
            }
        }
        
        participants = set()
        
        for message in messages:
            user_id = message.get('user')
            user_info = self.get_user_info(user_id)
            participants.add(user_info['clean_name'])
            
            # Clean message content
            text = self.clean_message_text(message.get('text', ''))
            
            # Add timestamp for readability
            timestamp = datetime.fromtimestamp(float(message.get('ts', 0)))
            time_str = timestamp.strftime('%H:%M')
            
            conversation_data['messages'].append({
                'sender': user_info['clean_name'],
                'message': text,
                'time': time_str,
                'timestamp': message.get('ts')
            })
        
        conversation_data['participants'] = list(participants)
        
        # Save to JSON file
        json_filename = f"{conversation_id}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        # Create index entry
        index_entry = {
            'id': conversation_id,
            'file': json_filename,
            'type': 'dm',
            'participants': conversation_data['participants'],
            'message_count': len(messages),
            'summary': f"DM conversation between {', '.join(conversation_data['participants'])}",
            'first_message': conversation_data['messages'][0]['message'] if conversation_data['messages'] else ""
        }
        
        return index_entry
    
    def process_channel_conversations(self, dump_folder: str) -> List[Dict[str, Any]]:
        """Process channel conversations from a dump folder"""
        channels_file = os.path.join(dump_folder, 'channels.json')
        conversations = []
        
        if not os.path.exists(channels_file):
            return conversations
        
        try:
            with open(channels_file, 'r', encoding='utf-8') as f:
                channels_data = json.load(f)
        except Exception as e:
            print(f"   Error loading {channels_file}: {e}")
            return conversations
        
        dump_name = os.path.basename(dump_folder)
        print(f"  💬 Processing channels from {dump_name}")
        
        for channel in channels_data:
            channel_id = channel.get('id')
            channel_name = channel.get('name', channel_id)
            
            if not channel_id:
                continue
            
            # Try both channel ID and channel name as folder names
            channel_folder = None
            for folder_name in [channel_id, channel_name, channel.get('name_normalized', '')]:
                if folder_name:
                    test_folder = os.path.join(dump_folder, folder_name)
                    if os.path.exists(test_folder):
                        channel_folder = test_folder
                        break
            
            if not channel_folder:
                continue
            
            print(f"    📁 Processing #{channel_name}")
            
            # Process all messages in this channel
            channel_conversations = self._process_channel_folder(
                channel_folder, channel_id, channel_name, dump_name
            )
            conversations.extend(channel_conversations)
        
        return conversations
    
    def _process_channel_folder(self, channel_folder: str, channel_id: str, 
                              channel_name: str, dump_name: str) -> List[Dict[str, Any]]:
        """Process messages from a single channel folder"""
        conversations = []
        
        # Get all JSON files in the channel folder
        json_files = [f for f in os.listdir(channel_folder) if f.endswith('.json')]
        
        for json_file in sorted(json_files):
            file_path = os.path.join(channel_folder, json_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    messages = json.load(f)
            except Exception as e:
                print(f"      Error loading {file_path}: {e}")
                continue
            
            # Group messages into conversation blocks
            grouped_conversations = self._group_channel_messages(
                messages, channel_id, channel_name, json_file
            )
            conversations.extend(grouped_conversations)
        
        return conversations
    
    def _group_channel_messages(self, messages: List[Dict], channel_id: str, 
                               channel_name: str, date_file: str) -> List[Dict[str, Any]]:
        """Group channel messages into conversation blocks (threads + conversational groups)"""
        conversations = []
        
        # Filter valid messages
        valid_messages = [
            msg for msg in messages 
            if isinstance(msg, dict) and msg.get('type') == 'message' 
            and msg.get('user') and not msg.get('subtype') in ['channel_join', 'channel_leave']
        ]
        
        # Group by threads first
        thread_groups = defaultdict(list)
        standalone_messages = []
        
        for message in valid_messages:
            thread_ts = message.get('thread_ts')
            message_ts = message.get('ts')
            
            if thread_ts:
                thread_groups[thread_ts].append(message)
            else:
                # Check if this message starts a thread
                is_thread_starter = any(
                    msg.get('thread_ts') == message_ts 
                    for msg in valid_messages
                )
                
                if is_thread_starter:
                    thread_groups[message_ts].append(message)
                else:
                    standalone_messages.append(message)
        
        # Process explicit threads
        for thread_ts, thread_messages in thread_groups.items():
            thread_messages.sort(key=lambda x: float(x.get('ts', 0)))
            conv_data = self._create_channel_conversation_json(
                thread_messages, channel_id, channel_name, date_file, 'thread'
            )
            conversations.append(conv_data)
        
        # Group standalone messages into conversational blocks
        conversational_groups = self._create_conversational_groups(standalone_messages)
        
        for group_messages in conversational_groups:
            conv_data = self._create_channel_conversation_json(
                group_messages, channel_id, channel_name, date_file, 'conversation'
            )
            conversations.append(conv_data)
        
        return conversations
    
    def _create_conversational_groups(self, messages: List[Dict]) -> List[List[Dict]]:
        """Create conversational groups from standalone messages"""
        if not messages:
            return []
        
        messages.sort(key=lambda x: float(x.get('ts', 0)))
        
        groups = []
        current_group = []
        conversation_window = 300  # 5 minutes
        
        for message in messages:
            message_time = float(message.get('ts', 0))
            
            if current_group:
                last_time = float(current_group[-1].get('ts', 0))
                time_diff = message_time - last_time
                
                # Start new group if too much time has passed
                if time_diff > conversation_window:
                    if len(current_group) >= 2:  # Only save multi-message conversations
                        groups.append(current_group[:])
                    elif len(current_group) == 1:
                        # Single message becomes its own group
                        groups.append(current_group[:])
                    current_group = []
            
            current_group.append(message)
        
        # Handle last group
        if current_group:
            if len(current_group) >= 2:
                groups.append(current_group)
            elif len(current_group) == 1:
                groups.append(current_group)
        
        return groups
    
    def _create_channel_conversation_json(self, messages: List[Dict], channel_id: str, 
                                        channel_name: str, date_file: str, 
                                        conversation_type: str) -> Dict[str, Any]:
        """Create a JSON file for a channel conversation"""
        # Generate unique conversation ID
        first_ts = messages[0].get('ts', '0')
        conversation_id = f"ch_{channel_id}_{date_file.replace('.json', '')}_{first_ts}_{conversation_type}"
        
        # Build clean conversation data
        conversation_data = {
            'id': conversation_id,
            'type': 'channel',
            'channel_name': channel_name,
            'conversation_type': conversation_type,
            'participants': [],
            'messages': [],
            'metadata': {
                'channel_id': channel_id,
                'channel_name': channel_name,
                'date_file': date_file,
                'message_count': len(messages),
                'start_time': messages[0].get('ts'),
                'end_time': messages[-1].get('ts'),
                'is_thread': conversation_type == 'thread'
            }
        }
        
        participants = set()
        
        for i, message in enumerate(messages):
            user_id = message.get('user')
            user_info = self.get_user_info(user_id)
            participants.add(user_info['clean_name'])
            
            # Clean message content
            text = self.clean_message_text(message.get('text', ''))
            
            # Handle file shares
            if message.get('subtype') == 'file_share' and message.get('files'):
                file_info = message['files'][0]
                text = f"[Shared file: {file_info.get('name', 'file')}] {text}"
            
            # Add timestamp for readability
            timestamp = datetime.fromtimestamp(float(message.get('ts', 0)))
            time_str = timestamp.strftime('%H:%M')
            
            # Format message based on conversation type
            if conversation_type == 'thread' and i > 0:
                # Thread reply
                message_entry = {
                    'sender': user_info['clean_name'],
                    'message': text,
                    'time': time_str,
                    'timestamp': message.get('ts'),
                    'is_reply': True
                }
            else:
                # Main message or conversational message
                message_entry = {
                    'sender': user_info['clean_name'],
                    'message': text,
                    'time': time_str,
                    'timestamp': message.get('ts'),
                    'is_reply': False
                }
            
            conversation_data['messages'].append(message_entry)
        
        conversation_data['participants'] = list(participants)
        
        # Save to JSON file
        json_filename = f"{conversation_id}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        # Create index entry
        index_entry = {
            'id': conversation_id,
            'file': json_filename,
            'type': 'channel',
            'channel_name': channel_name,
            'conversation_type': conversation_type,
            'participants': conversation_data['participants'],
            'message_count': len(messages),
            'summary': f"#{channel_name} {conversation_type} with {', '.join(conversation_data['participants'])}",
            'first_message': conversation_data['messages'][0]['message'] if conversation_data['messages'] else ""
        }
        
        return index_entry
    
    def normalize_all_data(self) -> List[Dict[str, Any]]:
        """Normalize all data into individual JSON files"""
        print("🔄 Starting JSON-based data normalization...")
        
        # Discover dump folders
        dump_folders = self.discover_dump_folders()
        
        if not dump_folders:
            print("❌ No Slack dump folders found!")
            return []
        
        print(f"📦 Found {len(dump_folders)} dump folder(s)")
        
        # Load users from all dumps
        print("\n📋 Loading users from all dumps...")
        self.users = {}
        
        for dump_folder in dump_folders:
            self.load_users_from_dump(dump_folder)
        
        print(f"   Loaded {len(self.users)} unique users")
        
        # Process all conversations
        all_conversations = []
        
        for dump_folder in dump_folders:
            print(f"\n🗂️  Processing dump: {os.path.basename(dump_folder)}")
            
            # Process DMs
            dm_conversations = self.process_dm_conversations(dump_folder)
            all_conversations.extend(dm_conversations)
            
            # Process channels
            channel_conversations = self.process_channel_conversations(dump_folder)
            all_conversations.extend(channel_conversations)
        
        # Save conversation index
        self.conversation_index = all_conversations
        index_file = os.path.join(self.output_dir, 'conversation_index.json')
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ JSON normalization complete!")
        print(f"   Created {len(all_conversations)} conversation JSON files")
        print(f"   DM conversations: {sum(1 for c in all_conversations if c['type'] == 'dm')}")
        print(f"   Channel conversations: {sum(1 for c in all_conversations if c['type'] == 'channel')}")
        print(f"   💾 Saved conversation index to {index_file}")
        
        return all_conversations


def main():
    """Main function for testing"""
    normalizer = JSONDataNormalizer()
    conversations = normalizer.normalize_all_data()
    
    print(f"\n📊 Summary:")
    print(f"   Total conversations: {len(conversations)}")
    print(f"   JSON files created in: {normalizer.output_dir}/")


if __name__ == "__main__":
    main()
