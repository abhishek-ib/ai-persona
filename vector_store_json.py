#!/usr/bin/env python3
"""
JSON-Based Vector Store for AI Persona Project

This vector store indexes conversation JSON file IDs and retrieves complete
conversation files for Gemini, providing clean, complete context.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pinecone
from pinecone import Pinecone

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class JSONVectorStore:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "all-MiniLM-L6-v2",
                 index_name: str = "ai-persona-json",
                 environment: str = "us-east-1",
                 json_dir: str = "generated",
                 enable_logging: bool = True):
        """
        Initialize JSON-based vector store
        
        Args:
            api_key: Pinecone API key
            model_name: Sentence transformer model
            index_name: Pinecone index name
            environment: Pinecone environment
            json_dir: Directory containing conversation JSON files
            enable_logging: Enable query logging
        """
        self.api_key = api_key or os.getenv('PINECONE_API_KEY') or ''
        self.model_name = model_name
        self.index_name = index_name
        self.environment = environment
        self.json_dir = json_dir
        self.enable_logging = enable_logging
        
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY environment variable.")
        
        # Setup logging
        if enable_logging:
            self._setup_logging()
        
        # Initialize sentence transformer
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self._setup_index()
        
        print("✅ Connected to Pinecone")
    
    def _setup_logging(self):
        """Setup logging for queries"""
        os.makedirs("logs", exist_ok=True)
        
        # Setup query logger
        self.query_logger = logging.getLogger('json_pinecone_queries')
        self.query_logger.setLevel(logging.INFO)
        
        query_handler = logging.FileHandler('logs/json_pinecone_queries.log')
        query_formatter = logging.Formatter('%(asctime)s - %(message)s')
        query_handler.setFormatter(query_formatter)
        
        # Clear existing handlers
        self.query_logger.handlers.clear()
        self.query_logger.addHandler(query_handler)
    
    def _setup_index(self):
        """Setup or connect to Pinecone index"""
        try:
            # Try to get existing index
            index_info = self.pc.describe_index(self.index_name)
            print(f"✅ Connected to Pinecone index: {self.index_name}")
        except Exception:
            # Create new index
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric='cosine',
                spec={'serverless': {'cloud': 'aws', 'region': self.environment}}
            )
            print(f"✅ Created Pinecone index: {self.index_name}")
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            time.sleep(10)
        
        self.index = self.pc.Index(self.index_name)
    
    def load_conversation_index(self) -> List[Dict[str, Any]]:
        """Load conversation index from JSON"""
        index_file = os.path.join(self.json_dir, 'conversation_index.json')
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Conversation index not found: {index_file}")
        
        with open(index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_error_messages(self, text: str) -> List[str]:
        """Extract key error messages from technical content"""
        import re
        
        errors = []
        
        # Common error patterns (case insensitive)
        patterns = [
            r'ERROR:.*?(?=\n|$)',
            r'FATAL:.*?(?=\n|$)',
            r'Exception:.*?(?=\n|$)',
            r'Error:.*?(?=\n|$)',
            r'\[ERROR\].*?(?=\n|$)',
            r'failed with exit status \d+.*?(?=\n|$)',
            r'provided hosts list is empty.*?(?=\n|$)',
            r'No inventory was parsed.*?(?=\n|$)',
            r'.*not found.*?(?=\n|$)',
            r'.*failed.*?(?=\n|$)',
            r'.*error.*?(?=\n|$)',
            r'Traceback.*?(?=\n|$)',
            r'.*Exception.*?(?=\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            # Take first 1000 chars of each error message
            errors.extend([match[:1000] for match in matches[:2]])  # Max 2 matches per pattern
        
        # Remove duplicates while preserving order
        seen = set()
        unique_errors = []
        for error in errors:
            error_key = error.lower()[:100]  # Use first 100 chars as key for dedup
            if error_key not in seen:
                seen.add(error_key)
                unique_errors.append(error)
        
        return unique_errors[:5]  # Max 5 error messages
    
    def _extract_tech_keywords(self, text: str) -> List[str]:
        """Extract technical keywords from content"""
        import re
        
        keywords = []
        
        # Technical keyword patterns
        patterns = [
            r'\b(ansible|docker|kubernetes|k8s|jenkins|gitlab|github)\b',
            r'\b(python|java|javascript|node|npm|pip|maven|gradle)\b',
            r'\b(mysql|postgres|mongodb|redis|elasticsearch)\b',
            r'\b(aws|azure|gcp|s3|ec2|lambda|api)\b',
            r'\b(localhost|127\.0\.0\.1|deployment|build|test|prod)\b',
            r'\b(error|failed|timeout|connection|network|permission)\b',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        # Remove duplicates and return unique keywords
        return list(set([kw.lower() for kw in keywords]))[:10]  # Max 10 keywords
    
    def _create_enhanced_search_text(self, conv: Dict[str, Any]) -> str:
        """Create enhanced search text focusing on first message and error extraction"""
        first_message = conv.get('first_message', '')
        
        # Start with the complete first message (this is the primary content)
        search_parts = [first_message]
        
        # Extract and add error messages (first 1000 chars each)
        error_messages = self._extract_error_messages(first_message)
        search_parts.extend(error_messages)
        
        # Extract and add technical keywords
        tech_keywords = self._extract_tech_keywords(first_message)
        if tech_keywords:
            search_parts.append(' '.join(tech_keywords))
        
        # Join all parts with separator to maintain context
        search_text = ' | '.join(filter(None, search_parts))
        
        return search_text.strip()
    
    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Search for exact keyword matches in conversations"""
        conversations = self.load_conversation_index()
        matches = []
        
        query_lower = query.lower().strip()
        
        for conv in conversations:
            first_message = conv.get('first_message', '').lower()
            
            # Check for exact substring match
            if query_lower in first_message:
                # Load full conversation JSON
                json_path = os.path.join(self.json_dir, conv['file'])
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        raw_json_content = f.read()
                    
                    # Create enhanced result for keyword match
                    enhanced_result = self._create_enhanced_search_result(
                        None, conv['file'], 1.0, raw_json_content, json_path, 'exact_keyword'
                    )
                    enhanced_result['conversation_id'] = conv['id']
                    matches.append(enhanced_result)
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue
        
        return matches
    
    def _create_enhanced_search_result(self, match, file_name: str, similarity_score: float, 
                                     raw_json_content: str, json_path: str, match_type: str) -> Dict[str, Any]:
        """Create enhanced search result with detailed message information"""
        try:
            # Parse the JSON content to get message details
            conv_data = json.loads(raw_json_content)
            
            # Get first message details
            first_message_details = None
            if conv_data.get('messages'):
                first_msg = conv_data['messages'][0]
                
                # Extract timestamp from the message or use conversation ID
                timestamp = first_msg.get('timestamp')
                if timestamp:
                    # Convert string timestamp to float if needed
                    try:
                        timestamp = float(timestamp)
                    except (ValueError, TypeError):
                        timestamp = None
                
                # Create the detailed message format
                first_message_details = {
                    'id': conv_data.get('id', ''),  # Conversation ID as UUID
                    'text': first_msg.get('message', ''),
                    'user': first_msg.get('user_id', ''),
                    'timestamp': timestamp,
                    'type': conv_data.get('conversation_type', 'single'),
                    'channel_name': conv_data.get('channel_name', '')
                }
            
            # Enhanced search result format
            result = {
                'file_name': file_name,
                'similarity_score': similarity_score,
                'raw_json_content': raw_json_content,
                'json_path': json_path,
                'match_type': match_type,
                'conversation_id': conv_data.get('id', ''),
                
                # Detailed message information
                'message_details': first_message_details,
                
                # Additional metadata from Pinecone (if available)
                'metadata': getattr(match, 'metadata', {}) if hasattr(match, 'metadata') else {}
            }
            
            return result
            
        except Exception as e:
            print(f"Error creating enhanced search result for {file_name}: {e}")
            # Fallback to basic format
            return {
                'file_name': file_name,
                'similarity_score': similarity_score,
                'raw_json_content': raw_json_content,
                'json_path': json_path,
                'match_type': match_type,
                'message_details': None,
                'metadata': {}
            }
    
    def build_index(self, batch_size: int = 100):
        """Build Pinecone index from conversation JSON files"""
        print("🔍 Building JSON-based Pinecone index...")
        
        # Load conversation index
        conversations = self.load_conversation_index()
        print(f"Found {len(conversations)} conversations to index")
        
        # Clear existing index
        try:
            self.index.delete(delete_all=True)
            time.sleep(3)  # Wait for deletion
            print("🗑️  Cleared existing vectors")
        except Exception as e:
            print(f"Note: Could not clear existing vectors: {e}")
            print("Proceeding with indexing...")
        
        vectors_to_upsert = []
        
        for i in tqdm(range(0, len(conversations), batch_size), desc="Creating embeddings"):
            batch = conversations[i:i + batch_size]
            
            # Create embeddings for this batch
            texts = []
            for conv in batch:
                # Create enhanced search text with error extraction and keywords
                search_text = self._create_enhanced_search_text(conv)
                texts.append(search_text)
            
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                
                # Prepare vectors for Pinecone
                for conv, embedding in zip(batch, embeddings):
                    vector_id = conv['id']
                    
                    first_message = conv.get('first_message', '')
                    
                    # Enhanced metadata with technical indicators and user IDs
                    metadata = {
                        'id': conv['id'],  # Add conversation ID
                        'file': conv['file'],
                        'type': conv['type'],
                        'conversation_type': conv.get('conversation_type', 'single'),
                        'participants': conv['participants'][:5],  # Limit for metadata size
                        'message_count': conv['message_count'],
                        'channel_name': conv.get('channel_name', '')[:50] if conv.get('channel_name') else '',
                        
                        # User ID information
                        'participant_user_ids': conv.get('participant_user_ids', [])[:5],  # Original Slack user IDs
                        'first_message_user_id': conv.get('first_message_user_id'),  # User ID of first message sender
                        
                        # First message content (truncated for metadata size limits)
                        'first_message_text': first_message if first_message else '',  # First 1000 chars
                        
                        # Technical indicators for better filtering
                        'has_error': any(keyword in first_message.lower() for keyword in ['error', 'failed', 'exception', 'fatal']),
                        'has_stack_trace': 'traceback' in first_message.lower() or 'stack trace' in first_message.lower(),
                        'message_length': len(first_message),
                        'tech_keywords': ' '.join(self._extract_tech_keywords(first_message)[:5]),  # Top 5 keywords
                        'first_100_chars': first_message[:100].replace('\n', ' ').strip()  # Preview for debugging
                    }
                    
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding.tolist(),
                        'metadata': metadata
                    })
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        # Upsert vectors to Pinecone
        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        
        for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Upserting to Pinecone"):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
                continue
        
        print("✅ Successfully upserted vectors to Pinecone")
        
        # Wait for indexing and verify
        print("⏳ Waiting for Pinecone to process vectors...")
        time.sleep(5)
        
        stats = self.index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        print(f"✅ Built JSON-based Pinecone index with {vector_count} vectors")
        
        return vector_count
    
    def search_similar_conversations(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and keyword matching"""
        print(f"🔍 Hybrid search for: '{query}'")
        
        # 1. Keyword search for exact matches (high priority)
        keyword_matches = self._keyword_search(query)
        print(f"   Found {len(keyword_matches)} exact keyword matches")
        
        # 2. Semantic search via Pinecone
        semantic_matches = self._semantic_search(query, k * 2)  # Get more for deduplication
        print(f"   Found {len(semantic_matches)} semantic matches")
        
        # 3. Merge results and remove duplicates
        merged_results = self._merge_search_results(keyword_matches, semantic_matches, k)
        
        # Log the combined results
        if self.enable_logging:
            conversations_for_logging = []
            for file_info in merged_results:
                try:
                    parsed_data = json.loads(file_info['raw_json_content'])
                    parsed_data['similarity_score'] = file_info['similarity_score']
                    parsed_data['match_type'] = file_info.get('match_type', 'semantic')
                    conversations_for_logging.append(parsed_data)
                except:
                    pass
            self._log_query_and_results(query, conversations_for_logging)
        
        return merged_results
    
    def _semantic_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using Pinecone"""
        # Create embedding for query
        query_embedding = self.model.encode([query])[0]
        
        # Search Pinecone
        try:
            search_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
        
        # Load raw JSON file content
        conversation_files = []
        
        for match in search_results.matches:
            file_name = match.metadata.get('file')
            similarity_score = float(match.score)
            
            if file_name:
                # Load raw JSON content as string
                json_path = os.path.join(self.json_dir, file_name)
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        raw_json_content = f.read()
                    
                    # Create enhanced search result with detailed message info
                    enhanced_result = self._create_enhanced_search_result(
                        match, file_name, similarity_score, raw_json_content, json_path, 'semantic'
                    )
                    conversation_files.append(enhanced_result)
                    
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue
        
        return conversation_files
    
    def _merge_search_results(self, keyword_matches: List[Dict[str, Any]], 
                            semantic_matches: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Merge keyword and semantic search results, avoiding duplicates"""
        # Use conversation ID to track unique results
        seen_ids = set()
        merged_results = []
        
        # 1. Add keyword matches first (highest priority)
        for match in keyword_matches:
            conv_id = match.get('conversation_id')
            if conv_id and conv_id not in seen_ids:
                seen_ids.add(conv_id)
                merged_results.append(match)
                if len(merged_results) >= k:
                    break
        
        # 2. Add semantic matches that aren't already included
        for match in semantic_matches:
            if len(merged_results) >= k:
                break
                
            # Extract conversation ID from file name or JSON content
            try:
                parsed_json = json.loads(match['raw_json_content'])
                conv_id = parsed_json.get('id')
                
                if conv_id and conv_id not in seen_ids:
                    seen_ids.add(conv_id)
                    merged_results.append(match)
            except:
                # If we can't parse JSON, use file name as fallback
                file_name = match.get('file_name', '')
                if file_name not in [m.get('file_name', '') for m in merged_results]:
                    merged_results.append(match)
        
        # Sort by similarity score (keyword matches already have score 1.0)
        merged_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        print(f"   Merged to {len(merged_results)} unique results")
        return merged_results[:k]
    
    def get_user_conversations(self, user_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations for a specific user"""
        # Load conversation index
        conversations = self.load_conversation_index()
        
        # Filter conversations by user participation
        user_conversations = [
            conv for conv in conversations 
            if user_name in conv.get('participants', [])
        ]
        
        # Sort by most recent (assuming ID contains timestamp)
        user_conversations.sort(key=lambda x: x['id'], reverse=True)
        
        # Load complete JSON data for top conversations
        complete_conversations = []
        
        for conv in user_conversations[:limit]:
            json_path = os.path.join(self.json_dir, conv['file'])
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                complete_conversations.append(conversation_data)
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue
        
        return complete_conversations
    
    def _log_query_and_results(self, query: str, conversations: List[Dict[str, Any]]):
        """Log query and complete conversation results"""
        if not self.enable_logging:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'results_count': len(conversations),
            'conversations': []
        }
        
        # Log each conversation with complete data
        for i, conv in enumerate(conversations):
            conv_entry = {
                'rank': i + 1,
                'similarity_score': conv.get('similarity_score', 0),
                'id': conv.get('id'),
                'type': conv.get('type'),
                'conversation_type': conv.get('conversation_type', ''),
                'participants': conv.get('participants', []),
                'message_count': len(conv.get('messages', [])),
                'channel_name': conv.get('channel_name', ''),
                'complete_conversation': conv  # Store complete conversation
            }
            log_entry['conversations'].append(conv_entry)
        
        # Log to file
        self.query_logger.info(f"JSON_QUERY: {json.dumps(log_entry, indent=2)}")
        
        # Console output
        print(f"\n🔍 JSON Pinecone query: '{query[:50]}...'")
        print(f"   Results returned: {len(conversations)}")
        if conversations:
            print("   Top conversations:")
            for i, conv in enumerate(conversations[:3], 1):
                score = conv.get('similarity_score', 0)
                participants = ', '.join(conv.get('participants', [])[:3])
                conv_type = conv.get('conversation_type', 'single')
                channel = conv.get('channel_name', 'DM')
                message_count = len(conv.get('messages', []))
                print(f"     {i}. [{score:.3f}] {channel} - {conv_type} ({message_count} msgs) - {participants}")


def main():
    """Test the JSON vector store"""
    print("🧪 Testing JSON Vector Store")
    
    try:
        store = JSONVectorStore()
        
        # Build index
        vector_count = store.build_index()
        
        if vector_count > 0:
            # Test search
            results = store.search_similar_conversations("language settings chat", k=3)
            print(f"✅ Found {len(results)} results for test query")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
