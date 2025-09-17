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
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
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
                # Create searchable text from conversation summary and complete first message
                summary = conv.get('summary', '')
                first_message = conv.get('first_message', '')
                search_text = f"{summary} {first_message}"
                texts.append(search_text)
            
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                
                # Prepare vectors for Pinecone
                for conv, embedding in zip(batch, embeddings):
                    vector_id = conv['id']
                    
                    # Prepare metadata (store JSON file reference)
                    metadata = {
                        'file': conv['file'],
                        'type': conv['type'],
                        'conversation_type': conv.get('conversation_type', 'single'),
                        'participants': conv['participants'][:5],  # Limit for metadata size
                        'message_count': conv['message_count'],
                        'summary': conv['summary'][:200],  # Limit for metadata size
                        'channel_name': conv.get('channel_name', '')[:50] if conv.get('channel_name') else ''
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
        """Search for similar conversations and return complete JSON data"""
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
        
        # Load complete conversation JSON files
        conversations = []
        
        for match in search_results.matches:
            file_name = match.metadata.get('file')
            similarity_score = float(match.score)
            
            if file_name:
                # Load complete conversation JSON
                json_path = os.path.join(self.json_dir, file_name)
                
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        conversation_data = json.load(f)
                    
                    # Add similarity score
                    conversation_data['similarity_score'] = similarity_score
                    conversations.append(conversation_data)
                    
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue
        
        # Log the query and results
        if self.enable_logging:
            self._log_query_and_results(query, conversations)
        
        return conversations
    
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
