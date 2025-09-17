"""
Pinecone Vector Store Implementation for AI Persona Project

This module creates and manages a scalable vector database using Pinecone
and sentence transformers for similarity search of normalized messages.
"""

import json
import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm
import numpy as np
from datetime import datetime


class PineconeVectorStore:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "all-MiniLM-L6-v2",
                 index_name: str = "ai-persona-messages",
                 environment: str = "us-east-1",
                 enable_logging: bool = True):
        """
        Initialize the Pinecone vector store
        
        Args:
            api_key: Pinecone API key (if None, will try to get from environment)
            model_name: Name of the sentence transformer model
            index_name: Name of the Pinecone index
            environment: Pinecone environment/region
            enable_logging: Whether to enable detailed logging
        """
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError("Pinecone API key not provided. Set PINECONE_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name = model_name
        self.index_name = index_name
        self.environment = environment
        self.enable_logging = enable_logging
        
        # Setup logging
        if enable_logging:
            self._setup_logging()
        
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
        # Initialize or connect to index
        self._setup_index()
    
    def _setup_logging(self):
        """Setup logging for Pinecone queries and results"""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Setup logger for Pinecone queries
        self.query_logger = logging.getLogger('pinecone_queries')
        self.query_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.query_logger.handlers.clear()
        
        # Create file handler for query logs
        query_handler = logging.FileHandler('logs/pinecone_queries.log')
        query_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        query_handler.setFormatter(formatter)
        
        self.query_logger.addHandler(query_handler)
    
    def _log_query_and_results(self, query: str, results: List[Dict[str, Any]], user_filter: Optional[str] = None):
        """Log the query and results from Pinecone"""
        if not self.enable_logging:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'user_filter': user_filter,
            'results_count': len(results),
            'results': []
        }
        
        # Log each result
        for i, result in enumerate(results):
            result_entry = {
                'rank': i + 1,
                'similarity_score': result.get('similarity_score', 0),
                'user_name': result.get('user_name', 'Unknown'),
                'user_id': result.get('user_id', 'Unknown'),
                'content': result.get('content', ''),  # Full content for logging
                'timestamp': result.get('timestamp', 0),
                'datetime': result.get('datetime', '')
            }
            log_entry['results'].append(result_entry)
        
        # Log to file
        self.query_logger.info(f"PINECONE_QUERY: {json.dumps(log_entry, indent=2)}")
        
        # Also print to console for immediate feedback
        print(f"\n🔍 Pinecone query logged: '{query[:50]}...'")
        print(f"   Results returned: {len(results)}")
        if user_filter:
            print(f"   Filtered by user: {user_filter}")
        if results:
            print("   Top results:")
            for i, result in enumerate(results[:3], 1):
                score = result.get('similarity_score', 0)
                user = result.get('user_name', 'Unknown')
                content = result.get('content', '')
                print(f"     {i}. [{score:.3f}] {user}: {content}")
        
    def _setup_index(self):
        """Setup or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric='cosine',
                    spec={
                        'serverless': {
                            'cloud': 'aws',
                            'region': self.environment
                        }
                    }
                )
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    print("Waiting for index to be ready...")
                    time.sleep(1)
            else:
                print(f"Connecting to existing Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            print(f"✅ Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            print(f"❌ Failed to setup Pinecone index: {e}")
            raise e
    
    def _create_message_id(self, message: Dict[str, Any]) -> str:
        """Create a unique ID for a message"""
        # Use user_id + timestamp + content hash for uniqueness
        content_hash = hashlib.md5(message['content'].encode()).hexdigest()[:8]
        return f"{message['user_id']}_{int(message['timestamp'])}_{content_hash}"
    
    def create_embeddings_batch(self, messages: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict]:
        """
        Create embeddings for messages in batches
        
        Args:
            messages: List of normalized message dictionaries
            batch_size: Size of each batch for processing
            
        Returns:
            List of vectors ready for Pinecone upsert
        """
        print(f"Creating embeddings for {len(messages)} messages in batches of {batch_size}...")
        
        vectors_to_upsert = []
        
        for i in tqdm(range(0, len(messages), batch_size), desc="Creating embeddings"):
            batch = messages[i:i + batch_size]
            
            # Extract text content for this batch
            texts = []
            for message in batch:
                user_context = f"User {message['user_name']}: "
                content = message['content']
                texts.append(user_context + content)
            
            # Create embeddings for this batch
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                
                # Prepare vectors for Pinecone
                for j, (message, embedding) in enumerate(zip(batch, embeddings)):
                    vector_id = self._create_message_id(message)
                    
                    # Prepare metadata (Pinecone has limits on metadata size)
                    metadata = {
                        'user_id': message['user_id'],
                        'user_name': message['user_name'][:50],  # Limit length
                        'user_display_name': message.get('user_display_name', '')[:50],
                        'content': message['content'][:1000],  # Limit content length
                        'timestamp': float(message['timestamp']),
                        'datetime': message.get('datetime', ''),
                        'message_type': message.get('message_type', 'message')
                    }
                    
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding.tolist(),
                        'metadata': metadata
                    })
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        return vectors_to_upsert
    
    def upsert_vectors(self, vectors: List[Dict], batch_size: int = 100):
        """
        Upsert vectors to Pinecone in batches
        
        Args:
            vectors: List of vector dictionaries
            batch_size: Batch size for upsert operations
        """
        print(f"Upserting {len(vectors)} vectors to Pinecone...")
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting to Pinecone"):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"✅ Successfully upserted vectors to Pinecone")
    
    def build_index(self, messages: List[Dict[str, Any]], batch_size: int = 100):
        """
        Build Pinecone index from messages
        
        Args:
            messages: List of normalized message dictionaries
            batch_size: Batch size for processing
        """
        print("Building Pinecone vector index...")
        
        # Create embeddings and prepare vectors
        vectors = self.create_embeddings_batch(messages, batch_size)
        
        if vectors:
            # Upsert to Pinecone
            self.upsert_vectors(vectors, batch_size)
            
            # Wait for index to be updated (Pinecone needs time to process)
            print("⏳ Waiting for Pinecone to process vectors...")
            time.sleep(5)
            
            # Check index stats multiple times
            for attempt in range(3):
                stats = self.index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                print(f"📊 Attempt {attempt + 1}: {vector_count} vectors in index")
                
                if vector_count > 0:
                    print(f"✅ Built Pinecone index with {vector_count} vectors")
                    break
                elif attempt < 2:
                    print("⏳ Waiting a bit more...")
                    time.sleep(3)
                else:
                    print("⚠️  Index stats still show 0 vectors - but this might be a Pinecone delay")
                    print("   Vectors may still be available for search")
        else:
            print("❌ No vectors created")
    
    def search_similar(self, query: str, k: int = 5, user_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar messages using Pinecone
        
        Args:
            query: Query text to search for
            k: Number of similar messages to return
            user_filter: Optional user ID to filter results by specific user
            
        Returns:
            List of similar messages with similarity scores
        """
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Prepare filter
        filter_dict = {}
        if user_filter:
            filter_dict['user_id'] = user_filter
        
        # Search Pinecone with retry logic
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                search_results = self.index.query(
                    vector=query_embedding[0].tolist(),
                    top_k=k,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None
                )
                
                # Format results
                results = []
                for match in search_results['matches']:
                    result = {
                        'similarity_score': float(match['score']),
                        **match['metadata']
                    }
                    results.append(result)
                
                # Log the query and results
                self._log_query_and_results(query, results, user_filter)
                
                # If no results on first attempt, wait and retry
                if not results and attempt < max_retries:
                    print(f"⏳ No results on attempt {attempt + 1}, waiting and retrying...")
                    time.sleep(2)
                    continue
                
                return results
                
            except Exception as e:
                print(f"Error searching Pinecone (attempt {attempt + 1}): {e}")
                # Log the error
                if self.enable_logging:
                    error_log = {
                        'timestamp': datetime.now().isoformat(),
                        'query': query,
                        'user_filter': user_filter,
                        'error': str(e),
                        'results_count': 0,
                        'results': []
                    }
                    self.query_logger.error(f"PINECONE_ERROR: {json.dumps(error_log)}")
                
                if attempt < max_retries:
                    print(f"⏳ Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    return []
        
        # If all retries failed
        return []
    
    def get_user_message_context(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent messages from a specific user for context
        
        Args:
            user_id: User ID to get messages for
            limit: Maximum number of messages to return
            
        Returns:
            List of user's messages sorted by timestamp
        """
        try:
            # Query for user's messages
            results = self.index.query(
                vector=[0.0] * self.embedding_dim,  # Dummy vector
                top_k=limit,
                include_metadata=True,
                filter={'user_id': user_id}
            )
            
            # Sort by timestamp (most recent first)
            user_messages = []
            for match in results['matches']:
                user_messages.append(match['metadata'])
            
            user_messages.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            context_results = user_messages[:limit]
            
            # Log the user context query
            if self.enable_logging:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'query_type': 'user_context',
                    'user_id': user_id,
                    'limit': limit,
                    'results_count': len(context_results),
                    'results': []
                }
                
                # Log each result (limited to first 10 for brevity)
                for i, result in enumerate(context_results[:10]):
                    result_entry = {
                        'rank': i + 1,
                        'user_name': result.get('user_name', 'Unknown'),
                        'content': result.get('content', '')[:150] + '...' if len(result.get('content', '')) > 150 else result.get('content', ''),
                        'timestamp': result.get('timestamp', 0),
                        'datetime': result.get('datetime', '')
                    }
                    log_entry['results'].append(result_entry)
                
                self.query_logger.info(f"PINECONE_USER_CONTEXT: {json.dumps(log_entry, indent=2)}")
                print(f"🔍 User context query logged: {user_id} ({len(context_results)} messages)")
            
            return context_results
            
        except Exception as e:
            print(f"Error getting user context: {e}")
            # Log the error
            if self.enable_logging:
                error_log = {
                    'timestamp': datetime.now().isoformat(),
                    'query_type': 'user_context_error',
                    'user_id': user_id,
                    'error': str(e),
                    'results_count': 0
                }
                self.query_logger.error(f"PINECONE_USER_CONTEXT_ERROR: {json.dumps(error_log, indent=2)}")
            return []
    
    def get_conversation_context(self, query: str, target_user_id: str, context_size: int = 10) -> Dict[str, Any]:
        """
        Get comprehensive context for generating a response as a specific user
        
        Args:
            query: The input query/message
            target_user_id: ID of user to respond as
            context_size: Number of similar messages to include
            
        Returns:
            Dictionary containing similar messages and user context
        """
        # Get similar messages from all users
        similar_messages = self.search_similar(query, k=context_size)
        
        # Get recent messages from the target user for personality context
        user_context = self.get_user_message_context(target_user_id, limit=20)
        
        # Get the user's name for reference
        target_user_name = "Unknown User"
        if user_context:
            target_user_name = user_context[0].get('user_name', 'Unknown User')
        
        return {
            'query': query,
            'target_user_id': target_user_id,
            'target_user_name': target_user_name,
            'similar_messages': similar_messages,
            'user_recent_messages': user_context,
            'context_summary': self._create_context_summary(similar_messages, user_context)
        }
    
    def _create_context_summary(self, similar_messages: List[Dict], user_messages: List[Dict]) -> str:
        """Create a text summary of the context for the AI model"""
        summary_parts = []
        
        if similar_messages:
            summary_parts.append("Similar conversations:")
            for i, msg in enumerate(similar_messages[:5], 1):
                content = msg.get('content', '')[:200]
                user_name = msg.get('user_name', 'Unknown')
                summary_parts.append(f"{i}. {user_name}: {content}...")
        
        if user_messages:
            user_name = user_messages[0].get('user_name', 'Unknown User')
            summary_parts.append(f"\nRecent messages from {user_name}:")
            for i, msg in enumerate(user_messages[:5], 1):
                content = msg.get('content', '')[:150]
                summary_parts.append(f"{i}. {content}...")
        
        return "\n".join(summary_parts)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}
    
    def delete_index(self):
        """Delete the Pinecone index (use with caution!)"""
        try:
            self.pc.delete_index(self.index_name)
            print(f"✅ Deleted Pinecone index: {self.index_name}")
        except Exception as e:
            print(f"Error deleting index: {e}")


def build_pinecone_vector_store_from_normalized_data(
    normalized_data_file: str = "normalized_messages.json",
    api_key: Optional[str] = None,
    max_messages: Optional[int] = None,
    batch_size: int = 100
):
    """
    Convenience function to build Pinecone vector store from normalized data file
    
    Args:
        normalized_data_file: Path to normalized messages JSON file
        api_key: Pinecone API key
        max_messages: Maximum number of messages to process
        batch_size: Batch size for processing
    """
    print(f"Loading normalized data from {normalized_data_file}")
    with open(normalized_data_file, 'r') as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} normalized messages")
    
    if max_messages and len(messages) > max_messages:
        print(f"Limiting to {max_messages} most recent messages")
        messages = sorted(messages, key=lambda x: x['timestamp'], reverse=True)[:max_messages]
    
    # Create and build vector store
    vector_store = PineconeVectorStore(api_key=api_key)
    vector_store.build_index(messages, batch_size=batch_size)
    
    print("Pinecone vector store built successfully!")
    return vector_store


if __name__ == "__main__":
    # Example usage
    build_pinecone_vector_store_from_normalized_data()
