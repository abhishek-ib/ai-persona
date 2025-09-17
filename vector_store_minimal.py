"""
Minimal Vector Store Implementation for AI Persona Project

This module provides a lightweight alternative that works without sentence-transformers
using scikit-learn's TfidfVectorizer for embeddings.
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


class MinimalVectorStore:
    def __init__(self, index_file: str = "message_index_minimal.pkl", 
                 metadata_file: str = "message_metadata_minimal.pkl"):
        """
        Initialize the minimal vector store using TF-IDF
        
        Args:
            index_file: Path to save/load TF-IDF vectors
            metadata_file: Path to save/load message metadata
        """
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        # Use TF-IDF instead of sentence transformers
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.vectors = None
        self.metadata = []
        
    def create_embeddings(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create TF-IDF embeddings for all messages
        
        Args:
            messages: List of normalized message dictionaries
            
        Returns:
            numpy array of TF-IDF vectors
        """
        print(f"Creating TF-IDF embeddings for {len(messages)} messages...")
        
        # Extract text content for embedding
        texts = []
        for message in messages:
            # Combine user context with message content
            user_context = f"User {message['user_name']}: "
            content = message['content']
            texts.append(user_context + content)
        
        # Create TF-IDF vectors
        vectors = self.vectorizer.fit_transform(texts)
        
        print(f"Created TF-IDF vectors with shape: {vectors.shape}")
        return vectors.toarray().astype('float32')
    
    def build_index(self, messages: List[Dict[str, Any]]):
        """
        Build TF-IDF index from messages
        
        Args:
            messages: List of normalized message dictionaries
        """
        print("Building TF-IDF index...")
        
        # Create embeddings
        self.vectors = self.create_embeddings(messages)
        
        # Store metadata
        self.metadata = []
        for message in messages:
            metadata_item = {
                'user_id': message['user_id'],
                'user_name': message['user_name'],
                'user_display_name': message['user_display_name'],
                'content': message['content'],
                'timestamp': message['timestamp'],
                'datetime': message['datetime'],
                'message_type': message['message_type'],
                'thread_ts': message.get('thread_ts')
            }
            self.metadata.append(metadata_item)
        
        print(f"Built index with {len(self.metadata)} messages")
        
    def save_index(self):
        """Save TF-IDF vectors and metadata to disk"""
        if self.vectors is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        print(f"Saving vectors and vectorizer to {self.index_file}")
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'vectorizer': self.vectorizer
            }, f)
        
        print(f"Saving metadata to {self.metadata_file}")
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self):
        """Load TF-IDF vectors and metadata from disk"""
        if not os.path.exists(self.index_file) or not os.path.exists(self.metadata_file):
            raise FileNotFoundError("Index files not found. Build index first.")
        
        print(f"Loading vectors from {self.index_file}")
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.vectorizer = data['vectorizer']
        
        print(f"Loading metadata from {self.metadata_file}")
        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {len(self.metadata)} messages")
    
    def search_similar(self, query: str, k: int = 5, user_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar messages using cosine similarity
        
        Args:
            query: Query text to search for
            k: Number of similar messages to return
            user_filter: Optional user ID to filter results by specific user
            
        Returns:
            List of similar messages with similarity scores
        """
        if self.vectors is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Transform query using the fitted vectorizer
        query_vector = self.vectorizer.transform([query]).toarray()
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k * 2]  # Get more for filtering
        
        results = []
        for idx in top_indices:
            if similarities[idx] < 0.01:  # Skip very low similarity scores
                continue
                
            message = self.metadata[idx].copy()
            message['similarity_score'] = float(similarities[idx])
            
            # Apply user filter if specified
            if user_filter is None or message['user_id'] == user_filter:
                results.append(message)
            
            if len(results) >= k:
                break
        
        return results
    
    def get_user_message_context(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent messages from a specific user for context
        
        Args:
            user_id: User ID to get messages for
            limit: Maximum number of messages to return
            
        Returns:
            List of user's messages sorted by timestamp
        """
        user_messages = [msg for msg in self.metadata if msg['user_id'] == user_id]
        user_messages.sort(key=lambda x: x['timestamp'], reverse=True)
        return user_messages[:limit]
    
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
            target_user_name = user_context[0]['user_name']
        
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
                summary_parts.append(f"{i}. {msg['user_name']}: {msg['content'][:200]}...")
        
        if user_messages:
            summary_parts.append(f"\nRecent messages from {user_messages[0]['user_name']}:")
            for i, msg in enumerate(user_messages[:5], 1):
                summary_parts.append(f"{i}. {msg['content'][:150]}...")
        
        return "\n".join(summary_parts)


def build_minimal_vector_store_from_normalized_data(normalized_data_file: str = "normalized_messages.json"):
    """
    Convenience function to build minimal vector store from normalized data file
    
    Args:
        normalized_data_file: Path to normalized messages JSON file
    """
    print(f"Loading normalized data from {normalized_data_file}")
    with open(normalized_data_file, 'r') as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} normalized messages")
    
    # Create and build vector store
    vector_store = MinimalVectorStore()
    vector_store.build_index(messages)
    vector_store.save_index()
    
    print("Minimal vector store built and saved successfully!")
    return vector_store


if __name__ == "__main__":
    # Example usage
    build_minimal_vector_store_from_normalized_data()
