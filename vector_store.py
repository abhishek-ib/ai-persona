"""
Vector Store Implementation for AI Persona Project

This module creates and manages a vector database using FAISS and all-MiniLM-L6-v2 embeddings
for similarity search of normalized messages.
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import os


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_file: str = "message_index.faiss", 
                 metadata_file: str = "message_metadata.pkl"):
        """
        Initialize the vector store
        
        Args:
            model_name: Name of the sentence transformer model
            index_file: Path to save/load FAISS index
            metadata_file: Path to save/load message metadata
        """
        self.model_name = model_name
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.metadata = []
        
    def create_embeddings(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create embeddings for all messages
        
        Args:
            messages: List of normalized message dictionaries
            
        Returns:
            numpy array of embeddings
        """
        print(f"Creating embeddings for {len(messages)} messages...")
        
        # Extract text content for embedding
        texts = []
        for message in messages:
            # Combine user context with message content for better embeddings
            user_context = f"User {message['user_name']}: "
            content = message['content']
            texts.append(user_context + content)
        
        # Create embeddings in batches for efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                print(f"Processed {i + len(batch_texts)}/{len(texts)} messages")
        
        return np.array(embeddings).astype('float32')
    
    def build_index(self, messages: List[Dict[str, Any]]):
        """
        Build FAISS index from messages
        
        Args:
            messages: List of normalized message dictionaries
        """
        print("Building FAISS index...")
        
        # Create embeddings
        embeddings = self.create_embeddings(messages)
        
        # Create FAISS index
        # Using IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store metadata (message info without embeddings)
        self.metadata = []
        for message in messages:
            # Store essential metadata for retrieval
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
        
        print(f"Built index with {self.index.ntotal} messages")
        
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        print(f"Saving index to {self.index_file}")
        faiss.write_index(self.index, self.index_file)
        
        print(f"Saving metadata to {self.metadata_file}")
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if not os.path.exists(self.index_file) or not os.path.exists(self.metadata_file):
            raise FileNotFoundError("Index files not found. Build index first.")
        
        print(f"Loading index from {self.index_file}")
        self.index = faiss.read_index(self.index_file)
        
        print(f"Loading metadata from {self.metadata_file}")
        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {len(self.metadata)} messages")
    
    def search_similar(self, query: str, k: int = 5, user_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar messages
        
        Args:
            query: Query text to search for
            k: Number of similar messages to return
            user_filter: Optional user ID to filter results by specific user
            
        Returns:
            List of similar messages with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k * 2)  # Get more results for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
                
            message = self.metadata[idx].copy()
            message['similarity_score'] = float(score)
            
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
        # Sort by timestamp (most recent first)
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


def build_vector_store_from_normalized_data(normalized_data_file: str = "normalized_messages.json"):
    """
    Convenience function to build vector store from normalized data file
    
    Args:
        normalized_data_file: Path to normalized messages JSON file
    """
    print(f"Loading normalized data from {normalized_data_file}")
    with open(normalized_data_file, 'r') as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} normalized messages")
    
    # Create and build vector store
    vector_store = VectorStore()
    vector_store.build_index(messages)
    vector_store.save_index()
    
    print("Vector store built and saved successfully!")
    return vector_store


if __name__ == "__main__":
    # Example usage
    build_vector_store_from_normalized_data()
