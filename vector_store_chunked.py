"""
Memory-Efficient Vector Store Implementation for Large Datasets

This module handles very large datasets by processing embeddings in chunks
and using memory-efficient techniques.
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import gc
from tqdm import tqdm


class ChunkedVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 index_file: str = "message_index_chunked.faiss",
                 metadata_file: str = "message_metadata_chunked.pkl",
                 chunk_size: int = 1000):
        """
        Initialize memory-efficient vector store
        
        Args:
            model_name: Name of the sentence transformer model
            index_file: Path to save/load FAISS index
            metadata_file: Path to save/load message metadata
            chunk_size: Number of messages to process at once
        """
        self.model_name = model_name
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.chunk_size = chunk_size
        
        # Only import and initialize when needed
        self.model = None
        self.embedding_dim = None
        self.index = None
        self.metadata = []
        
    def _init_model(self):
        """Initialize the sentence transformer model lazily"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading sentence transformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"Failed to load sentence transformers: {e}")
                print("Falling back to TF-IDF...")
                raise e
    
    def create_embeddings_chunked(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create embeddings for messages in chunks to avoid memory issues
        
        Args:
            messages: List of normalized message dictionaries
            
        Returns:
            numpy array of embeddings
        """
        self._init_model()
        
        print(f"Creating embeddings for {len(messages)} messages in chunks of {self.chunk_size}...")
        
        all_embeddings = []
        
        # Process in chunks
        for i in tqdm(range(0, len(messages), self.chunk_size), desc="Processing chunks"):
            chunk = messages[i:i + self.chunk_size]
            
            # Extract text content for this chunk
            texts = []
            for message in chunk:
                user_context = f"User {message['user_name']}: "
                content = message['content']
                texts.append(user_context + content)
            
            try:
                # Create embeddings for this chunk
                chunk_embeddings = self.model.encode(
                    texts, 
                    show_progress_bar=False,
                    batch_size=32,
                    convert_to_numpy=True
                )
                all_embeddings.append(chunk_embeddings)
                
                # Force garbage collection to free memory
                gc.collect()
                
            except Exception as e:
                print(f"Error processing chunk {i//self.chunk_size + 1}: {e}")
                # Create zero embeddings as fallback
                zero_embeddings = np.zeros((len(texts), self.embedding_dim), dtype='float32')
                all_embeddings.append(zero_embeddings)
        
        # Combine all chunks
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings).astype('float32')
            print(f"Created embeddings with shape: {final_embeddings.shape}")
            return final_embeddings
        else:
            raise ValueError("No embeddings created")
    
    def build_index(self, messages: List[Dict[str, Any]], max_messages: Optional[int] = None):
        """
        Build FAISS index from messages with memory efficiency
        
        Args:
            messages: List of normalized message dictionaries
            max_messages: Optional limit on number of messages to process
        """
        if max_messages and len(messages) > max_messages:
            print(f"Limiting to {max_messages} most recent messages for memory efficiency")
            # Sort by timestamp and take most recent
            messages = sorted(messages, key=lambda x: x['timestamp'], reverse=True)[:max_messages]
        
        print("Building memory-efficient FAISS index...")
        
        try:
            # Create embeddings in chunks
            embeddings = self.create_embeddings_chunked(messages)
            
            # Create FAISS index
            import faiss
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
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
            
            print(f"Built index with {self.index.ntotal} messages")
            
        except Exception as e:
            print(f"Failed to build neural embeddings index: {e}")
            raise e
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        import faiss
        print(f"Saving index to {self.index_file}")
        faiss.write_index(self.index, self.index_file)
        
        print(f"Saving metadata to {self.metadata_file}")
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        if not os.path.exists(self.index_file) or not os.path.exists(self.metadata_file):
            raise FileNotFoundError("Index files not found. Build index first.")
        
        import faiss
        print(f"Loading index from {self.index_file}")
        self.index = faiss.read_index(self.index_file)
        
        print(f"Loading metadata from {self.metadata_file}")
        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {len(self.metadata)} messages")
    
    def search_similar(self, query: str, k: int = 5, user_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar messages"""
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        self._init_model()
        
        # Create query embedding
        import faiss
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k * 2)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                break
                
            message = self.metadata[idx].copy()
            message['similarity_score'] = float(score)
            
            if user_filter is None or message['user_id'] == user_filter:
                results.append(message)
            
            if len(results) >= k:
                break
        
        return results


def build_chunked_vector_store_from_normalized_data(
    normalized_data_file: str = "normalized_messages.json",
    max_messages: int = 50000,  # Limit for memory safety
    chunk_size: int = 1000
):
    """
    Build chunked vector store from normalized data file
    
    Args:
        normalized_data_file: Path to normalized messages JSON file
        max_messages: Maximum number of messages to process (for memory safety)
        chunk_size: Size of processing chunks
    """
    print(f"Loading normalized data from {normalized_data_file}")
    with open(normalized_data_file, 'r') as f:
        messages = json.load(f)
    
    print(f"Loaded {len(messages)} normalized messages")
    
    if len(messages) > max_messages:
        print(f"⚠️  Dataset is very large ({len(messages)} messages)")
        print(f"Limiting to {max_messages} most recent messages for memory safety")
    
    # Create and build vector store
    vector_store = ChunkedVectorStore(chunk_size=chunk_size)
    vector_store.build_index(messages, max_messages=max_messages)
    vector_store.save_index()
    
    print("Chunked vector store built and saved successfully!")
    return vector_store


if __name__ == "__main__":
    build_chunked_vector_store_from_normalized_data()
