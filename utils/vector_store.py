import faiss
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import pickle
import json

class VectorStore:
    """
    FAISS-based vector store for document embeddings
    """
    
    def __init__(self, dimension: int = 1000):  # Local TF-IDF embedding dimension
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0
    
    def add_vector(self, vector_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Add vector to the store"""
        try:
            # Normalize embedding for cosine similarity
            embedding = embedding.astype('float32')
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            
            # Store metadata and mappings
            current_index = self.next_index
            self.id_to_index[vector_id] = current_index
            self.index_to_id[current_index] = vector_id
            self.metadata_store[vector_id] = metadata
            
            self.next_index += 1
            
        except Exception as e:
            raise Exception(f"Failed to add vector {vector_id}: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_embedding = query_embedding.astype('float32')
            if np.linalg.norm(query_embedding) > 0:
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search FAISS index
            query_reshaped = query_embedding.reshape(1, -1)
            similarities, indices = self.index.search(query_reshaped, min(top_k, self.index.ntotal))
            
            # Convert results
            results = []
            for similarity, index in zip(similarities[0], indices[0]):
                if index in self.index_to_id:
                    vector_id = self.index_to_id[index]
                    results.append((vector_id, float(similarity)))
            
            return results
            
        except Exception as e:
            raise Exception(f"Vector search failed: {str(e)}")
    
    def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a vector"""
        return self.metadata_store.get(vector_id)
    
    def get_vector_count(self) -> int:
        """Get total number of vectors"""
        return self.index.ntotal
    
    def remove_vector(self, vector_id: str) -> bool:
        """Remove vector (FAISS doesn't support removal, so we mark as deleted)"""
        if vector_id in self.metadata_store:
            # Mark as deleted in metadata
            self.metadata_store[vector_id]["deleted"] = True
            return True
        return False
    
    def clear(self) -> None:
        """Clear all vectors"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.next_index = 0
    
    def save_to_disk(self, index_path: str, metadata_path: str) -> None:
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata and mappings
            save_data = {
                "metadata_store": self.metadata_store,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "next_index": self.next_index,
                "dimension": self.dimension
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
        except Exception as e:
            raise Exception(f"Failed to save vector store: {str(e)}")
    
    def load_from_disk(self, index_path: str, metadata_path: str) -> None:
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata and mappings
            with open(metadata_path, 'r') as f:
                save_data = json.load(f)
            
            self.metadata_store = save_data["metadata_store"]
            self.id_to_index = save_data["id_to_index"]
            # Convert string keys back to int for index_to_id
            self.index_to_id = {int(k): v for k, v in save_data["index_to_id"].items()}
            self.next_index = save_data["next_index"]
            self.dimension = save_data["dimension"]
            
        except Exception as e:
            raise Exception(f"Failed to load vector store: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        active_vectors = sum(1 for metadata in self.metadata_store.values() 
                           if not metadata.get("deleted", False))
        
        return {
            "total_vectors": self.index.ntotal,
            "active_vectors": active_vectors,
            "deleted_vectors": self.index.ntotal - active_vectors,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP"
        }
