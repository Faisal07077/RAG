import asyncio
import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class LocalEmbeddingGenerator:
    """
    Local embedding generator using TF-IDF instead of OpenAI
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.dimension = 1000  # TF-IDF dimension
        self.is_fitted = False
        self.corpus = []
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            if not cleaned_text.strip():
                # Return zero vector for empty text
                return np.zeros(self.dimension, dtype=np.float32)
            
            # Add to corpus if not fitted yet
            if not self.is_fitted:
                self.corpus.append(cleaned_text)
                # Fit vectorizer with current corpus
                self.vectorizer.fit(self.corpus)
                self.is_fitted = True
            else:
                # Add to corpus and refit
                self.corpus.append(cleaned_text)
                self.vectorizer.fit(self.corpus)
            
            # Transform text to vector
            sparse_vector = self.vectorizer.transform([cleaned_text])
            try:
                vector = sparse_vector.toarray()[0]
            except (AttributeError, TypeError):
                # Handle case where sparse_vector is not sparse or has different structure
                if hasattr(sparse_vector, '__getitem__') and hasattr(sparse_vector, '__len__'):
                    vector = np.array(sparse_vector[0] if len(sparse_vector) > 0 else [])
                else:
                    vector = np.array(sparse_vector) if hasattr(sparse_vector, '__iter__') else np.zeros(self.dimension)
            
            # Ensure consistent dimension and type
            vector = np.array(vector, dtype=np.float32)
            if len(vector) < self.dimension:
                # Pad with zeros if needed
                padded_vector = np.zeros(self.dimension, dtype=np.float32)
                padded_vector[:len(vector)] = vector
                return padded_vector
            else:
                return vector[:self.dimension]
            
        except Exception as e:
            print(f"Warning: Failed to generate embedding: {str(e)}")
            return np.random.rand(self.dimension).astype(np.float32) * 0.1
    
    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for embedding"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower().strip()
        
        # Truncate if too long
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text
    
    def get_model_info(self) -> dict:
        """Get current model information"""
        return {
            "model": "TF-IDF Local",
            "dimension": self.dimension,
            "is_fitted": self.is_fitted,
            "corpus_size": len(self.corpus)
        }