import asyncio
import os
import numpy as np
from typing import List, Union
from openai import OpenAI
import time

class EmbeddingGenerator:
    """
    Embedding generator using OpenAI's text-embedding models
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dimension = 1536  # Default dimension for text-embedding-3-small
        self.rate_limit_delay = 0.1  # Delay between requests to avoid rate limits
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            if not cleaned_text.strip():
                # Return zero vector for empty text
                return np.zeros(self.dimension, dtype=np.float32)
            
            # Add small delay to respect rate limits
            await asyncio.sleep(self.rate_limit_delay)
            
            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=cleaned_text,
                encoding_format="float"
            )
            
            # Extract embedding
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self._generate_batch(batch)
            embeddings.extend(batch_embeddings)
            
            # Add delay between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(self.rate_limit_delay * batch_size)
        
        return embeddings
    
    async def _generate_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a single batch"""
        try:
            # Clean texts
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Filter out empty texts but keep track of indices
            non_empty_texts = []
            text_indices = []
            
            for i, text in enumerate(cleaned_texts):
                if text.strip():
                    non_empty_texts.append(text)
                    text_indices.append(i)
            
            if not non_empty_texts:
                # Return zero vectors for all texts
                return [np.zeros(self.dimension, dtype=np.float32) for _ in texts]
            
            # Generate embeddings for non-empty texts
            response = self.client.embeddings.create(
                model=self.model,
                input=non_empty_texts,
                encoding_format="float"
            )
            
            # Create result array with zero vectors for empty texts
            embeddings = [np.zeros(self.dimension, dtype=np.float32) for _ in texts]
            
            # Fill in embeddings for non-empty texts
            for i, embedding_data in enumerate(response.data):
                original_index = text_indices[i]
                embeddings[original_index] = np.array(embedding_data.embedding, dtype=np.float32)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for embedding"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (OpenAI has token limits)
        max_chars = 8000  # Conservative limit
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text
    
    def set_model(self, model: str) -> None:
        """Change the embedding model"""
        self.model = model
        
        # Update dimension if known
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if model in model_dimensions:
            self.dimension = model_dimensions[model]
    
    def get_model_info(self) -> dict:
        """Get current model information"""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "rate_limit_delay": self.rate_limit_delay
        }
    
    def set_rate_limit_delay(self, delay: float) -> None:
        """Set delay between API requests"""
        self.rate_limit_delay = max(0, delay)
