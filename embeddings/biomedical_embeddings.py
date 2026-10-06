"""
Biomedical Embeddings using Pre-trained Models
No GPU required - runs on CPU!
"""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BiomedicalEmbeddingGenerator:
    """
    Generates embeddings using pre-trained biomedical models.
    Uses: uc-ctds/bge-large-en-v1.5-bio-mapping
    """
    
    def __init__(self, model_name: str = "uc-ctds/bge-large-en-v1.5-bio-mapping"):
        """
        Initialize the embedding generator with a pre-trained model.
        
        Args:
            model_name: Hugging Face model name (default: biomedical embedding model)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model."""
        try:
            logger.info(f"Loading biomedical embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to a smaller model if the main one fails
            try:
                logger.info("Falling back to all-MiniLM-L6-v2...")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("✅ Fallback model loaded")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding for a text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of float values (embedding vector)
        """
        if not text or not text.strip():
            return []
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()