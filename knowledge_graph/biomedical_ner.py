"""
Biomedical Named Entity Recognition using Pre-trained Models
Extracts: Drugs, Diseases, Genes, Proteins, etc.
"""

import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logger = logging.getLogger(__name__)


class BiomedicalNER:
    """
    Named Entity Recognition for biomedical text.
    Uses: Pre-trained biomedical NER models
    """
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        """
        Initialize the NER model with a biomedical model.
        
        Args:
            model_name: Hugging Face model name for biomedical NER
        """
        self.model_name = model_name
        self.ner_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the biomedical NER model."""
        try:
            logger.info(f"Loading biomedical NER model: {self.model_name}")
            
            # Use AutoModelForTokenClassification with a biomedical model
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=9  # Standard NER labels
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.ner_pipeline = pipeline(
                "ner", 
                model=model, 
                tokenizer=tokenizer, 
                aggregation_strategy="simple",
                device=-1  # CPU
            )
            logger.info(f"✅ Biomedical NER model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            # Fallback to another biomedical model
            try:
                logger.info("Falling back to allenai/scibert_scivocab_uncased...")
                fallback_model = "allenai/scibert_scivocab_uncased"
                
                model = AutoModelForTokenClassification.from_pretrained(fallback_model)
                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.ner_pipeline = pipeline(
                    "ner", 
                    model=model, 
                    tokenizer=tokenizer, 
                    aggregation_strategy="simple",
                    device=-1
                )
                self.model_name = fallback_model
                logger.info(f"✅ Fallback model loaded: {fallback_model}")
                
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                # Last resort fallback
                try:
                    logger.info("Falling back to dslim/bert-base-NER (general NER)...")
                    fallback_model = "dslim/bert-base-NER"
                    model = AutoModelForTokenClassification.from_pretrained(fallback_model)
                    tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self.ner_pipeline = pipeline(
                        "ner", 
                        model=model, 
                        tokenizer=tokenizer, 
                        aggregation_strategy="simple",
                        device=-1
                    )
                    self.model_name = fallback_model
                    logger.info(f"✅ Fallback model loaded: {fallback_model}")
                except Exception as e3:
                    logger.error(f"Failed to load all models: {e3}")
                    raise
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract biomedical entities from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with entity types and their values
        """
        if not text or not text.strip():
            return {}
        
        try:
            # Run NER pipeline
            results = self.ner_pipeline(text)
            
            # Initialize entity dictionary
            entities = {
                "drugs": [],
                "diseases": [],
                "genes": [],
                "proteins": [],
                "organisms": [],
                "symptoms": [],
                "other": []
            }
            
            # Process results
            for entity in results:
                entity_type = entity.get("entity_group", "").lower()
                entity_word = entity.get("word", "")
                entity_score = entity.get("score", 0)
                
                # Skip low confidence entities
                if entity_score < 0.5:
                    continue
                
                # Map entity types to our categories
                if entity_type in ["drug", "chemical", "chem"]:
                    entities["drugs"].append(entity_word)
                elif entity_type in ["disease", "disorder"]:
                    entities["diseases"].append(entity_word)
                elif entity_type in ["gene", "protein"]:
                    entities["genes"].append(entity_word)
                elif entity_type in ["organism", "species"]:
                    entities["organisms"].append(entity_word)
                elif entity_type in ["symptom"]:
                    entities["symptoms"].append(entity_word)
                else:
                    entities["other"].append(entity_word)
            
            # Remove duplicates while preserving order
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            
            return entities
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return {}
    
    def extract_entities_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of entity dictionaries
        """
        return [self.extract_entities(text) for text in texts]