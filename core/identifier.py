"""Core language identification functionality."""

import os
import math
import logging
import pickle
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any


from config import LanguageIdentifierConfig, get_default_config
from .preprocessing import normalize_text, extract_ngrams
from .exceptions import ModelNotTrainedError, DataProcessingError, ModelSerializationError

# Setup logging
logger = logging.getLogger(__name__)


class LanguageIdentifier:
    """
    Language identification based on n-gram frequency analysis.
    
    Based on: Cavnar, W. B., & Trenkle, J. M. (1994). "N-gram-based text categorization"
    """
    
    def __init__(self, config: Optional[LanguageIdentifierConfig] = None):
        """
        Initialize the language identifier.
        
        Args:
            config: Configuration for the language identifier
        """
        self.config = config or get_default_config()
        self._setup_logging()
        
        self.language_profiles = {}
        self.languages = set()
        self.model_metadata = {
            "version": 1,
            "trained": False,
            "languages": [],
            "config": self.config.to_dict()
        }
        
        logger.info(f"Language identifier initialized with config: {self.config}")
    
    def _setup_logging(self) -> None:
        """Configure logging based on configuration."""
        log_config = self.config.logging
        logging.basicConfig(
            level=getattr(logging, log_config["level"]),
            format=log_config["format"],
            filename=log_config["file"]
        )
    
    def _extract_profile(self, text: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Extract n-gram profile from text.
        
        Args:
            text: Text to extract profile from
            
        Returns:
            Tuple of (ranked_ngrams, frequency_map)
        """
        # Normalize text
        normalized = normalize_text(text)
        
        if len(normalized) < self.config.min_text_length:
            logger.warning(f"Text too short ({len(normalized)} chars) for reliable analysis")
        
        # Extract n-grams
        ngrams = extract_ngrams(
            normalized,
            self.config.min_ngram_size,
            self.config.max_ngram_size
        )
        
        # Count n-grams
        ngram_counts = Counter(ngrams)
        
        # Get most common n-grams
        most_common = ngram_counts.most_common(self.config.profile_size)
        total = sum(ngram_counts.values()) or 1  # Avoid division by zero
        
        # Create ranked list and frequency map
        ranked_ngrams = [ngram for ngram, _ in most_common]
        freq_map = {ngram: count/total for ngram, count in most_common}
        
        return ranked_ngrams, freq_map
    
    def train(self, 
              language_samples: Dict[str, List[str]], 
              model_name: Optional[str] = None) -> None:
        """
        Train the language identifier on sample texts.
        
        Args:
            language_samples: Dictionary mapping language codes to lists of sample texts
            model_name: Name to save the trained model as
        """
        logger.info(f"Training language identifier on {len(language_samples)} languages")
        
        if not language_samples:
            raise DataProcessingError("No language samples provided for training")
        
        self.language_profiles = {}
        self.languages = set(language_samples.keys())
        
        for language, samples in language_samples.items():
            if not samples:
                logger.warning(f"No samples provided for language {language}")
                continue
                
            logger.info(f"Processing {len(samples)} samples for language {language}")
            
            # Combine samples
            combined_text = " ".join(samples)
            
            # Extract profile
            ranked_ngrams, freq_map = self._extract_profile(combined_text)
            
            # Store profile
            self.language_profiles[language] = {
                'ranked_ngrams': ranked_ngrams,
                'freq_map': freq_map
            }
        
        # Update metadata
        self.model_metadata.update({
            "trained": True,
            "languages": list(self.languages),
            "num_languages": len(self.languages),
            "samples_per_language": {lang: len(samples) for lang, samples in language_samples.items()}
        })
        
        logger.info(f"Training completed for languages: {', '.join(self.languages)}")
        
        # Save model if name provided
        if model_name:
            self.save_model(model_name)
    
    def train_from_directory(self, 
                            directory: str, 
                            file_extension: str = '.txt', 
                            model_name: Optional[str] = None) -> None:
        """
        Train from a directory of language samples.
        
        Args:
            directory: Path to directory containing language samples
            file_extension: File extension for sample files
            model_name: Name to save the trained model as
        """
        logger.info(f"Training from directory: {directory}")
        
        if not os.path.isdir(directory):
            raise DataProcessingError(f"Directory not found: {directory}")
        
        language_samples = {}
        language_dirs = []
        
        # List language directories
        try:
            language_dirs = [d for d in os.listdir(directory) 
                           if os.path.isdir(os.path.join(directory, d))]
        except Exception as e:
            raise DataProcessingError(f"Error reading directory {directory}: {e}")
        
        if not language_dirs:
            raise DataProcessingError(f"No language directories found in {directory}")
        
        # Process each language directory
        for lang_dir in language_dirs:
            lang_path = os.path.join(directory, lang_dir)
            language = lang_dir
            samples = []
            
            logger.info(f"Processing language directory: {lang_dir}")
            
            # Find all sample files
            try:
                files = [f for f in os.listdir(lang_path) 
                       if f.endswith(file_extension)]
            except Exception as e:
                logger.error(f"Error listing files in {lang_path}: {e}")
                continue
            
            logger.info(f"Found {len(files)} sample files for language {language}")
            
            # Read each sample file
            for filename in files:
                file_path = os.path.join(lang_path, filename)
                
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            samples.append(f.read())
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                        break
            
            if samples:
                language_samples[language] = samples
            else:
                logger.warning(f"No valid samples found for language {language}")
        
        if not language_samples:
            raise DataProcessingError("No valid language samples found")
        
        # Train on collected samples
        self.train(language_samples, model_name)
    
    def identify_language(self, 
                         text: str, 
                         method: Optional[str] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Identify the language of the given text.
        
        Args:
            text: Text to identify
            method: Comparison method ('rank', 'vector', or 'combined')
            
        Returns:
            Tuple of (language_code, confidence_score, all_scores)
        """
        if not self.language_profiles:
            raise ModelNotTrainedError("Language identifier has not been trained")
        
        method = method or self.config.default_method
        
        if method not in ["rank", "vector", "combined"]:
            logger.warning(f"Unknown method '{method}', falling back to '{self.config.default_method}'")
            method = self.config.default_method
        
        logger.debug(f"Identifying language using method: {method}")
        
        # Extract profile from text
        text_ranked, text_freq = self._extract_profile(text)
        
        distances = {}
        
        # Calculate distances to each language profile
        for language, profile in self.language_profiles.items():
            if method == 'rank':
                distance = self._out_of_place_measure(
                    text_ranked, profile['ranked_ngrams'])
            elif method == 'vector':
                distance = self._cosine_distance(
                    text_freq, profile['freq_map'])
            else:  # 'combined'
                rank_dist = self._out_of_place_measure(
                    text_ranked, profile['ranked_ngrams'])
                vector_dist = self._cosine_distance(
                    text_freq, profile['freq_map'])
                
                # Normalize and combine distances
                max_rank_dist = self.config.profile_size ** 2
                norm_rank_dist = rank_dist / max_rank_dist if max_rank_dist > 0 else 0
                distance = (norm_rank_dist + vector_dist) / 2
            
            distances[language] = distance
        
        # Find best match
        best_language = min(distances, key=distances.get)
        
        # Convert distances to confidence scores
        scores = {}
        for lang, dist in distances.items():
            if method == 'rank':
                max_distance = self.config.profile_size ** 2
                scores[lang] = 1.0 - (dist / max_distance if max_distance > 0 else 0)
            else:
                scores[lang] = 1.0 - dist
        
        return best_language, scores[best_language], scores
    
    def _out_of_place_measure(self, profile1: List[str], profile2: List[str]) -> float:
        """
        Calculate the out-of-place measure between two profiles.
        
        Args:
            profile1: First profile as a list of n-grams
            profile2: Second profile as a list of n-grams
            
        Returns:
            Distance between profiles
        """
        distance = 0
        pos1 = {ngram: i for i, ngram in enumerate(profile1)}
        
        for i, ngram in enumerate(profile2):
            if ngram in pos1:
                distance += abs(i - pos1[ngram])
            else:
                distance += self.config.profile_size
        
        return distance
    
    def _cosine_distance(self, profile1: Dict[str, float], profile2: Dict[str, float]) -> float:
        """
        Calculate cosine distance between two n-gram profiles.
        
        Args:
            profile1: First profile as a frequency map
            profile2: Second profile as a frequency map
            
        Returns:
            Cosine distance between profiles
        """
        common_ngrams = set(profile1.keys()) & set(profile2.keys())
        
        if not common_ngrams:
            return 1.0
        
        dot_product = sum(profile1[ngram] * profile2[ngram] for ngram in common_ngrams)
        
        magnitude1 = math.sqrt(sum(freq**2 for freq in profile1.values()))
        magnitude2 = math.sqrt(sum(freq**2 for freq in profile2.values()))
        
        if magnitude1 * magnitude2 == 0:
            return 1.0
        
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        return 1.0 - max(0.0, min(1.0, cosine_similarity))
    
    def save_model(self, model_name: str) -> str:
        """
        Save the trained model to disk.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the saved model
        """
        if not self.language_profiles:
            raise ModelNotTrainedError("Cannot save untrained model")
        
        # Clean up model name
        model_name = model_name.replace(" ", "_").lower()
        if not model_name.endswith(".lid"):
            model_name += ".lid"
        
        model_path = os.path.join(self.config.models_dir, model_name)
        
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model data
            model_data = {
                "metadata": self.model_metadata,
                "config": self.config.to_dict(),
                "profiles": self.language_profiles
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            error_msg = f"Error saving model to {model_path}: {str(e)}"
            logger.error(error_msg)
            raise ModelSerializationError(error_msg)
    
    @classmethod
    def load_model(cls, model_path: str) -> 'LanguageIdentifier':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded LanguageIdentifier instance
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance with saved config
            config = LanguageIdentifierConfig(**model_data["config"])
            instance = cls(config)
            
            # Restore model state
            instance.language_profiles = model_data["profiles"]
            instance.languages = set(model_data["profiles"].keys())
            instance.model_metadata = model_data["metadata"]
            
            logger.info(f"Model loaded from {model_path}")
            return instance
            
        except Exception as e:
            error_msg = f"Error loading model from {model_path}: {str(e)}"
            logger.error(error_msg)
            raise ModelSerializationError(error_msg)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.languages)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "trained": bool(self.language_profiles),
            "languages": list(self.languages),
            "num_languages": len(self.languages),
            "config": self.config.to_dict(),
            "metadata": self.model_metadata
        }