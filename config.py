"""Configuration management for language identifier."""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from core.exceptions import InvalidConfigError

DEFAULT_CONFIG = {
    "min_ngram_size": 1,
    "max_ngram_size": 5,
    "profile_size": 300,
    "default_method": "vector",
    "min_text_length": 15,
    "models_dir": os.path.join(os.path.dirname(__file__), "models"),
    "logging": {
        "level": "WARNING",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None
    }
}


@dataclass
class LanguageIdentifierConfig:
    """Configuration for language identifier."""
    min_ngram_size: int = 5
    max_ngram_size: int = 8
    profile_size: int = 300
    default_method: str = "combined"
    min_text_length: int = 10
    models_dir: str = os.path.join(os.path.dirname(__file__), "models")
    logging: Dict[str, Any] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_ngram_size < 1:
            raise InvalidConfigError("min_ngram_size must be at least 1")
        if self.max_ngram_size < self.min_ngram_size:
            raise InvalidConfigError("max_ngram_size must be >= min_ngram_size")
        if self.profile_size < 10:
            raise InvalidConfigError("profile_size must be at least 10")
        if self.default_method not in ["rank", "vector", "combined"]:
            raise InvalidConfigError("default_method must be one of: rank, vector, combined")
        if self.min_text_length < 1:
            raise InvalidConfigError("min_text_length must be at least 1")
        
        # Set default logging if not provided
        if self.logging is None:
            self.logging = DEFAULT_CONFIG["logging"]
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, file_path: str) -> None:
        """Save configuration to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> 'LanguageIdentifierConfig':
        """Load configuration from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise InvalidConfigError(f"Error loading configuration: {e}")


def get_default_config() -> LanguageIdentifierConfig:
    """Get default configuration."""
    return LanguageIdentifierConfig(**DEFAULT_CONFIG)