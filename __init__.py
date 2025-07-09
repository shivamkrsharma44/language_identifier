"""
Language Identifier - A library for language identification.

Based on: Cavnar, W. B., & Trenkle, J. M. (1994). "N-gram-based text categorization"
"""

__version__ = "1.0.0"

from .core.identifier import LanguageIdentifier
from ..extra.exceptions import (
    LanguageIdentifierError,
    ModelNotTrainedError,
    InvalidConfigError,
    DataProcessingError,
    ModelSerializationError,
)

__all__ = [
    "LanguageIdentifier",
    "LanguageIdentifierError",
    "ModelNotTrainedError",
    "InvalidConfigError",
    "DataProcessingError",
    "ModelSerializationError",
]