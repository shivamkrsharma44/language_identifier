"""Custom exceptions for the language identifier package."""

class LanguageIdentifierError(Exception):
    """Base exception for all language identifier errors."""
    pass


class ModelNotTrainedError(LanguageIdentifierError):
    """Raised when trying to use a model that hasn't been trained."""
    pass


class InvalidConfigError(LanguageIdentifierError):
    """Raised when configuration parameters are invalid."""
    pass


class DataProcessingError(LanguageIdentifierError):
    """Raised when there's an error processing input data."""
    pass


class ModelSerializationError(LanguageIdentifierError):
    """Raised when there's an error serializing or deserializing a model."""
    pass