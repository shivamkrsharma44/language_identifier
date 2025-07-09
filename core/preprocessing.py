"""Text preprocessing utilities for language identification."""

import re
import unicodedata
from typing import Optional, List, Set

# Regex patterns
WHITESPACE_PATTERN = re.compile(r'\s+')
NUMBER_PATTERN = re.compile(r'\d+')


def normalize_text(text: str, 
                   lowercase: bool = True, 
                   remove_numbers: bool = True,
                   remove_punctuation: bool = False,
                   normalize_unicode: bool = True) -> str:
    """
    Normalize text for n-gram extraction.
    
    Args:
        text: Text to normalize
        lowercase: Convert to lowercase
        remove_numbers: Remove numeric characters
        remove_punctuation: Remove punctuation marks
        normalize_unicode: Convert to NFKD normalized form
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Handle Unicode normalization
    if normalize_unicode:
        text = unicodedata.normalize('NFKD', text)
    
    # Lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove numbers if requested
    if remove_numbers:
        text = NUMBER_PATTERN.sub('', text)
    
    # Remove punctuation if requested
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(' ', text)
    
    return text.strip()


def extract_ngrams(text: str, 
                   min_n: int, 
                   max_n: int, 
                   pad: bool = True,
                   pad_char: str = '_') -> List[str]:
    """
    Extract n-grams from text.
    
    Args:
        text: Text to extract n-grams from
        min_n: Minimum n-gram size
        max_n: Maximum n-gram size
        pad: Whether to pad the text
        pad_char: Character to use for padding
        
    Returns:
        List of n-grams
    """
    if not text:
        return []
    
    ngrams = []
    
    if pad:
        # Handle each n-gram size separately to ensure proper padding
        for n in range(min_n, max_n + 1):
            # Add enough padding characters for this n-gram size
            padded_text = pad_char + text + pad_char * (n-1)
            
            # Extract exactly (len(text) + 1) n-grams of size n
            for i in range(len(text) + 1):
                ngrams.append(padded_text[i:i+n])
    else:
        # If no padding, extract n-grams as before
        for n in range(min_n, max_n + 1):
            ngrams.extend([text[i:i+n] for i in range(len(text) - n + 1)])
    
    return ngrams