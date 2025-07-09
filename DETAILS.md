# Language Identifier Details
## Dataset Source
Dataset obtained from University of Leipzig Wortschatz: https://wortschatz.uni-leipzig.de/en/download
- 30,000 sentences per language by default (English, French, German, Italian, Spanish)

## Overview
This tool implements the n-gram-based text categorization method described by Cavnar & Trenkle (1994).

**Reference:**
Cavnar, W. B., & Trenkle, J. M. (1994). "N-gram-based text categorization." Proceedings of SDAIR-94.

## Preprocessing Steps
1. Normalize text to lowercase, remove non-alphabetic characters, collapse whitespace.
2. Extract character n-grams (sizes 1–5 by default).
3. Build frequency profiles, select top 300 n-grams per language.

## Distance Metrics
- **vector**: Cosine distance between n-gram frequency vectors.
- **rank**: Out-of-place rank distance on top n-grams.
- **combined**: Normalized combination of vector and rank distances.

## Supported Languages
- English, French, German, Italian, Spanish.

## Assumptions
- All text in a single input is assumed to be in one language.
