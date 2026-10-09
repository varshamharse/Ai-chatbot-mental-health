"""Tokenizer module for word-level and subword tokenization."""

import re
from typing import List


class WordTokenizer:
    """Configurable word tokenizer with fallback support."""

    def __init__(self, use_nltk: bool = False, min_token_len: int = 1):
        self.use_nltk = use_nltk
        self.min_token_len = min_token_len
        self._regex_pattern = re.compile(r"\b\w+(?:'\w+)?\b")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of word tokens.
        
        Args:
            text: Input string.
            
        Returns:
            List of word tokens.
        """
        if not text:
            return []

        tokens: List[str] = []
        if self.use_nltk:
            try:
                import nltk
                tokens = nltk.word_tokenize(text)
            except Exception:
                tokens = self._regex_pattern.findall(text)
        else:
            tokens = self._regex_pattern.findall(text)

        if self.min_token_len > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_len]

        return tokens
