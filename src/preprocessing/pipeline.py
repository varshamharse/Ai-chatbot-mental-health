"""Unified text preprocessing pipeline orchestration."""

import os
import yaml
from typing import List, Optional, Union, Dict, Any
import pandas as pd

from .cleaner import TextCleaner
from .text_normalizer import TextNormalizer
from .tokenizer import WordTokenizer
from .stopwords import StopwordsHandler
from .lemmatizer import TextLemmatizer


class TextPreprocessingPipeline:
    """Full text preprocessing pipeline for Mental Health AI Chatbot."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """Initialize pipeline with configuration dict or path."""
        self.config = config or {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                self.config = loaded.get("preprocessing", loaded)

        # Initialize sub-modules from config
        self.cleaner = TextCleaner(
            remove_urls=self.config.get("remove_urls", True),
            remove_html=self.config.get("remove_html", True),
            remove_special_chars=self.config.get("remove_special_chars", True),
            keep_punctuation=not self.config.get("remove_punctuation", False)
        )
        self.normalizer = TextNormalizer(
            lowercase=self.config.get("lowercase", True),
            expand_contractions=self.config.get("expand_contractions", True)
        )
        self.tokenizer = WordTokenizer(
            use_nltk=False,
            min_token_len=self.config.get("min_token_length", 2)
        )
        self.stopwords_handler = StopwordsHandler(
            preserve_negations=self.config.get("preserve_sentiment_negations", True)
        )
        self.lemmatizer = TextLemmatizer() if self.config.get("lemmatize", True) else None
        self.do_stopwords = self.config.get("stopwords_removal", True)

    def preprocess_text(self, text: str) -> str:
        """Process a single text sample into clean tokenized normalized text.
        
        Args:
            text: Raw input string.
            
        Returns:
            Preprocessed normalized text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Clean
        cleaned = self.cleaner.clean(text)

        # 2. Normalize casing & contractions
        normalized = self.normalizer.normalize(cleaned)

        # 3. Tokenize
        tokens = self.tokenizer.tokenize(normalized)

        # 4. Remove stopwords
        if self.do_stopwords:
            tokens = self.stopwords_handler.remove_stopwords(tokens)

        # 5. Lemmatize
        if self.lemmatizer:
            tokens = self.lemmatizer.lemmatize_tokens(tokens)

        return " ".join(tokens)

    def transform(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """Preprocess a sequence of text strings."""
        if isinstance(texts, pd.Series):
            texts = texts.fillna("").tolist()
        return [self.preprocess_text(t) for t in texts]

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        output_column: str = "cleaned_text"
    ) -> pd.DataFrame:
        """Preprocess text column in a DataFrame and return updated DataFrame."""
        result_df = df.copy()
        result_df[output_column] = self.transform(result_df[text_column])
        return result_df
