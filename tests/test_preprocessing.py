"""Unit tests for NLP text preprocessing pipeline."""

import pytest
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.text_normalizer import TextNormalizer
from src.preprocessing.tokenizer import WordTokenizer
from src.preprocessing.stopwords import StopwordsHandler
from src.preprocessing.lemmatizer import TextLemmatizer
from src.preprocessing.pipeline import TextPreprocessingPipeline


def test_text_cleaner():
    cleaner = TextCleaner()
    raw = "Check this out <url> https://example.com/test and <b>bold text</b> with r/depression"
    cleaned = cleaner.clean(raw)
    assert "https://" not in cleaned
    assert "<b>" not in cleaned
    assert "r/depression" not in cleaned
    assert "bold text" in cleaned


def test_text_normalizer_contractions():
    normalizer = TextNormalizer(lowercase=True, expand_contractions=True)
    text = "I don't think I'll be fine, they're not here"
    norm = normalizer.normalize(text)
    assert "do not" in norm
    assert "i will" in norm
    assert "they are" in norm


def test_tokenizer():
    tokenizer = WordTokenizer()
    text = "I feel stressed and anxious today."
    tokens = tokenizer.tokenize(text)
    assert len(tokens) >= 5
    assert "stressed" in tokens
    assert "anxious" in tokens


def test_stopwords_preserves_negation():
    handler = StopwordsHandler(preserve_negations=True)
    tokens = ["i", "am", "not", "feeling", "good", "never"]
    filtered = handler.remove_stopwords(tokens)
    assert "not" in filtered
    assert "never" in filtered
    assert "am" not in filtered


def test_full_preprocessing_pipeline():
    pipeline = TextPreprocessingPipeline()
    sample = "I'm feeling completely stressed out and I can't sleep at night! Visit https://help.org"
    processed = pipeline.preprocess_text(sample)
    assert isinstance(processed, str)
    assert len(processed) > 0
    assert "https" not in processed
    assert "not" in processed or "cannot" in processed
