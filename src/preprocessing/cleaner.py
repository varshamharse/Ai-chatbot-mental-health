"""Text cleaner module for cleaning raw social media and conversational text."""

import re
import html
from typing import Optional


class TextCleaner:
    """Cleans raw text by stripping HTML, URLs, Reddit artifacts, and noise."""

    def __init__(
        self,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_special_chars: bool = True,
        keep_punctuation: bool = True
    ):
        """Initialize TextCleaner with cleaning options."""
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_special_chars = remove_special_chars
        self.keep_punctuation = keep_punctuation

        # Regex patterns
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+|<url>", re.IGNORECASE)
        self.html_tag_pattern = re.compile(r"<.*?>")
        self.reddit_user_sub_pattern = re.compile(r"\b[ru]/\w+\b", re.IGNORECASE)
        self.multispace_pattern = re.compile(r"\s+")
        self.repeated_chars_pattern = re.compile(r"(.)\1{2,}")

    def clean(self, text: Optional[str]) -> str:
        """Clean a single text string.
        
        Args:
            text: Input string.
            
        Returns:
            Cleaned string.
        """
        if not text or not isinstance(text, str):
            return ""

        # Unescape HTML entities (&amp; -> &, etc.)
        cleaned = html.unescape(text)

        # Remove HTML tags
        if self.remove_html:
            cleaned = self.html_tag_pattern.sub(" ", cleaned)

        # Remove URLs
        if self.remove_urls:
            cleaned = self.url_pattern.sub(" ", cleaned)

        # Remove Reddit sub/user tags (e.g. r/assistance)
        cleaned = self.reddit_user_sub_pattern.sub(" ", cleaned)

        # Normalize repeated characters (e.g. "sooooo" -> "soo")
        cleaned = self.repeated_chars_pattern.sub(r"\1\1", cleaned)

        # Clean special non-ASCII or unprintable characters if configured
        if self.remove_special_chars:
            cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)

        if not self.keep_punctuation:
            cleaned = re.sub(r"[^\w\s]", " ", cleaned)

        # Collapse whitespace
        cleaned = self.multispace_pattern.sub(" ", cleaned).strip()

        return cleaned

    def clean_series(self, series) -> list:
        """Clean an entire pandas Series or list of texts."""
        return [self.clean(t) for t in series]
