"""Lemmatizer module with WordNet and rule-based fallback."""

from typing import List


class TextLemmatizer:
    """Lemmatizes tokens to base dictionary forms."""

    def __init__(self, use_pos: bool = False):
        self.use_pos = use_pos
        self.lemmatizer = None
        self._init_lemmatizer()

    def _init_lemmatizer(self):
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            # Test if wordnet resource is loaded
            lemmatizer.lemmatize("running")
            self.lemmatizer = lemmatizer
        except Exception:
            self.lemmatizer = None

    def lemmatize_token(self, token: str, pos: str = "v") -> str:
        """Lemmatize single token."""
        token = token.lower()
        if self.lemmatizer:
            try:
                lemma = self.lemmatizer.lemmatize(token, pos=pos)
                if lemma == token:
                    lemma = self.lemmatizer.lemmatize(token, pos="n")
                return lemma
            except Exception:
                pass

        # Rule-based fallback if WordNet resource not present
        if len(token) > 4:
            if token.endswith("ies"):
                return token[:-3] + "y"
            elif token.endswith("ing"):
                return token[:-3]
            elif token.endswith("ed"):
                return token[:-2]
            elif token.endswith("s") and not token.endswith("ss"):
                return token[:-1]
        return token

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize a sequence of tokens."""
        return [self.lemmatize_token(t) for t in tokens]
