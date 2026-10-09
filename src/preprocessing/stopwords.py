"""Stopword handler with mental-health and negation preservation support."""

from typing import Set, List


DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "hed", "hell", "hes", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "if", "in", "into", "is", "it", "its",
    "itself", "more", "most", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "shed", "shell", "shes", "so", "some", "such", "than", "that", "thats",
    "the", "their", "theirs", "them", "themselves", "then", "there", "theres",
    "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "wed",
    "well", "were", "weve", "what", "whats", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "youd", "youll", "youre", "youve",
    "your", "yours", "yourself", "yourselves"
}

# Negations and emotive words critical to preserve for mental-health context
CRITICAL_NEGATIONS_AND_PRONOUNS: Set[str] = {
    "not", "no", "nor", "never", "cannot", "cant", "dont", "doesnt", "didnt",
    "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt", "wont",
    "wouldnt", "couldnt", "shouldnt", "nothing", "nobody", "nowhere", "none",
    "i", "me", "my", "myself"
}


class StopwordsHandler:
    """Removes non-informative stopwords while preserving crucial emotional/negation signals."""

    def __init__(self, preserve_negations: bool = True, custom_stopwords: Set[str] = None):
        self.preserve_negations = preserve_negations
        base_stops = set(DEFAULT_STOPWORDS)

        # Try to extend with NLTK stopwords if available
        try:
            from nltk.corpus import stopwords as nltk_stopwords
            base_stops.update(nltk_stopwords.words("english"))
        except Exception:
            pass

        if custom_stopwords:
            base_stops.update(custom_stopwords)

        if self.preserve_negations:
            self.stopwords = base_stops - CRITICAL_NEGATIONS_AND_PRONOUNS
        else:
            self.stopwords = base_stops

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out stopwords from a list of tokens."""
        return [t for t in tokens if t.lower() not in self.stopwords]

    def filter_text(self, text: str) -> str:
        """Remove stopwords from whitespace-delimited text."""
        tokens = text.split()
        return " ".join(self.remove_stopwords(tokens))
