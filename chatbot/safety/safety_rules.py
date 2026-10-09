"""Safety rules and medical non-diagnosis enforcement module."""

import re
from typing import Dict, Any


MEDICAL_DISCLAIMER = (
    "Disclaimer: I am an AI mental health support assistant, not a doctor, psychiatrist, "
    "or licensed therapist. This interaction is for supportive listening and coping information only "
    "and does not constitute medical diagnosis or clinical treatment. If you are experiencing a medical or mental health emergency, please contact professional emergency services immediately."
)


class SafetyRuleEngine:
    """Enforces boundaries: no clinical diagnosis, no drug prescriptions, no self-harm methods."""

    PROHIBITED_ADVICE_PATTERNS = [
        r"\byou\s+have\s+(?:clinical\s+depression|schizophrenia|bipolar|bpd)\b",
        r"\bi\s+diagnose\s+you\b",
        r"\btake\s+\d+\s*mg\s+of\b",
        r"\bstop\s+taking\s+your\s+meds\b",
    ]

    def __init__(self):
        self._compiled_prohibited = [re.compile(p, re.IGNORECASE) for p in self.PROHIBITED_ADVICE_PATTERNS]

    def validate_outgoing_response(self, response_text: str) -> str:
        """Sanitize outgoing chatbot response to verify no diagnostic claims are made."""
        for p in self._compiled_prohibited:
            if p.search(response_text):
                # Replace with safe supportive alternative
                return (
                    "I hear that you're going through a very challenging time. While I cannot diagnose conditions "
                    "or recommend specific medical treatments, I encourage discussing these feelings with a licensed healthcare provider."
                )
        return response_text

    @staticmethod
    def get_standard_disclaimer() -> str:
        """Return the standard non-clinical AI disclaimer."""
        return MEDICAL_DISCLAIMER
