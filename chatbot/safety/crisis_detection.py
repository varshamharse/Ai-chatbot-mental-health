"""Crisis detection and risk level assessment module for mental health safety."""

import re
from typing import Dict, Any, List


class CrisisDetector:
    """Detects crisis cues, self-harm signals, suicidal ideation, and acute distress."""

    CRITICAL_PATTERNS = [
        r"\b(?:kill|end|take)\s+my\s*life\b",
        r"\b(?:want|going)\s+to\s+die\b",
        r"\bsuicid(?:e|al)\b",
        r"\bkill\s+myself\b",
        r"\bend\s+it\s+all\b",
        r"\bhang\s+myself\b",
        r"\boverdose\b",
        r"\bcut(?:ting)?\s+my\s*(?:wrists?|arms?|thighs?)\b",
        r"\bself[\s-]harm\b",
        r"\bno\s+reason\s+to\s+live\b",
        r"\bbetter\s+off\s+dead\b",
        r"\bdon'?t\s+want\s+to\s+wake\s+up\b",
    ]

    SEVERE_PATTERNS = [
        r"\bhurt\s+myself\b",
        r"\bcan'?t\s+take\s+this\s+anymore\b",
        r"\beverything\s+is\s+hopeless\b",
        r"\bi\s+hate\s+my\s+life\b",
        r"\bsevere\s+panic\s+attack\b",
        r"\bbeing\s+abused\b",
        r"\bphysically\s+abused\b",
        r"\bdomestic\s+violence\b",
    ]

    MODERATE_PATTERNS = [
        r"\bfeeling\s+overwhelmed\b",
        r"\banxious\s+and\s+scared\b",
        r"\bcan'?t\s+stop\s+crying\b",
        r"\bfeeling\s+so\s+alone\b",
        r"\bdepressed\b",
        r"\bpanic\s+attack\b",
    ]

    def __init__(self):
        self._compiled_critical = [re.compile(p, re.IGNORECASE) for p in self.CRITICAL_PATTERNS]
        self._compiled_severe = [re.compile(p, re.IGNORECASE) for p in self.SEVERE_PATTERNS]
        self._compiled_moderate = [re.compile(p, re.IGNORECASE) for p in self.MODERATE_PATTERNS]

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Assess crisis severity of user message.
        
        Args:
            text: User input message string.
            
        Returns:
            Dictionary containing is_crisis boolean, risk_level string, and matched triggers.
        """
        if not text:
            return {"is_crisis": False, "risk_level": "NONE", "triggers": []}

        matched_critical = [p.pattern for p in self._compiled_critical if p.search(text)]
        if matched_critical:
            return {
                "is_crisis": True,
                "risk_level": "CRITICAL",
                "triggers": matched_critical,
                "requires_immediate_escalation": True
            }

        matched_severe = [p.pattern for p in self._compiled_severe if p.search(text)]
        if matched_severe:
            return {
                "is_crisis": True,
                "risk_level": "SEVERE",
                "triggers": matched_severe,
                "requires_immediate_escalation": True
            }

        matched_mod = [p.pattern for p in self._compiled_moderate if p.search(text)]
        if matched_mod:
            return {
                "is_crisis": False,
                "risk_level": "MODERATE",
                "triggers": matched_mod,
                "requires_immediate_escalation": False
            }

        return {
            "is_crisis": False,
            "risk_level": "LOW",
            "triggers": [],
            "requires_immediate_escalation": False
        }
