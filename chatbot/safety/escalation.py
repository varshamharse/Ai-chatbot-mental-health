"""Escalation protocol providing verified crisis resources and hotlines."""

from typing import Dict, Any


CRISIS_HOTLINES = {
    "US_Canada_Suicide_Lifeline": "Call or text 988 (Available 24/7, free and confidential)",
    "Crisis_Text_Line": "Text 'HOME' to 741741 (Free, 24/7 crisis support via SMS)",
    "The_Trevor_Project_LGBTQ": "Call 1-866-488-7386 or text 'START' to 678-678",
    "National_Domestic_Violence_Hotline": "Call 1-800-799-SAFE (7233) or text 'START' to 88788",
    "UK_Samaritans": "Call 116 123 (Free 24/7 helpline)",
    "India_Tele_MANAS": "Call 14416 or 1800 891 4416 (Govt. 24/7 mental health helpline)",
    "International_Directory": "https://findahelpline.com/ or https://www.befrienders.org/"
}


class CrisisEscalator:
    """Generates immediate escalation guidance when severe distress or self-harm is detected."""

    @staticmethod
    def get_crisis_response(severity: str = "CRITICAL") -> str:
        """Construct an immediate compassionate intervention response with hotlines."""
        return (
            "I hear how much pain you are in right now, and I want you to be safe. "
            "Because your safety is the most important thing, please connect with someone who can help right away:\n\n"
            "🚨 Immediate Crisis Support Resources (Free, Confidential, 24/7):\n"
            "• US & Canada: Call or text 988 to reach the Suicide & Crisis Lifeline.\n"
            "• Crisis Text Line: Text HOME to 741741 to connect with a crisis counselor.\n"
            "• India: Call Tele-MANAS at 14416 or 1800-891-4416.\n"
            "• UK: Call 116 123 to reach Samaritans.\n"
            "• Worldwide Directory: Visit https://findahelpline.com to find immediate local support in your country.\n\n"
            "You do not have to carry this alone. Please reach out to one of these lifelines or a trusted person right now."
        )

    @staticmethod
    def get_all_hotlines() -> Dict[str, str]:
        """Return full directory of hotlines."""
        return CRISIS_HOTLINES
