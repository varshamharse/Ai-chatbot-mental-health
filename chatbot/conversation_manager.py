"""Conversation manager for multi-turn dialogue state and risk tracking."""

import time
from typing import List, Dict, Any, Optional


class ConversationManager:
    """Tracks session history, turn counts, and risk progression in chat interactions."""

    def __init__(self, session_id: str = "default_session", max_history: int = 10):
        self.session_id = session_id
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.cumulative_stress_turns = 0

    def add_turn(
        self,
        user_message: str,
        predicted_class: str,
        confidence: float,
        risk_level: str,
        bot_response: str
    ) -> None:
        """Record a single interaction turn."""
        turn = {
            "turn_index": len(self.history) + 1,
            "timestamp": time.time(),
            "user_message": user_message,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "risk_level": risk_level,
            "bot_response": bot_response
        }
        self.history.append(turn)

        if predicted_class in ["Stress", 1, "1"]:
            self.cumulative_stress_turns += 1

        # Prune old turns if exceeding max history length
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_recent_history(self, num_turns: int = 3) -> List[Dict[str, Any]]:
        """Return the most recent dialogue turns."""
        return self.history[-num_turns:]

    def get_summary(self) -> Dict[str, Any]:
        """Return session statistics."""
        return {
            "session_id": self.session_id,
            "total_turns": len(self.history),
            "stress_turns_count": self.cumulative_stress_turns,
            "last_risk_level": self.history[-1]["risk_level"] if self.history else "NONE"
        }

    def clear(self) -> None:
        """Reset conversation state."""
        self.history = []
        self.cumulative_stress_turns = 0
