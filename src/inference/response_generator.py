"""Empathetic response generation module supporting mental health and coping strategies."""

import random
from typing import Dict, Any, List


class MentalHealthResponseGenerator:
    """Generates empathetic, supportive, evidence-based coping responses for users."""

    STRESS_COPING_EXERCISES = [
        (
            "🌬️ Quick Grounding Exercise — Box Breathing:\n"
            "1. Inhale slowly through your nose for 4 seconds.\n"
            "2. Gently hold your breath for 4 seconds.\n"
            "3. Exhale smoothly through your mouth for 4 seconds.\n"
            "4. Rest and hold empty for 4 seconds.\n"
            "Repeat this 3 to 4 times to help calm your nervous system."
        ),
        (
            "🌿 5-4-3-2-1 Sensory Grounding Technique:\n"
            "Look around your room right now and name:\n"
            "• 5 things you can see\n"
            "• 4 things you can physically touch or feel\n"
            "• 3 things you can hear\n"
            "• 2 things you can smell\n"
            "• 1 thing you are grateful for in this moment."
        ),
        (
            "💙 Self-Compassion Pause:\n"
            "Put one hand on your chest and take one slow, deep breath. "
            "Remind yourself: 'This is a moment of difficulty. Difficulty is part of being human. "
            "May I be kind to myself right now.'"
        )
    ]

    STRESS_RESPONSES = [
        "I hear how heavy things feel right now, and I want to acknowledge how much courage it takes to share this. It is completely understandable to feel overwhelmed when carrying so much.",
        "It sounds like you're experiencing a significant amount of stress and emotional strain. Please know that your feelings are valid, and you don't have to face this all at once.",
        "Thank you for reaching out and sharing what you're experiencing. Feeling stressed or pressured can be exhausting on both mind and body."
    ]

    NON_STRESS_RESPONSES = [
        "Thank you for sharing with me. How are things feeling for you overall today?",
        "I'm here to listen. Whether you want to talk through your thoughts or simply check in, take all the time you need.",
        "I'm glad you're taking a moment for yourself today. What's on your mind right now?"
    ]

    def generate_response(
        self,
        predicted_class: str,
        confidence: float,
        risk_level: str = "NONE",
        user_message: str = ""
    ) -> str:
        """Create an empathetic response based on predicted state and confidence.
        
        Args:
            predicted_class: "Stress" or "Non-Stress".
            confidence: Float confidence score (0.0 - 1.0).
            risk_level: Safety risk category.
            user_message: Original user text.
            
        Returns:
            Formatted compassionate response text.
        """
        if predicted_class == "Stress":
            intro = random.choice(self.STRESS_RESPONSES)
            exercise = random.choice(self.STRESS_COPING_EXERCISES)
            
            response = (
                f"{intro}\n\n"
                f"When stress rises, taking a minute to reset our body can help create space to breathe:\n\n"
                f"{exercise}\n\n"
                f"What part of this feels most overwhelming right now? I am here to listen whenever you're ready."
            )
            return response
        else:
            base = random.choice(self.NON_STRESS_RESPONSES)
            return f"{base}\n\nRemember to take things one step at a time today."
