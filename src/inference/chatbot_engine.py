"""End-to-end Mental Health Chatbot Engine combining ML prediction, crisis safety, and empathetic dialog."""

from typing import Dict, Any, Optional

from .predictor import MentalHealthPredictor
from .response_generator import MentalHealthResponseGenerator
from chatbot.safety.crisis_detection import CrisisDetector
from chatbot.safety.safety_rules import SafetyRuleEngine, MEDICAL_DISCLAIMER
from chatbot.safety.escalation import CrisisEscalator
from chatbot.conversation_manager import ConversationManager


class MentalHealthChatbotEngine:
    """Production and research chatbot engine orchestrating safety, prediction, and dialogue."""

    def __init__(
        self,
        model_path: str = "models/best_model/best_model.pkl",
        vectorizer_path: str = "models/vectorizers/tfidf_vectorizer.pkl"
    ):
        self.predictor = MentalHealthPredictor(model_path=model_path, vectorizer_path=vectorizer_path)
        self.response_generator = MentalHealthResponseGenerator()
        self.crisis_detector = CrisisDetector()
        self.safety_rules = SafetyRuleEngine()
        self.conversation_manager = ConversationManager()

    def process_message(self, user_message: str) -> Dict[str, Any]:
        """Process an incoming user message through the full chatbot pipeline.
        
        Workflow:
            1. Input Validation
            2. Crisis / Risk Analysis
            3. If Crisis -> Immediate Hotlines & Escalation
            4. ML Mental Health & Stress Prediction (with confidence score)
            5. Empathetic Dialogue Generation
            6. Safety Rule Verification
            7. Conversation History Tracking
            
        Args:
            user_message: Raw string message from user.
            
        Returns:
            Dictionary with response text, predicted class, confidence, and crisis flags.
        """
        user_message_clean = (user_message or "").strip()
        if not user_message_clean:
            return {
                "response": "Hello! I am here to support you. How are you feeling today?",
                "predicted_class": "Non-Stress",
                "confidence": 1.0,
                "is_crisis": False,
                "risk_level": "NONE",
                "disclaimer": MEDICAL_DISCLAIMER
            }

        # 1. Crisis Detection
        crisis_info = self.crisis_detector.analyze_text(user_message_clean)
        if crisis_info["requires_immediate_escalation"]:
            escalation_response = CrisisEscalator.get_crisis_response(crisis_info["risk_level"])
            self.conversation_manager.add_turn(
                user_message=user_message_clean,
                predicted_class="Crisis",
                confidence=1.0,
                risk_level=crisis_info["risk_level"],
                bot_response=escalation_response
            )
            return {
                "response": escalation_response,
                "predicted_class": "Crisis",
                "confidence": 1.0,
                "is_crisis": True,
                "risk_level": crisis_info["risk_level"],
                "disclaimer": MEDICAL_DISCLAIMER
            }

        # 2. ML Intent & Mental Health State Prediction
        pred = self.predictor.predict(user_message_clean)
        predicted_class = pred["predicted_class"]
        confidence = pred["confidence"]

        # 3. Empathetic Response Generation
        raw_response = self.response_generator.generate_response(
            predicted_class=predicted_class,
            confidence=confidence,
            risk_level=crisis_info["risk_level"],
            user_message=user_message_clean
        )

        # 4. Outgoing Safety Filtering
        sanitized_response = self.safety_rules.validate_outgoing_response(raw_response)

        # 5. Record Turn
        self.conversation_manager.add_turn(
            user_message=user_message_clean,
            predicted_class=predicted_class,
            confidence=confidence,
            risk_level=crisis_info["risk_level"],
            bot_response=sanitized_response
        )

        return {
            "response": sanitized_response,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": pred.get("probabilities", {}),
            "is_crisis": False,
            "risk_level": crisis_info["risk_level"],
            "latency_ms": pred.get("latency_ms", 0.0),
            "disclaimer": MEDICAL_DISCLAIMER
        }
