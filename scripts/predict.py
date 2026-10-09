"""Interactive prediction and mental health chatbot CLI script.

Usage:
    python scripts/predict.py --text "I have been having panic attacks and cannot sleep."
    python scripts/predict.py --chat
"""

import os
import sys
import argparse

# Add repo root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.chatbot_engine import MentalHealthChatbotEngine
from src.utils.logger import get_logger

logger = get_logger("predict_cli")


def run_interactive_chat():
    """Run interactive terminal chatbot loop."""
    print("\n" + "="*70)
    print("🧠 AI MENTAL HEALTH SUPPORT CHATBOT (Interactive Session)")
    print("="*70)
    print("Type your message below. Type 'exit' or 'quit' to end the session.\n")

    engine = MentalHealthChatbotEngine()
    print("Bot: Hello. I am an AI mental health support assistant here to listen.\nHow are you feeling today?\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nBot: Take good care of yourself. Remember you are never alone. Goodbye!")
                break

            result = engine.process_message(user_input)

            print(f"\n[AI Assessment] Class: {result['predicted_class']} | Confidence: {result['confidence']:.2f} | Risk: {result['risk_level']}")
            print(f"\nBot: {result['response']}\n")
            print("-" * 70)

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Take care!")
            break


def main():
    parser = argparse.ArgumentParser(description="Mental Health Chatbot & Prediction CLI")
    parser.add_argument("--text", type=str, default=None, help="Single input text to classify.")
    parser.add_argument("--chat", action="store_true", help="Launch interactive chatbot conversation.")
    args = parser.parse_args()

    engine = MentalHealthChatbotEngine()

    if args.text:
        result = engine.process_message(args.text)
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Input:           {args.text}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence:      {result['confidence']:.4f}")
        print(f"Risk Level:      {result['risk_level']}")
        print(f"Is Crisis:       {result['is_crisis']}")
        print(f"\nBot Response:\n{result['response']}")
        print("="*60 + "\n")
    else:
        run_interactive_chat()


if __name__ == "__main__":
    main()
