import asyncio
import warnings
from typing import Any, Dict, Optional

from rasa.core.agent import Agent

warnings.filterwarnings("ignore")


# A placeholder for your Whisper function (remains the same)
def get_text_from_voice() -> str:
    """
    This function represents your Whisper model.
    It captures voice and returns the transcribed text.
    """
    # Simulate the output from Whisper
    # user_voice_input = "I want to do a card to card transaction"
    # print(f"🎤 Whisper transcribed text: '{user_voice_input}'")
    user_voice_input = input("Enter your command: ")
    return user_voice_input


# --- RASA NLU PROCESSOR CLASS (Updated for Rasa 3.x) ---


class NLUProcessor:
    """
    A class to load a Rasa model and use it for NLU-only tasks.
    """

    def __init__(self, agent: Agent):
        """
        Private constructor. Use the `create` classmethod to instantiate.
        """
        self.agent = agent
        if self.agent:
            print("✅ Rasa NLUProcessor initialized successfully.")

    @classmethod
    def create(cls, model_path: str) -> Optional["NLUProcessor"]:
        """
        Asynchronously loads the Rasa model and returns a class instance.

        Args:
            model_path: Path to the trained Rasa model (.tar.gz).

        Returns:
            An instance of NLUProcessor, or None if loading fails.
        """
        try:
            # Use Agent.load() to load the model. This is the correct method in Rasa 3.x.
            agent = Agent.load(model_path=model_path)
            return cls(agent)
        except Exception as e:
            print(f"❌ Error loading Rasa model: {e}")
            return None

    async def classify_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Uses the loaded agent to classify the intent of the given text.

        Args:
            text: The user input text.

        Returns:
            The classification result dictionary from Rasa, or None.
        """
        if not self.agent:
            print("Agent not available. Cannot classify intent.")
            return None

        # agent.parse_message() returns a list containing one dictionary.
        result = await self.agent.parse_message(text)
        return result or None

    async def process_command(self, text: str, confidence_threshold: float = 0.80):
        """
        Processes a text command, classifies it, and prints the determined action.

        Args:
            text: The user input text.
            confidence_threshold: The minimum confidence to consider an intent valid.
        """
        classification_result = await self.classify_intent(text)

        if classification_result:
            intent = classification_result.get("intent", {})
            intent_name = intent.get("name")
            confidence = intent.get("confidence")

            print("\n--- Rasa NLU Analysis ---")
            print(f"Intent: {intent_name}")
            print(f"Confidence: {confidence:.2f}")

            # Your application logic based on the intent
            if confidence and confidence > confidence_threshold:
                if intent_name == "send_money":
                    print("\nAction: 🚀 Initiating 'send money' flow...")
                    print(f"🔍 Classification result: {classification_result}")
                    return {
                        "action": "send_money",
                        "message": "Initiating 'send money' flow...",
                        "entities": [
                            x for x in classification_result.get("entities", [])
                        ],
                    }
                elif intent_name == "top_up":
                    print("\nAction: 📱 Initiating 'mobile top-up' flow...")
                    return {
                        "action": "top_up",
                        "message": "Initiating 'mobile top-up' flow...",
                    }
                elif intent_name == "card_to_card_transfer":
                    print("\nAction: 💳 Initiating 'card to card transfer' flow...")
                    return {
                        "action": "card_to_card_transfer",
                        "message": "Initiating 'card to card transfer' flow...",
                    }
                else:
                    print(
                        f"\nAction: 🤔 Intent recognized, but no specific action is defined. {intent_name}"
                    )
                    return {
                        "action": "unknown",
                        "message": "Intent recognized, but no specific action is defined.",
                    }
            else:
                print(
                    f"\nAction: 🤷 Could not determine action. Confidence ({confidence:.2f}) is below threshold ({confidence_threshold})."
                )
                return {
                    "action": "unknown",
                    "message": "Could not determine action. Confidence is below threshold.",
                }


# --- MAIN EXECUTION LOGIC ---


async def get_processor() -> NLUProcessor | None:
    NLU_MODEL_PATH = "./models/nlu_model.tar.gz"
    processor = NLUProcessor.create(NLU_MODEL_PATH)
    print("✅ Rasa NLUProcessor initialized successfully.")
    return processor


async def process_command(
    command: str, processor: NLUProcessor | None
) -> dict[str, str] | None:
    """
    Main function to set up and run the NLU processor.
    """

    if processor:
        return await processor.process_command(command)


async def main():
    processor = await get_processor()
    command = get_text_from_voice()
    result = await process_command(command=command, processor=processor)
    print(f"🔍 Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
