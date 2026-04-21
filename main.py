# main.py

from dotenv import load_dotenv
load_dotenv()

from rag.retriever import HybridRetriever
from voice.speech_to_text import SpeechToText
from voice.text_to_speech import TextToSpeech


def print_banner():
    print("\n" + "="*50)
    print("       Local AI Assistant (Ollama + RAG)")
    print("="*50)
    print("Commands:")
    print("  'quit' or 'exit'  — Exit the chatbot")
    print("  'clear'           — Clear conversation memory")
    print("  'memory'          — Show conversation history")
    print("  'delete'          — Delete last conversation turn")
    print("  'voice'           — Switch to voice input mode")
    print("  'text'            — Switch to text input mode")
    print("  'speak on'        — Enable voice responses")
    print("  'speak off'       — Disable voice responses")
    print("  'stop'            — Stop current speech")
    print("  'help'            — Show this menu")
    print("="*50 + "\n")


def get_input_mode() -> str:
    print("How would you like to interact?")
    print("  1. Type your questions")
    print("  2. Speak your questions")
    choice = input("Enter 1 or 2: ").strip()
    return "voice" if choice == "2" else "text"


def get_output_mode() -> str:
    print("\nHow would you like to receive responses?")
    print("  1. Read responses")
    print("  2. Listen to responses")
    choice = input("Enter 1 or 2: ").strip()
    return "speak" if choice == "2" else "read"


def main():
    print_banner()

    print("Initializing assistant...")
    retriever = HybridRetriever()

    print("Initializing speech to text...")
    stt = SpeechToText()

    print("Initializing text to speech...")
    tts = TextToSpeech()

    print("\nAssistant ready!\n")

    input_mode = get_input_mode()
    output_mode = get_output_mode()

    print(f"\n✅ Input mode  : {input_mode.upper()}")
    print(f"✅ Output mode : {output_mode.upper()}")
    print("\nStart chatting! Type 'help' to see commands.\n")

    while True:
        try:
            # -------------------------
            # Get user input
            # -------------------------
            if input_mode == "voice":
                raw = input("You (press Enter to speak, or type command): ").strip()

                if raw.lower() in [
                    "quit", "exit", "clear", "memory", "delete",
                    "text", "voice", "speak on", "speak off", "stop", "help"
                ]:
                    user_input = raw
                else:
                    transcribed = stt.transcribe()
                    if not transcribed:
                        print("Could not transcribe. Please try again.\n")
                        continue
                    print(f"You (voice): {transcribed}")
                    user_input = transcribed

            else:
                user_input = input("You: ").strip()

            if not user_input:
                continue

            # -------------------------
            # Handle commands
            # -------------------------
            if user_input.lower() in ["quit", "exit"]:
                tts.stop()
                print("Goodbye!")
                if output_mode == "speak":
                    tts.speak("Goodbye!")
                break

            elif user_input.lower() == "clear":
                retriever.clear_memory()
                print("Memory cleared.\n")

            elif user_input.lower() == "memory":
                print("\n" + retriever.get_memory_summary() + "\n")

            elif user_input.lower() == "delete":
                retriever.delete_last_turn()
                print()

            elif user_input.lower() == "voice":
                input_mode = "voice"
                print("✅ Switched to voice input mode.\n")

            elif user_input.lower() == "text":
                input_mode = "text"
                print("✅ Switched to text input mode.\n")

            elif user_input.lower() == "speak on":
                output_mode = "speak"
                print("✅ Voice responses enabled.\n")

            elif user_input.lower() == "speak off":
                output_mode = "read"
                print("✅ Voice responses disabled.\n")

            elif user_input.lower() == "stop":
                tts.stop()
                print("✅ Speech stopped.\n")

            elif user_input.lower() == "help":
                print_banner()

            # -------------------------
            # Normal query
            # -------------------------
            else:
                print("\nAssistant: ", end="", flush=True)
                response = retriever.query_agent(user_input)
                print(response)
                print()

                if output_mode == "speak":
                    tts.speak_async(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            tts.stop()
            break

        except Exception as e:
            print(f"\nError: {e}\n")
            continue


if __name__ == "__main__":
    main()