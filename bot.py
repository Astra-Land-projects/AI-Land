"""
Simple Console Chatbot using the Claude API
Requires: pip install anthropic
Requires an Anthropic API key: https://console.anthropic.com/settings/keys

Set your API key as an environment variable before running:
  Windows (PowerShell):  $env:ANTHROPIC_API_KEY="your-key-here"
  macOS/Linux:           export ANTHROPIC_API_KEY="your-key-here"
"""

import os
from anthropic import Anthropic

MODEL = "claude-sonnet-5"  # fast and capable, good default for a chatbot
MAX_TOKENS = 1024


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Get a key at https://console.anthropic.com/settings/keys")
        print("Then set it in your terminal before running this script, e.g.:")
        print('  export ANTHROPIC_API_KEY="your-key-here"   (macOS/Linux)')
        print('  $env:ANTHROPIC_API_KEY="your-key-here"     (Windows PowerShell)')
        exit(1)
    return Anthropic(api_key=api_key)


def chat():
    client = get_client()

    # Conversation history - keeps context across turns
    messages = []

    print("=" * 50)
    print("Simple Claude Chatbot")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'clear' to reset the conversation history.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye! 👋")
            break

        if user_input.lower() == "clear":
            messages = []
            print("Conversation history cleared.")
            continue

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
            )
        except Exception as e:
            print(f"Error calling the API: {e}")
            messages.pop()  # remove the failed user message
            continue

        reply_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        print(f"\nClaude: {reply_text}")

        messages.append({"role": "assistant", "content": reply_text})


if __name__ == "__main__":
    chat()