import os
import sys
import io
from dotenv import load_dotenv
from google import genai
from transformers import AutoTokenizer

# Fix Windows Unicode issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# -------------------------
# LOAD API KEY
# -------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=API_KEY)

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "gemma-3-1b-it"
MAX_TOKENS = 4096   

def count_tokens(text: str) -> int:
    return len(text) // 4   # simple approximation

chat_history = []



# -------------------------
# BUILD CONTEXT WITH TOKEN LIMIT
# -------------------------
MAX_TOKENS = 4096

def build_context(chat_history, max_tokens=4096):
    context = ""

    for turn in reversed(chat_history):
        new_piece = f"{turn['role'].upper()}: {turn['text']}\n"
        temp_context = new_piece + context

        if count_tokens(temp_context) > max_tokens:
            break

        context = temp_context

    return context
# -------------------------
# GENERATE RESPONSE
# -------------------------
def generate_response(user_input: str) -> str:
    context = build_context(chat_history, MAX_TOKENS)

    prompt = f"""


---

CONVERSATION:
{context}

USER: {user_input}

ASSISTANT:
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text.strip(), context

    except Exception as e:
        return f"Error: {e}", context

# -------------------------
# CHAT LOOP
# -------------------------
print("\n==============================")
print("   TOKEN-LIMITED GEMMA CHATBOT")
print(f"   Max Context: {MAX_TOKENS} tokens")
print("Type 'exit' to stop")
print("==============================\n")

while True:
    user_input = input("User: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break

    # Store user input
    chat_history.append({"role": "user", "text": user_input})

    # Generate response
    answer, used_context = generate_response(user_input)

    # Store bot response
    chat_history.append({"role": "assistant", "text": answer})

    # Debug info
    context_tokens = count_tokens(used_context)
    total_history_tokens = count_tokens(
        " ".join([turn["text"] for turn in chat_history])
    )

    print(f"\n[Context Tokens Used: {context_tokens}/{MAX_TOKENS}]")
    print(f"[Total History Tokens: {total_history_tokens}]")

    print("\nBot:", answer)
    print("\n" + "-" * 50)