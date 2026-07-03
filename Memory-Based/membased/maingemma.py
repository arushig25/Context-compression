from manager import MemoryManager
import numpy as np
import time
import json
import re
import sys
import io
import os
from google import genai
from transformers import AutoTokenizer, AutoModel
import torch 

# Fix Windows Unicode issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
client = genai.Client(api_key=API_KEY)

# -------------------------
# LLM (used for memory extraction inside MemoryManager)
# -------------------------
def local_llm(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemma-3-4b-it",
            contents=prompt
        )

        result = response.text.strip()
        #print("\n--- RAW MODEL OUTPUT ---\n", result)

        # Extract JSON safely
        match = re.search(r'\{[\s\S]*?\}', result)
        if match:
            json_str = match.group(0)
            try:
                json.loads(json_str)
                return json_str
            except:
                pass

    except Exception as e:
        print("Gemini error:", e)

    return '{"facts": [], "tasks": [], "constraints": []}'




# -------------------------
# Transformer-based Embedding
# -------------------------
class TransformerEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text: str) -> np.ndarray:
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the mean pooling of the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embeddings


# Initialize the embedder
embedder = TransformerEmbedder()

# Wrapper function for embedding
def transformer_embed(text: str):
    return embedder.embed(text)


MAX_TOKENS = 4096

def limit_context(text: str, max_tokens=4096) -> str:
    words = text.split()
    limited_text = ""

    for word in words:
        temp = limited_text + " " + word
        if len(temp) // 4 > max_tokens: 
            break
        limited_text = temp

    return limited_text.strip()
# -------------------------
# Initialize Memory System
# -------------------------
manager = MemoryManager(local_llm, transformer_embed)


# -------------------------
# FINAL ANSWER GENERATION (RAG)
# -------------------------
def generate_answer(query: str, context: dict) -> str:
    formatted_context = ""

    for k, v in context.items():
        if v:
            formatted_context += f"\n{k.upper()}:\n{v}\n"

    limited_context = limit_context(formatted_context, MAX_TOKENS)

    #print("\n--- FINAL CONTEXT SENT TO MODEL  ---\n")
    #print(limited_context)


    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the user.

If context is insufficient, answer using general knowledge but mention limitation.

---

CONTEXT:
{limited_context}

---

USER QUERY:
{query}

---

Give a clear, structured response:
"""

    try:
        response = client.models.generate_content(
            model="gemma-3-1b-it",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        return f"Error generating response: {e}"


# -------------------------
# CHATBOT LOOP
# -------------------------
print("\n==============================")
print("     MEMORY CHATBOT STARTED   ")
print("Type 'exit' to stop")
print("==============================\n")

from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def count_tokens(text: str) -> int:
   
    return len(tokenizer.encode(text, truncation=True))


# Initialize raw history
raw_history = ""

while True:
    user_input = input("User: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye")
        break

    # 1. Store + extract memory
    manager.process_input(user_input)

    memory = manager.store.get_all()

    #print(memory) 

    # 2. Retrieve relevant memory
    context = manager.get_context(user_input)
    #print("context:", context)
    '''
    for key, value in context.items():
        print(f"\n{key.upper()}:")
        print(value)
    '''
    answer = generate_answer(user_input, context)

    # Add user input and model response to raw history
    raw_history += f"User: {user_input}\nBot: {answer}\n"
    # Count tokens in raw history
    raw_history_tokens = count_tokens(raw_history)

    # 4. Process and store the response summary in short-term memory
    manager.process_and_store_response(answer)

    # Count tokens
    combined_memory = " ".join(
        [
            " ".join(map(str, memory[key]))  # Ensure all elements are strings
            for key in ["semantic", "episodic", "pattern", "short_term"]
            if key in memory and memory[key]  # Check if key exists and is not empty
        ]
    )
    #original_tokens = count_tokens(combined_memory)

    # Combine context into a single string for token counting
    combined_context = " ".join(
        [" ".join(context[key]) for key in ["semantic", "episodic", "pattern", "short_term"] if key in context and context[key]]
    )

    compressed_tokens = count_tokens(combined_context)
    limited_context = count_tokens(limit_context(combined_context, MAX_TOKENS))
    # Calculate compression ratio and token reduction
    compression_ratio = limited_context / raw_history_tokens if raw_history_tokens > 0 else 0
    token_reduction = (1 - compression_ratio) * 100 if raw_history_tokens > 0 else 0

    # Factual Retention or Recall
    # Search for facts across all context categories
    all_context_data = " ".join(
        [" ".join(context[key]) for key in ["semantic", "episodic", "pattern", "short_term"] if key in context and context[key]]
    )
    factual_retention = sum(1 for fact in all_context_data.split() if fact in answer) / len(all_context_data.split()) if all_context_data else 0
    # Coherence Over Long Turns
    # Placeholder: Implement a coherence scoring function
    coherence_score = 1.0  # Assume perfect coherence for now

    # Tool-Call Correctness
    # Placeholder: Log tool calls and compare with expected outputs
    tool_call_correctness = True  # Assume all tool calls are correct for now

    # Omission or Distortion Rate
    omission_rate = (1 - (compressed_tokens / raw_history_tokens)) * 100 if raw_history_tokens > 0 else 0

    # Display the results
   

    # Display the results
    print(f"Original Tokens: {raw_history_tokens}")
    print(f"Compressed Tokens: {compressed_tokens}")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Token Reduction: {token_reduction:.2f}%")
    print(f"Factual Retention: {factual_retention:.2f}")
    print(f"Coherence Score: {coherence_score:.2f}")
    print(f"Tool-Call Correctness: {tool_call_correctness}")
    print(f"Omission Rate: {omission_rate:.2f}%")

    # 4. Output
    print("\nBot:", answer)
    print("\n" + "-" * 50)