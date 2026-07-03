from manager import MemoryManager
import numpy as np


from transformers import pipeline
import json

# load model (first time will download)
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # lightweight + good
    device=-1  # CPU, use 0 if GPU
)

import re

def local_llm(prompt: str) -> str:
    output = pipe(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]

    result = output[len(prompt):].strip()

    print("\n--- RAW MODEL OUTPUT ---\n", result)  # 👈 ADD THIS

    match = re.search(r'\{[\s\S]*?\}', result)

    if match:
        json_str = match.group(0)
        try:
            json.loads(json_str)
            return json_str
        except:
            pass

    return '{"facts": [], "tasks": [], "constraints": []}'

# -------------------------
# Dummy embedding (keep for now)
# -------------------------
def dummy_embed(text: str):
    return np.array([len(text), sum(ord(c) for c in text) % 1000], dtype=float)


# -------------------------
# Initialize
# -------------------------
manager = MemoryManager(local_llm, dummy_embed)

# -------------------------
# Simulate conversation
# -------------------------
inputs = [
    "I want you to help me plan a trip to japan. I do not eat nonveg.",
    "It should be under 3000 dollars",
    "Now I want you to suggest good places to visit and good restraunts."
]

print("\n--- Processing Inputs ---")
for inp in inputs:
    print(f"\nUser: {inp}")
    manager.process_input(inp)

# -------------------------
# Check full memory
# -------------------------
print("\n\n--- FULL MEMORY ---")
memory = manager.store.get_all()

for key, value in memory.items():
    print(f"\n{key.upper()}:")
    print(value)

# -------------------------
# Test retrieval
# -------------------------
query = "suggest itinerary for japan trip"
print("\n\n--- RETRIEVED CONTEXT ---")
context = manager.get_context(query)

for key, value in context.items():
    print(f"\n{key.upper()}:")
    print(value)