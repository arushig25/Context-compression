import json

class MemoryExtractor:
    def __init__(self, llm_call):
        """
        llm_call: function(prompt: str) -> str
        """
        self.llm_call = llm_call

    def extract(self, user_input: str) -> dict:
        prompt = f"""
You are a strict JSON generator.

Return ONLY valid JSON.

Format EXACTLY like this:
{{
  "facts": ["fact1", "fact2"],
  "tasks": ["task1"],
  "constraints": ["constraint1"]
}}

Rules:
- Each item MUST be a STRING (not object)
- facts: Declarative statements or information about the user provides (e.g., "i am a student" convert to "user is a student"). Dont put tasks/actions here.
- tasks: Instructions, actions, questions the user wants you to perform/answer (e.g., "suggest an itinerary", convert "is Rome safe?" to "find safety of Rome")
- constraints: Preferences, restrictions, or limitations (e.g., "under $2500", "vegetarian suggestions only")
- No nested JSON
- No explanations or extra text
- Output must start with {{ and end with }}
- If input has no relevant items for a field, use an empty list []
- Extract only from the input; do not infer or add external knowledge
Input:
{user_input}
"""


        response = self.llm_call(prompt)

        try:
            data = json.loads(response)

            # FIX: ensure lists of strings
            def clean_list(lst):
                cleaned = []
                for item in lst:
                    if isinstance(item, str):
                        cleaned.append(item)
                    elif isinstance(item, dict):
                        # extract best possible string
                        cleaned.append(str(list(item.values())[0]))
                return cleaned

            return {
                "facts": clean_list(data.get("facts", [])),
                "tasks": clean_list(data.get("tasks", [])),
                "constraints": clean_list(data.get("constraints", []))
            }

        except:
            return {"facts": [], "tasks": [], "constraints": []}



