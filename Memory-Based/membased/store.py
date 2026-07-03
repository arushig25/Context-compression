import time
from schema import MemorySystem

def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())

class MemoryStore:
    MAX_SHORT_TERM = 7

    def __init__(self):
        self.memory = MemorySystem()

    def get_all(self) -> dict:
        return {
            "semantic": {
                "facts": list(self.memory.semantic.facts),
                "constraints": list(self.memory.semantic.constraints),
            },
            "episodic": list(self.memory.episodic.events),
            "pattern": list(self.memory.pattern.patterns),
            "short_term": list(self.memory.short_term.recent_turns),
        }

    def _build_event_summary(self, extracted: dict) -> str:
        parts = []
        for key in ["facts", "tasks", "constraints"]:
            items = extracted.get(key, [])
            if items:
                normalized_items = [normalize_text(i) for i in items if i]
                if normalized_items:
                    parts.append(f"{key}: " + "; ".join(normalized_items))
        return " | ".join(parts).strip()

    def _is_duplicate_event(self, summary: str) -> bool:
        if not summary:
            return False
        normalized_summary = normalize_text(summary)
        for event in self.memory.episodic.events:
            if normalize_text(event.get("summary", "")) == normalized_summary:
                return True
        return False

    def update(self, user_input: str, extracted: dict):
        normalized_input = normalize_text(user_input)
        if normalized_input:
            self._update_short_term(normalized_input)

        self._update_episodic(normalized_input, extracted)
        self._update_semantic(extracted)
        self._update_pattern(normalized_input, extracted)  # Add this line

    def _update_pattern(self, normalized_input: str, extracted: dict):
        # Analyze the input for tone, emotion, or specific instructions
        tones = ["angry", "happy", "sad", "neutral", "formal", "informal","excited","stressed","frustrated"]
        instructions = ["shorter responses", "detailed explanation", "summarize", "elaborate", "use examples", "step-by-step", "detailed","reasoning"]

        # Detect tone or emotion
        detected_tone = next((tone for tone in tones if tone in normalized_input), None)
        if detected_tone:
            tone_entry = f"Tone: {detected_tone}"
            if tone_entry not in self.memory.pattern.patterns:
                self.memory.pattern.patterns.append(tone_entry)

        # Detect specific instructions
        detected_instruction = next((instr for instr in instructions if instr in normalized_input), None)
        if detected_instruction:
            instruction_entry = f"Instruction: {detected_instruction}"
            if instruction_entry not in self.memory.pattern.patterns:
                self.memory.pattern.patterns.append(instruction_entry)
                
    def _update_short_term(self, normalized_input: str):
        if not self.memory.short_term.recent_turns or self.memory.short_term.recent_turns[-1] != normalized_input:
            self.memory.short_term.recent_turns.append(normalized_input)
        self.memory.short_term.recent_turns = self.memory.short_term.recent_turns[-self.MAX_SHORT_TERM :]

    def _update_episodic(self, normalized_input: str, extracted: dict):
        """
        Update episodic memory with situational or time-bound information.
        Only tasks and situational constraints are added to episodic memory.
        """
        tasks = extracted.get("tasks", [])
        constraints = extracted.get("constraints", [])

        # Build a summary for episodic memory
        summary_parts = []
        if tasks:
            summary_parts.append(f"tasks: {'; '.join(normalize_text(task) for task in tasks)}")

        summary = " | ".join(summary_parts).strip()

        if summary and not self._is_duplicate_event(summary):
            self.memory.episodic.events.append({
                "input": normalized_input,
                "summary": summary,
                "timestamp": time.time()
            })

    def _update_semantic(self, extracted: dict):
        """
        Update semantic memory with timeless, general information.
        Only facts and general constraints are added to semantic memory.
        """
        # Add facts to semantic memory
        for fact in extracted.get("facts", []):
            normalized_fact = normalize_text(fact)
            if normalized_fact and normalized_fact not in self.memory.semantic.facts:
                self.memory.semantic.facts.append(normalized_fact)

        # Add general constraints to semantic memory
        for constraint in extracted.get("constraints", []):
            normalized_constraint = normalize_text(constraint)
            if normalized_constraint and normalized_constraint not in self.memory.semantic.constraints:
                self.memory.semantic.constraints.append(normalized_constraint)