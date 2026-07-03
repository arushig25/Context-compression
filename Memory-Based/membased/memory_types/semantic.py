class SemanticStore:
    def __init__(self):
        self.facts = set()
        self.constraints = set()

    def update(self, extracted: dict):
        self.facts.update(extracted.get("facts", []))
        self.constraints.update(extracted.get("constraints", []))

    def get(self):
        return {
            "facts": list(self.facts),
            "constraints": list(self.constraints)
        }