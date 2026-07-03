class PatternStore:
    def __init__(self):
        self.patterns = set()

    def update(self, extracted: dict):
        tasks = extracted.get("tasks", [])

        # naive pattern learning (upgrade later)
        for task in tasks:
            if "modular" in task.lower():
                self.patterns.add("prefers modular design")
            if "step by step" in task.lower():
                self.patterns.add("prefers stepwise explanations")

    def get(self):
        return list(self.patterns)