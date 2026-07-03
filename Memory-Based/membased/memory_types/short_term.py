class ShortTermStore:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.turns = []

    def update(self, user_input: str):
        self.turns.append(user_input)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get(self):
        return self.turns