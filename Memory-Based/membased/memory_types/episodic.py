import time

class EpisodicStore:
    def __init__(self):
        self.events = []

    def update(self, user_input: str, summary: str):
        self.events.append({
            "input": user_input,
            "summary": summary,
            "timestamp": time.time()
        })

    def get(self):
        return self.events