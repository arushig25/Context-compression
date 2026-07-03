import numpy as np

class MemoryRetriever:
    def __init__(self, embed_fn):
        """
        embed_fn: function(text) -> vector
        """
        self.embed_fn = embed_fn

    def _episodic_list(self, memory: dict):
        episodic = memory.get("episodic", [])
        if isinstance(episodic, dict):
            episodic = episodic.get("events", [])
        return episodic or []

    def cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def rank_items(self, query, items):
        q_emb = self.embed_fn(query)

        scored = []
        for item in items:
            emb = self.embed_fn(item)
            score = self.cosine_sim(q_emb, emb)
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored]

    def retrieve(self, query, memory):
        context = {}

        # 1. Short-term → always include
        context["short_term"] = memory["short_term"]

        # 2. Semantic → rank facts
        semantic_items = memory["semantic"]["facts"] + memory["semantic"]["constraints"]
        top_semantic = self.rank_items(query, semantic_items)[:8]
        context["semantic"] = top_semantic

        # 3. Episodic → rank summaries
        episodic_entries = self._episodic_list(memory)
        episodic_summaries = [e["summary"] for e in episodic_entries if isinstance(e, dict) and e.get("summary")]

        top_episodic = self.rank_items(query, episodic_summaries)[:6]
        context["episodic"] = top_episodic

        # 4. Pattern → always include (small anyway)
        context["pattern"] = memory["pattern"]

        return context