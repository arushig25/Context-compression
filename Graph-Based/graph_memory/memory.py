from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .compressor import build_compressed_context
from .context_extractor_gemini import GeminiExtractor, apply_patch, sanitize_patch
from .extractor import check_invalidation, extract_entities
from .graph import KnowledgeGraph
from .metrics import measure_compression


def build_baseline_context(raw_history: List[str], *, max_turns: Optional[int] = None) -> str:
    """
    Baseline (naive) context: stuff the raw transcript into the context window.
    """
    if max_turns is not None:
        raw_history = raw_history[-max_turns:]
    return "\n".join(raw_history)


@dataclass
class GraphMemoryModule:
    """
    Orchestrates:
      - extraction + invalidation on every user turn
      - compressed context generation
      - simple compression metrics
    """

    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    raw_history: List[str] = field(default_factory=list)
    gemini_extractor: Optional[GeminiExtractor] = None

    def ingest_turn(self, role: str, text: str, *, extract: bool = False) -> None:
        """
        Generic ingestion hook. By default, only user turns should extract.
        Set extract=True for tool/noise turns in evaluation harnesses.
        """
        if role.strip().lower() == "user":
            self.ingest_user_turn(text)
            return
        if role.strip().lower() == "assistant":
            self.ingest_assistant_turn(text)
        else:
            self.add_turn(role, text)
        if extract:
            # Don't apply invalidation on tool outputs by default; only extract entities.
            extract_entities(text, self.graph)

    def add_turn(self, role: str, text: str) -> None:
        role = role.strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        self.raw_history.append(f"{role.title()}: {text.strip()}")

    def ingest_user_turn(self, text: str) -> None:
        self.add_turn("user", text)
        check_invalidation(text, self.graph)
        extract_entities(text, self.graph)

    def ingest_user_turn_gemini(self, text: str, *, model: str = "gemini-2.5-flash") -> None:
        """
        LLM-based extraction for more general domains.
        Uses the current compressed memory + the new user message to produce a JSON patch.
        """
        self.add_turn("user", text)
        # Still run simple invalidation heuristics (cheap + deterministic).
        check_invalidation(text, self.graph)
        if self.gemini_extractor is None or self.gemini_extractor.model != model:
            self.gemini_extractor = GeminiExtractor(model=model)
        patch = self.gemini_extractor.extract_patch(memory=self.compressed_context(), user_text=text)
        patch = sanitize_patch(patch, user_text=text, graph=self.graph)
        apply_patch(self.graph, patch)
        # Deterministic backstop: ensure we still capture common entities when the LLM extractor misses them.
        extract_entities(text, self.graph)

    def ingest_assistant_turn(self, text: str) -> None:
        self.add_turn("assistant", text)

    def compressed_context(self) -> str:
        return build_compressed_context(self.graph, style="dense")

    def compressed_context_verbose(self) -> str:
        return build_compressed_context(self.graph, style="verbose")

    def baseline_context(self, *, max_turns: Optional[int] = None) -> str:
        return build_baseline_context(self.raw_history, max_turns=max_turns)

    def compression_report(self) -> Dict[str, Any]:
        ctx = self.compressed_context()
        return measure_compression(self.raw_history, ctx)
