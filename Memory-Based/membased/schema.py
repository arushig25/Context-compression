from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SemanticMemory:
    facts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

@dataclass
class EpisodicMemory:
    events: List[Dict] = field(default_factory=list)  
    # each event = {"input": ..., "summary": ..., "timestamp": ...}

@dataclass
class PatternMemory:
    patterns: List[str] = field(default_factory=list)

@dataclass
class ShortTermMemory:
    recent_turns: List[str] = field(default_factory=list)

@dataclass
class MemorySystem:
    semantic: SemanticMemory = field(default_factory=SemanticMemory)
    episodic: EpisodicMemory = field(default_factory=EpisodicMemory)
    pattern: PatternMemory = field(default_factory=PatternMemory)
    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)