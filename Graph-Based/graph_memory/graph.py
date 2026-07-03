from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


STATUS_ACTIVE = "active"
STATUS_INVALIDATED = "invalidated"
STATUS_SUPERSEDED = "superseded"


def _now_ts() -> float:
    return time.time()


def _norm(s: str) -> str:
    return " ".join(s.strip().split()).lower()


@dataclass
class Node:
    id: str
    type: str
    name: str
    status: str = STATUS_ACTIVE
    created_at: float = field(default_factory=_now_ts)
    updated_at: float = field(default_factory=_now_ts)
    data: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = _now_ts()


@dataclass(frozen=True)
class Edge:
    src: str
    rel: str
    dst: str


class KnowledgeGraph:
    """
    Minimal directed property graph.

    Nodes are uniquely addressable by id, but we also maintain a (type, name) index
    for typical "entity upsert" operations.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges_out: Dict[str, List[Edge]] = {}
        self.edges_in: Dict[str, List[Edge]] = {}
        self._index: Dict[Tuple[str, str], str] = {}
        self.warnings: List[str] = []

    # ---- node helpers ----
    def _new_id(self) -> str:
        return uuid.uuid4().hex

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def find_by_type_name(self, type_: str, name: str) -> Optional[Node]:
        nid = self._index.get((type_, _norm(name)))
        if not nid:
            return None
        return self.nodes.get(nid)

    def upsert(
        self,
        type_: str,
        name: str,
        *,
        status: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        unique: bool = True,
    ) -> Node:
        """
        Upsert by (type, name). If unique=False, always creates a new node.
        """
        key = (type_, _norm(name))
        if unique and key in self._index:
            node = self.nodes[self._index[key]]
            if status is not None:
                node.status = status
            if data:
                node.data.update(data)
            node.name = name.strip()
            node.touch()
            return node

        node = Node(id=self._new_id(), type=type_, name=name.strip())
        if status is not None:
            node.status = status
        if data:
            node.data.update(data)
        self.nodes[node.id] = node
        if unique:
            self._index[key] = node.id
        return node

    def query(
        self,
        type_: str,
        *,
        status: Optional[str] = None,
        name: Optional[str] = None,
    ) -> List[Node]:
        out: List[Node] = []
        for node in self.nodes.values():
            if node.type != type_:
                continue
            if status is not None and node.status != status:
                continue
            if name is not None and _norm(node.name) != _norm(name):
                continue
            out.append(node)
        # stable ordering for deterministic summaries
        out.sort(key=lambda n: (n.type, _norm(n.name), n.created_at))
        return out

    # ---- edges ----
    def add_edge(self, src_id: str, rel: str, dst_id: str) -> None:
        e = Edge(src=src_id, rel=rel, dst=dst_id)
        self.edges_out.setdefault(src_id, []).append(e)
        self.edges_in.setdefault(dst_id, []).append(e)

    def out_neighbors(self, src_id: str, rel: Optional[str] = None) -> Iterable[str]:
        for e in self.edges_out.get(src_id, []):
            if rel is None or e.rel == rel:
                yield e.dst

    # ---- invalidation ----
    def invalidate_subtree(self, *, entity_name: str, root_type: str = "city") -> int:
        """
        Marks a root node (by type+name) invalidated and cascades to descendants.

        Returns number of nodes invalidated.
        """
        root = self.find_by_type_name(root_type, entity_name)
        if not root:
            # fallback: try any node name match
            for n in self.nodes.values():
                if _norm(n.name) == _norm(entity_name):
                    root = n
                    break
        if not root:
            # If a user says "forget Rome" before Rome ever existed in the plan,
            # we still want to remember that Rome was rejected to avoid re-suggesting it.
            if root_type == "city":
                self.upsert("city", entity_name, status=STATUS_INVALIDATED)
                return 1
            return 0

        q = [root.id]
        seen = set()
        n_invalidated = 0
        while q:
            nid = q.pop()
            if nid in seen:
                continue
            seen.add(nid)
            node = self.nodes.get(nid)
            if not node:
                continue
            # Constraints are "never expire". Events/expenses are treated as factual history and
            # are preserved unless explicitly invalidated via a dedicated trigger.
            if node.type not in {"constraint", "event", "expense"} and node.status != STATUS_INVALIDATED:
                node.status = STATUS_INVALIDATED
                node.touch()
                n_invalidated += 1
            for child in self.out_neighbors(nid):
                q.append(child)
        return n_invalidated

    def invalidate_all(self, *, type_: str) -> int:
        n = 0
        for node in self.query(type_, status=STATUS_ACTIVE):
            node.status = STATUS_INVALIDATED
            node.touch()
            n += 1
        return n

    # ---- budget ----
    def set_budget_total(self, total: int) -> None:
        # Keep old budget nodes, but treat them as superseded.
        for n in self.query("constraint"):
            if n.data.get("kind") == "budget_total" and n.status == STATUS_ACTIVE:
                n.status = STATUS_SUPERSEDED
                n.touch()
        self.upsert(
            "constraint",
            name=f"budget_total:{total}",
            status=STATUS_ACTIVE,
            data={"kind": "budget_total", "total": int(total)},
            unique=False,  # allow multiple budget nodes (history)
        )

    def add_expense(
        self,
        *,
        amount: float,
        description: str,
        city: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> Node:
        node = self.upsert(
            "expense",
            name=description or f"expense:{amount}",
            data={"amount": float(amount), "description": description, "raw_text": raw_text},
            unique=False,
        )
        if city:
            city_node = self.upsert("city", city)
            self.add_edge(city_node.id, "has_expense", node.id)
            self.add_edge(node.id, "for_city", city_node.id)
        return node

    def compute_budget_state(self) -> Dict[str, Any]:
        totals = [
            n.data.get("total")
            for n in self.query("constraint")
            if n.data.get("kind") == "budget_total" and isinstance(n.data.get("total"), (int, float))
        ]
        total = int(totals[-1]) if totals else 0
        spent = 0.0
        for e in self.query("expense", status=STATUS_ACTIVE):
            amt = e.data.get("amount")
            if isinstance(amt, (int, float)):
                spent += float(amt)
        remaining = float(total) - float(spent)
        return {"total": total, "spent": spent, "remaining": remaining}

    # ---- contradictions / notes ----
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
