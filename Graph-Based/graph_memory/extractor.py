from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Match, Optional

from .graph import KnowledgeGraph


@dataclass(frozen=True)
class EntityRule:
    type: str
    pattern: re.Pattern[str]
    handler: Callable[[Match[str], KnowledgeGraph, str], None]


def _money_to_float(s: str) -> float:
    return float(s.replace(",", "").strip())


def _clean_city(s: str) -> str:
    cleaned = " ".join(s.strip().split()).strip(" .,!?:;")
    # If the user typed a place in all-lowercase, title-case it for nicer memory display.
    if cleaned and not any(ch.isupper() for ch in cleaned):
        cleaned = cleaned.title()
    return cleaned


def _clean_place(s: str) -> str:
    # Same behavior as city for now; separate helper so we can evolve place normalization safely.
    return _clean_city(s)


def _upsert_constraint(graph: KnowledgeGraph, kind: str, value: str) -> None:
    # Constraints should persist; we keep historical nodes, but only some are "active" (budget_total).
    graph.upsert(
        "constraint",
        name=f"{kind}:{value}",
        data={"kind": kind, "value": value},
        unique=False,
    )


def _upsert_fact(graph: KnowledgeGraph, kind: str, value: str) -> None:
    value = " ".join(value.strip().split()).strip(" .,!?:;")
    if not value:
        return
    graph.upsert(
        "fact",
        name=f"{kind}:{value}",
        data={"kind": kind, "value": value},
        unique=True,
    )


def _upsert_place(graph: KnowledgeGraph, name: str, *, role: str | None = None) -> None:
    name = _clean_place(name)
    if not name:
        return
    data = {"role": role} if role else None
    graph.upsert("place", name, data=data)


def _upsert_profile(graph: KnowledgeGraph, kind: str, value: str) -> None:
    value = " ".join(value.strip().split()).strip(" .,!?:;")
    if not value:
        return
    graph.upsert(
        "profile",
        name=f"{kind}:{value}",
        data={"kind": kind, "value": value},
        unique=True,
    )


def _ensure_user_person(graph: KnowledgeGraph) -> str:
    node = graph.upsert("person", "user", data={"kind": "person"}, unique=True)
    return node.id


def _upsert_person(graph: KnowledgeGraph, name: str, *, relation_to_user: str | None = None, **attrs) -> None:
    name = " ".join(name.strip().split()).strip(" .,!?:;")
    if not name:
        return
    node = graph.upsert("person", name, data=attrs or None, unique=True)
    if relation_to_user:
        user_id = _ensure_user_person(graph)
        graph.add_edge(user_id, relation_to_user, node.id)


def _add_person_preference(graph: KnowledgeGraph, owner: str, category: str, value: str, *, priority: str = "low") -> None:
    owner = owner.strip()
    category = category.strip().lower()
    value = " ".join(value.strip().split()).strip(" .,!?:;").lower()
    if not (owner and category and value):
        return
    graph.upsert(
        "constraint",
        name=f"preference:{owner}:{category}:{value}",
        data={"kind": "preference", "owner": owner, "category": category, "value": value, "priority": priority},
        unique=False,
    )


def _upsert_item(graph: KnowledgeGraph, name: str, *, status: str | None = None, prop: str | None = None) -> None:
    name = " ".join(name.strip().split()).strip(" .,!?:;").lower()
    if not name:
        return
    node = graph.upsert("item", name, unique=True)
    if status:
        node.data["status"] = status.strip().lower()
    if prop:
        props = node.data.get("properties")
        if not isinstance(props, list):
            props = []
        p = prop.strip().lower()
        if p and p not in props:
            props.append(p)
        node.data["properties"] = props
    node.touch()


def _infer_item_from_context(graph: KnowledgeGraph) -> Optional[str]:
    # Prefer a single broken item, otherwise fall back to any known item.
    broken = [i for i in graph.query("item") if (i.data or {}).get("status") == "broken"]
    if len(broken) == 1:
        return broken[0].name
    items = graph.query("item")
    if items:
        return items[-1].name
    return None


def _add_user_preference(
    graph: KnowledgeGraph, category: str, value: str, *, priority: str = "low"
) -> None:
    category = category.strip().lower() or "general"
    value = " ".join(value.strip().split()).strip(" .,!?:;").lower()
    if not value:
        return
    graph.upsert(
        "constraint",
        name=f"preference:{category}:{value}",
        data={"kind": "preference", "category": category, "value": value, "priority": priority},
        unique=False,
    )


def _infer_pref_category(value: str) -> str:
    v = value.strip().lower()
    if v in {"football", "cricket", "basketball", "chess"}:
        return "hobby"
    if v in {"pasta", "sushi", "pizza", "coffee", "tea", "ice cream", "icecream"}:
        return "food"
    if v in {"mountains", "beaches", "beach"}:
        return "place"
    return "general"


def _split_values(raw: str) -> list[str]:
    s = raw.strip()
    if not s:
        return []
    # Replace common joiners with commas.
    s = re.sub(r"\s+(and|&)\s+", ",", s, flags=re.IGNORECASE)
    # Remove trailing clauses like "more than X" (handled separately elsewhere).
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # Trim determiners.
    cleaned = []
    for p in parts:
        p = re.sub(r"^(a|an|the)\s+", "", p, flags=re.IGNORECASE).strip()
        if p:
            cleaned.append(p)
    return cleaned


def _extract_like_list(raw: str, graph: KnowledgeGraph) -> None:
    # Handle "A and B more than C"
    m = re.search(r"(.+?)\s+more\s+than\s+(.+)$", raw, flags=re.IGNORECASE)
    if m:
        left = _split_values(m.group(1))
        right = _split_values(m.group(2))
        for v in left:
            _add_user_preference(graph, _infer_pref_category(v), v, priority="high")
        for v in right:
            _add_user_preference(graph, _infer_pref_category(v), v, priority="low")
        return

    for v in _split_values(raw):
        _add_user_preference(graph, _infer_pref_category(v), v, priority="low")


def _extract_owner_like_list(owner: str, raw: str, graph: KnowledgeGraph) -> None:
    for v in _split_values(raw):
        _add_person_preference(graph, owner, _infer_pref_category(v), v, priority="low")


def _handle_two_cities(m: Match[str], graph: KnowledgeGraph, turn_text: str) -> None:
    graph.upsert("city", _clean_city(m.group(1)))
    graph.upsert("city", _clean_city(m.group(2)))


def build_rules() -> list[EntityRule]:
    rules: list[EntityRule] = []

    # ---- constraints ----
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\ballergic to\s+(.+?)(?:\s+and\b|[.,;\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "allergy", m.group(1).strip()),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bcan(?:not|'t)\s+eat\s+(.+?)(?:\s+and\b|[.,;\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "dietary_avoid", m.group(1).strip()),
        )
    )

    # Intolerances / dietary restrictions (health-critical)
    # Examples:
    # - "I have lactose intolerance"
    # - "I think I have lactose intolerance"
    # - "I'm lactose intolerant"
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(
                r"\bI\s+(?:think\s+I\s+)?have\s+([A-Za-z][A-Za-z ]{2,30}?)\s+intolerance\b",
                re.IGNORECASE,
            ),
            handler=lambda m, g, t: (
                _upsert_constraint(g, "intolerance", f"{m.group(1).strip().lower()} intolerance"),
                _upsert_constraint(g, "dietary_avoid", "dairy")
                if "lactose" in m.group(1).lower()
                else None,
            ),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(
                r"\bI(?:'m|\s+am)\s+([A-Za-z][A-Za-z ]{2,30}?)\s+intolerant\b",
                re.IGNORECASE,
            ),
            handler=lambda m, g, t: (
                _upsert_constraint(g, "intolerance", f"{m.group(1).strip().lower()} intolerant"),
                _upsert_constraint(g, "dietary_avoid", "dairy")
                if "lactose" in m.group(1).lower()
                else None,
            ),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\blactose\s+intoleran(?:t|ce)\b", re.IGNORECASE),
            handler=lambda m, g, t: (
                _upsert_constraint(g, "intolerance", "lactose intolerance"),
                _upsert_constraint(g, "dietary_avoid", "dairy"),
            ),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bmax\s+(\d+)\s+activities\b(?:\s+per\s+day)?", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "max_activities", m.group(1).strip()),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bno\s+packed\s+schedule\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "pace", "no_packed_schedule"),
        )
    )
    # In build_rules(), add after the existing weather rules:

    # General "I am X" identity preferences (vegan, vegetarian, introvert, etc.)
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bI(?:'m|\s+am)\s+(vegan|vegetarian|gluten.free|diabetic|lactose.intolerant)\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "diet", m.group(1).lower()),
        )
    )

    # "I hate/dislike/can't stand X" — general aversions, not just cold weather
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bI\s+(?:hate|dislike|can'?t\s+stand|don'?t\s+like|avoid)\s+([^.,\n]{3,40}?)(?:[.,\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "aversion", m.group(1).strip()),
        )
    )

    # "I love/prefer/enjoy X"
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bI\s+(?:love|prefer|enjoy|like|always\s+want)\s+([^.,\n]{3,40}?)(?:[.,\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "preference", m.group(1).strip()),
        )
    )

    # Profile / identity
    rules.append(
        EntityRule(
            type="profile",
            pattern=re.compile(r"\bI\s+am\s+a\s+college\s+student\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_profile(g, "role", "college student"),
        )
    )
    rules.append(
        EntityRule(
            type="profile",
            pattern=re.compile(r"\bI\s+am\s+(?:a|an)\s+student\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_profile(g, "role", "student"),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bstudying\s+([A-Za-z][A-Za-z ]{2,30}?)(?=\s*(?:[.,;!?]|$))", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "education", m.group(1).strip().lower()),
        )
    )

    # Interests/preferences list: "I like physics and math more than coding"
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bI\s+like\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _extract_like_list(m.group(1), g),
        )
    )

    # Goals/plans
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bmy\s+goal\s+is\s+to\s+become\s+(?:an?\s+)?(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "goal", m.group(1).strip().lower()),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+am\s+thinking\s+of\s+switching\s+to\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "plan", f"switch to {m.group(1).strip().lower()}"),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+started\s+learning\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "learning", m.group(1).strip().lower()),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+started\s+with\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "learning", m.group(1).strip().lower()),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+built\s+(?:a\s+)?(?:small\s+)?(.+?)\s+project\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "project", m.group(1).strip().lower()),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+struggled\s+with\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "difficulty", m.group(1).strip().lower()),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+got\s+stuck\s+in\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "difficulty", m.group(1).strip().lower()),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bmore\s+interested\s+in\s+(.+?)\s+now\b", re.IGNORECASE),
            handler=lambda m, g, t: _add_user_preference(g, "interest", m.group(1).strip().lower(), priority="high"),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\b(?:now\s+)?I\s+am\s+interested\s+in\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: [_add_user_preference(g, "interest", v, priority="high") for v in _split_values(m.group(1))],
        )
    )

    # Multi-person world model
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bMy\s+name\s+is\s+([A-Za-z][A-Za-z]+)\b", re.IGNORECASE),
            handler=lambda m, g, t: (
                _upsert_fact(g, "name", m.group(1).strip()),
                g.upsert("person", "user", data={"kind": "person", "display_name": m.group(1).strip()}, unique=True),
            ),
        )
    )
    rules.append(
        EntityRule(
            type="person",
            pattern=re.compile(r"\bMy\s+brother\s+([A-Za-z][A-Za-z]+)\s+is\s+working\s+in\s+([A-Za-z][A-Za-z ]{2,30})", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_person(
                g, m.group(1).strip(), relation_to_user="brother", occupation=m.group(2).strip().lower()
            ),
        )
    )
    rules.append(
        EntityRule(
            type="person",
            pattern=re.compile(r"\bMy\s+friend\s+([A-Za-z][A-Za-z]+)\s+is\s+(?:a|an)\s+([A-Za-z][A-Za-z ]{2,30})", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_person(
                g, m.group(1).strip(), relation_to_user="friend", occupation=m.group(2).strip().lower()
            ),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\b([A-Za-z][A-Za-z]+)\s+likes\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _extract_owner_like_list(m.group(1).strip(), m.group(2), g),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+am\s+planning\s+a\s+trip\s+with\s+([A-Za-z][A-Za-z]+)\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "trip_with", m.group(1).strip()),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\b([A-Za-z][A-Za-z]+)\s+is\s+not\s+joining\s+us\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "not_joining", m.group(1).strip()),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bI\s+prefer\s+(.+?)\s+over\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: (
                _add_user_preference(g, _infer_pref_category(m.group(1)), m.group(1).strip().lower(), priority="high"),
                _upsert_constraint(g, "aversion", m.group(2).strip().lower()),
            ),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\b([A-Za-z][A-Za-z]+)\s+prefers\s+(.+?)(?:[.!?\n]|$)", re.IGNORECASE),
            handler=lambda m, g, t: _add_person_preference(
                g,
                m.group(1).strip(),
                _infer_pref_category(m.group(2)),
                m.group(2).strip(),
                priority="low",
            ),
        )
    )

    # Items / ambiguity
    rules.append(
        EntityRule(
            type="item",
            pattern=re.compile(
                r"\bI\s+bought\s+(?:a|an)\s+([A-Za-z][A-Za-z0-9 ]+?)\s+and\s+(?:a|an)\s+([A-Za-z][A-Za-z0-9 ]+?)\s+last\s+week\b",
                re.IGNORECASE,
            ),
            handler=lambda m, g, t: (
                _upsert_item(g, m.group(1), status="owned"),
                _upsert_item(g, m.group(2), status="owned"),
                _upsert_fact(g, "purchase_time", "last week"),
            ),
        )
    )
    rules.append(
        EntityRule(
            type="item",
            pattern=re.compile(r"\bThe\s+(phone|laptop)\s+is\s+([A-Za-z]+)\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_item(g, m.group(1), prop=m.group(2)),
        )
    )
    rules.append(
        EntityRule(
            type="item",
            pattern=re.compile(r"\b(?:and\s+)?the\s+(phone|laptop)\s+is\s+([A-Za-z]+)\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_item(g, m.group(1), prop=m.group(2)),
        )
    )
    rules.append(
        EntityRule(
            type="item",
            pattern=re.compile(r"\bThe\s+(phone|laptop)\s+stopped\s+working\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_item(g, m.group(1), status="broken"),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+contacted\s+customer\s+support\b(?:\s+about\s+it|\s+about\s+this)?", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "support_contact", _infer_item_from_context(g) or "unknown"),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bThey\s+told\s+me\s+to\s+wait\s+(\d+)\s+days\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_fact(g, "support_wait_days", m.group(1).strip()),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(r"\bI\s+might\s+return\s+the\s+(phone|laptop)\b", re.IGNORECASE),
            handler=lambda m, g, t: (
                _upsert_fact(g, "return_plan", m.group(1).strip().lower()),
                _upsert_item(g, m.group(1), status="returning"),
            ),
        )
    )
    # Weather preference (common across many travel chats)
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile( r"\b(i\s+hate|i\s+can'?t\s+stand|i\s+dislike|i\s+don'?t\s+like)\s+cold(?:\s+weather|\s+places|\s+climates?)?\b",
    re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "weather_preference", "avoid_cold"),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\b(prefer|want)\s+warm\s+weather\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "weather_preference", "prefer_warm"),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            # Avoid greedy-dot eating digits (e.g. capturing only the last digit).
            pattern=re.compile(r"\bbudget\b\D{0,20}\$?\s*([\d,]+(?:\.\d+)?)\b", re.IGNORECASE),
            handler=lambda m, g, t: g.set_budget_total(int(_money_to_float(m.group(1)))),
        )
    )

    # Trip duration, used for contradiction checks like "max 2 activities/day" vs "book all for 3-day trip".
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\b(\d+)[- ]day\s+trip\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "trip_days", m.group(1).strip()),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\b(?:planning|plan(?:ning)?|for)\s+(\d+)\s+days\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "trip_days", m.group(1).strip()),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            # Handles "I'm planning 6 days: 3 in Paris, 3 in Amsterdam"
            pattern=re.compile(r"\b(\d+)\s+days\s*:\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "trip_days", m.group(1).strip()),
        )
    )
    # After the "can't eat" rule, add:
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bI(?:'m|\s+am)\s+vegan\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "diet", "vegan"),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\bI(?:'m|\s+am)\s+vegetarian\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "diet", "vegetarian"),
        )
    )
    rules.append(
        EntityRule(
            type="constraint",
            pattern=re.compile(r"\b(?:I\s+)?(?:follow|eat)\s+(?:a\s+)?vegan\s+diet\b", re.IGNORECASE),
            handler=lambda m, g, t: _upsert_constraint(g, "diet", "vegan"),
        )
    )
    

    # ---- cities / trip plan ----
    # Examples: "trip to Paris", "going to New York", "visit San Francisco, California"
    city_pat = (
        r"(?:trip to|visit|going to|travel to|plan(?:ning)?\s+(?:a\s+trip\s+)?to)\s+"
        r"([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*(?:,\s*[A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)?)"
    )
    rules.append(
        EntityRule(
            type="city",
            pattern=re.compile(city_pat, re.IGNORECASE),
            handler=lambda m, g, t: g.upsert("city", _clean_city(m.group(1))),
        )
    )

    # "trip to Tokyo and Kyoto"
    rules.append(
        EntityRule(
            type="city",
            pattern=re.compile(
                r"(?:trip to|visit|going to|travel to|plan(?:ning)?\s+(?:a\s+trip\s+)?to)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)\s+and\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)",
                re.IGNORECASE,
            ),
            handler=_handle_two_cities,
        )
    )

    # "vacation in Bali", "staying in Zurich"
    rules.append(
        EntityRule(
            type="city",
            pattern=re.compile(
                r"\b(?:vacation|holiday|stay(?:ing)?)\s+in\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)\b",
                re.IGNORECASE,
            ),
            handler=lambda m, g, t: g.upsert("city", _clean_city(m.group(1))),
        )
    )

    # Examples: "3 nights in Paris"
    rules.append(
        EntityRule(
            type="city",
            pattern=re.compile(r"\b(\d+)\s+nights?\s+in\s+(.+)", re.IGNORECASE),
            handler=lambda m, g, t: g.upsert(
                "city", _clean_city(m.group(2)), data={"nights": int(m.group(1))}
            ),
        )
    )

    # Examples: "hotels in Kyoto for 3 nights", "in Kyoto for 3 nights"
    rules.append(
        EntityRule(
            type="city",
            pattern=re.compile(
                r"\b(?:hotels?\s+)?in\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)\s+for\s+(\d+)\s+nights?\b",
                re.IGNORECASE,
            ),
            handler=lambda m, g, t: g.upsert(
                "city", _clean_city(m.group(1)), data={"nights": int(m.group(2))}
            ),
        )
    )

    # Examples: "3 in Paris, 3 in Amsterdam"
    rules.append(
        EntityRule(
            type="city",
            pattern=re.compile(
                r"\b(\d+)\s+in\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)\b", re.IGNORECASE
            ),
            handler=lambda m, g, t: g.upsert(
                "city", _clean_city(m.group(2)), data={"nights": int(m.group(1))}
            ),
        )
    )

    # ---- fixed commitments / events ----
    # Example: "meeting in Paris on Wednesday at 3pm"
    rules.append(
        EntityRule(
            type="event",
            pattern=re.compile(
                r"\bmeeting\s+(?:in|at)\s+(.+?)\s+on\s+(\w+)\s+at\s+([\d:]+(?:am|pm)?)\b",
                re.IGNORECASE,
            ),
            handler=_handle_meeting,
        )
    )

    # ---- expenses (for budget recompute) ----
    # Example: "booked hotel for $200", "spent 35 on taxi"
    rules.append(
        EntityRule(
            type="expense",
            pattern=re.compile(
                r"\b(?:book(?:ed|ing)?|spent|pay(?:ing|ed)?|cost(?:s)?|price(?:d)?)\b"
                r".{0,30}?\$?\s*([\d,]+(?:\.\d+)?)\b",
                re.IGNORECASE,
            ),
            handler=_handle_expense,
        )
    )

    # ---- activities/attractions (typically from tool results / research output) ----
    rules.append(
        EntityRule(
            type="activity",
            pattern=re.compile(r"\b(?:activity|attraction)\s*:\s*([^\n\r]+)", re.IGNORECASE),
            handler=_handle_activity,
        )
    )

    # ---- general goals / places (non-travel specific) ----
    # Example: "I want to study abroad specifically Greenland"
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(
                r"\b(?:i\s+want\s+to\s+)?(?:study|studying)\s+abroad\b"
                r"(?:\s+(?:in|to|at|specifically)\s+"
                r"([A-Za-z][A-Za-z]+(?:\s+(?!suggest\b|recommend\b|good\b|best\b|universit(?:y|ies)\b|college(?:s)?\b|please\b|help\b)[A-Za-z][A-Za-z]+){0,3})"
                r"(?=\s*(?:[.,;!?]|$|\band\b|\bsuggest\b|\brecommend\b|\buniversit(?:y|ies)\b|\bcollege(?:s)?\b)))?",
                re.IGNORECASE,
            ),
            handler=lambda m, g, t: (
                _upsert_fact(g, "goal", "study abroad"),
                _upsert_place(g, m.group(1), role="target") if m.group(1) else None,
                _upsert_fact(g, "target_place", _clean_place(m.group(1))) if m.group(1) else None,
            ),
        )
    )

    # Example: "I'm in Greenland", "I live in Reykjavik"
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(
                r"\bI(?:'m|\s+am)\s+in\s+([A-Za-z][A-Za-z]+(?:\s+(?!and\b|now\b|today\b|currently\b)[A-Za-z][A-Za-z]+){0,3})(?=\s*(?:[.,;!?]|$|\band\b))",
                re.IGNORECASE,
            ),
            handler=lambda m, g, t: (
                _upsert_fact(g, "location", _clean_place(m.group(1))),
                _upsert_place(g, m.group(1), role="current"),
            ),
        )
    )
    rules.append(
        EntityRule(
            type="fact",
            pattern=re.compile(
                r"\bI\s+live\s+in\s+([A-Za-z][A-Za-z]+(?:\s+(?!and\b|now\b|today\b|currently\b)[A-Za-z][A-Za-z]+){0,3})(?=\s*(?:[.,;!?]|$|\band\b))",
                re.IGNORECASE,
            ),
            handler=lambda m, g, t: (
                _upsert_fact(g, "location", _clean_place(m.group(1))),
                _upsert_place(g, m.group(1), role="current"),
            ),
        )
    )
    return rules


def _handle_meeting(m: Match[str], graph: KnowledgeGraph, turn_text: str) -> None:
    city = _clean_city(m.group(1))
    day = m.group(2).strip()
    time_s = m.group(3).strip()
    city_node = graph.upsert("city", city)
    event_node = graph.upsert(
        "event",
        name=f"meeting:{city}:{day}:{time_s}",
        data={"kind": "meeting", "city": city, "day": day, "time": time_s, "raw_text": turn_text},
        unique=False,
    )
    graph.add_edge(city_node.id, "has_event", event_node.id)
    graph.add_edge(event_node.id, "in_city", city_node.id)


def _infer_city_from_turn(graph: KnowledgeGraph, turn_text: str) -> Optional[str]:
    # Heuristic: if exactly one active city exists, attribute expense to it.
    active_cities = graph.query("city", status="active")
    if len(active_cities) == 1:
        return active_cities[0].name
    # Otherwise, try to mention-match any known city.
    for c in active_cities:
        if re.search(rf"\b{re.escape(c.name)}\b", turn_text, re.IGNORECASE):
            return c.name
    return None


def _handle_expense(m: Match[str], graph: KnowledgeGraph, turn_text: str) -> None:
    amount = _money_to_float(m.group(1))
    city = _infer_city_from_turn(graph, turn_text)
    # Keep a short description to avoid bloating the compressed context.
    desc = "expense"
    graph.add_expense(amount=amount, description=desc, city=city, raw_text=turn_text)


def _handle_activity(m: Match[str], graph: KnowledgeGraph, turn_text: str) -> None:
    name = m.group(1).strip().strip(" .,!?:;")
    if not name:
        return
    graph.upsert("activity", name=name, data={"raw_text": turn_text}, unique=False)


INVALIDATION_SCRATCH = re.compile(r"\bscratch\s+(.+?)\s+entirely\b", re.IGNORECASE)
INVALIDATION_FORGET = re.compile(r"\bforget(?:\s+about)?\s+(.+?)(?:[.!?]|$)", re.IGNORECASE)
INVALIDATION_CHANGE = re.compile(r"\bchange of plans\b\s*:?\s*(.+)$", re.IGNORECASE)

INSTEAD_TRIGGER = re.compile(
    r"\b(?:actually,\s*)?let'?s\s+do\s+(.+?)\s+instead\b", re.IGNORECASE
)


def extract_entities(turn_text: str, graph: KnowledgeGraph) -> None:
    rules = build_rules()
    for rule in rules:
        for m in rule.pattern.finditer(turn_text):
            try:
                rule.handler(m, graph, turn_text)
            except Exception as e:  # pragma: no cover
                graph.add_warning(f"Extractor rule failed ({rule.type}): {e}")


def check_invalidation(turn_text: str, graph: KnowledgeGraph) -> None:
    # Collect targets first to avoid double-applying patterns like:
    # "Change of plans: forget Rome" (matches both CHANGE and FORGET).
    targets: list[str] = []

    m = INVALIDATION_SCRATCH.search(turn_text)
    if m:
        targets.append(m.group(1))

    m = INVALIDATION_CHANGE.search(turn_text)
    if m:
        tail = m.group(1).strip()
        # Common: "change of plans: forget Rome"
        m2 = INVALIDATION_FORGET.search(tail)
        targets.append(m2.group(1) if m2 else tail)
    else:
        # Only apply top-level "forget X" if this wasn't a "change of plans" wrapper.
        m = INVALIDATION_FORGET.search(turn_text)
        if m:
            targets.append(m.group(1))

    # Apply deduped invalidations.
    seen = set()
    for raw in targets:
        target = raw.strip().strip(" .,!?:;")
        key = target.lower()
        if not target or key in seen:
            continue
        seen.add(key)
        graph.invalidate_subtree(entity_name=target, root_type="city")

    # "Let's do X instead" usually implies "discard current plan and switch to X".
    m2 = INSTEAD_TRIGGER.search(turn_text)
    if m2:
        new_city = _clean_city(m2.group(1))
        # Invalidate all existing city subgraphs, then set the new city active.
        for c in graph.query("city", status="active"):
            if c.name.lower() != new_city.lower():
                graph.invalidate_subtree(entity_name=c.name, root_type="city")
        graph.upsert("city", new_city, status="active")
