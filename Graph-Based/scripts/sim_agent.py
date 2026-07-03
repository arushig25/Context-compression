from __future__ import annotations

import re
from typing import Optional

from graph_memory import GraphMemoryModule


def _get_constraint_value(mem: GraphMemoryModule, kind: str) -> Optional[str]:
    # Take the most recent matching constraint value.
    vals = []
    for c in mem.graph.query("constraint"):
        if c.data.get("kind") == kind:
            vals.append(str(c.data.get("value") or "").strip())
    return vals[-1] if vals else None


def respond(user_text: str, mem: GraphMemoryModule) -> str:
    """
    Tiny deterministic "agent" used only for validating the memory/compression module.
    It reads state from the graph (which is what the compressed context represents).
    """
    t = user_text.lower()
    g = mem.graph

    # A: allergy preservation
    if ("dinner" in t or "restaurant" in t or "spots" in t) and ("tsukiji" in t or "tokyo" in t):
        allergy = _get_constraint_value(mem, "allergy")
        avoid = _get_constraint_value(mem, "dietary_avoid")
        if (allergy and "shellfish" in allergy.lower()) or (avoid and "shellfish" in avoid.lower()):
            return (
                "Heads up: you mentioned a severe shellfish allergy. Tsukiji is seafood-heavy, so I will filter out "
                "seafood/sushi-focused spots and suggest non-seafood options only."
            )
        return "I can suggest dinner spots there. Any dietary restrictions I should know?"

    # B: budget anchor
    if "hotel" in t and ("amalfi" in t or "amalfi coast" in t):
        b = g.compute_budget_state()
        rem = b["remaining"]
        return (
            f"Based on your current spend, you have about ${rem:.0f} remaining. "
            "I'll prioritize budget-friendly options on the Amalfi Coast and flag anything that seems too pricey."
        )

    # C: pivot summary (must not mention invalidated plan)
    if "summarize" in t and ("trip" in t or "plan" in t):
        cities = [c.name for c in g.query("city", status="active")]
        events = g.query("event", status="active")
        parts = []
        if cities:
            parts.append("Active trip: " + ", ".join(cities))
        if events:
            e = events[0]
            parts.append(f"Commitment: {e.data.get('kind','event')} in {e.data.get('city')} on {e.data.get('day')} at {e.data.get('time')}")
        return " | ".join(parts) if parts else "No active trip plan captured yet."

    # D: logistics puzzle (meeting constraint)
    if "train" in t and ("paris" in t and "amsterdam" in t or "paris to amsterdam" in t):
        events = g.query("event", status="active")
        meeting = None
        for e in events:
            if e.data.get("kind") == "meeting" and (e.data.get("city") or "").lower() == "paris":
                meeting = e
                break
        if meeting:
            day = meeting.data.get("day")
            time_s = meeting.data.get("time")
            return (
                f"You have a meeting in Paris on {day} at {time_s}, so take the train after that "
                "(Wednesday evening at the earliest) or Thursday morning for a smoother plan."
            )
        return "Without fixed commitments, a morning train is usually best. Do you have any meetings in Paris?"

    # E: contradiction detector
    if "book all" in t or "book everything" in t:
        max_act_s = _get_constraint_value(mem, "max_activities")
        trip_days_s = _get_constraint_value(mem, "trip_days")
        activities = g.query("activity", status="active")
        try:
            max_act = int(max_act_s) if max_act_s else None
        except ValueError:
            max_act = None
        try:
            trip_days = int(trip_days_s) if trip_days_s else None
        except ValueError:
            trip_days = None

        if max_act and trip_days and activities:
            allowed = max_act * trip_days
            if len(activities) > allowed:
                return (
                    f"That conflicts with your preference of max {max_act} activities/day: "
                    f"you have {len(activities)} activities across {trip_days} days (max {allowed}). "
                    "Want me to help prioritize?"
                )
        return "I can book them, but confirm your pace preferences and trip length first."

    # Default
    return "OK."


def make_noise(n: int, *, seed_text: str = "TOOL_RESULT:") -> list[str]:
    # Repeat a long-ish string to simulate tool output bloat.
    blob = (seed_text + " lorem ipsum " * 40).strip()
    return [blob for _ in range(n)]


def make_attraction_noise(names: list[str]) -> list[str]:
    return [f"TOOL_RESULT attraction: {n}" for n in names]

