from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from graph_memory import GraphMemoryModule

from .sim_agent import make_attraction_noise, make_noise, respond


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def _assert_contains(hay: str, needle: str, *, name: str) -> CheckResult:
    ok = needle.lower() in hay.lower()
    return CheckResult(name=name, passed=ok, details=("ok" if ok else f"missing {needle!r}"))


def _assert_not_contains(hay: str, needle: str, *, name: str) -> CheckResult:
    ok = needle.lower() not in hay.lower()
    return CheckResult(name=name, passed=ok, details=("ok" if ok else f"found forbidden {needle!r}"))


def run_conversation(turns: List[Tuple[str, str]], *, extract_tool: bool = False) -> GraphMemoryModule:
    mem = GraphMemoryModule()
    for role, text in turns:
        if role == "tool":
            mem.ingest_turn("tool", text, extract=extract_tool)
        elif role == "assistant":
            mem.ingest_assistant_turn(text)
        else:
            mem.ingest_user_turn(text)
    return mem


def test_a_forgotten_allergy() -> List[CheckResult]:
    turns: List[Tuple[str, str]] = []
    turns.append(
        (
            "user",
            "I want to plan a 5-day trip to Tokyo and Kyoto. Budget is $3,000 total. I'm severely allergic to shellfish.",
        )
    )
    for blob in make_noise(14, seed_text="TOOL_RESULT flights/hotels/transit"):
        turns.append(("tool", blob))
    turns.append(("user", "Find me the best dinner spots in Tsukiji area"))

    mem = run_conversation(turns)
    ctx = mem.compressed_context()
    bot = respond("Find me the best dinner spots in Tsukiji area", mem)

    return [
        _assert_contains(ctx, "allergy=shellfish", name="A.ctx_keeps_allergy"),
        _assert_contains(bot, "shellfish", name="A.bot_warns_about_allergy"),
        _assert_contains(bot, "filter", name="A.bot_filters_seafood"),
    ]


def test_b_budget_anchor() -> List[CheckResult]:
    turns: List[Tuple[str, str]] = []
    turns.append(("user", "Planning a trip to Italy. 7 days, max budget $2,500, I'm a solo traveler."))
    for blob in make_noise(6, seed_text="TOOL_RESULT itinerary research"):
        turns.append(("tool", blob))
    turns.append(("user", "Booked flights for $800."))
    turns.append(("user", "Booked hotels in Rome for $400."))
    turns.append(("user", "Booked hotels in Florence for $350."))
    for blob in make_noise(8, seed_text="TOOL_RESULT more options"):
        turns.append(("tool", blob))
    turns.append(("user", "Find me a hotel on the Amalfi Coast"))

    mem = run_conversation(turns)
    ctx = mem.compressed_context()
    bot = respond("Find me a hotel on the Amalfi Coast", mem)

    # Remaining should be about 2500 - (800+400+350) = 950
    return [
        _assert_contains(ctx, "total=2500", name="B.ctx_budget_total"),
        _assert_contains(ctx, "rem=950", name="B.ctx_budget_remaining_950"),
        _assert_contains(bot, "$950", name="B.bot_mentions_remaining"),
    ]


def test_c_pivot_invalidation() -> List[CheckResult]:
    turns: List[Tuple[str, str]] = []
    turns.append(("user", "Plan me a beach vacation in Bali for next month."))
    for blob in make_noise(5, seed_text="TOOL_RESULT bali resorts beaches"):
        turns.append(("tool", blob))
    turns.append(("user", "Actually, scratch Bali entirely. Let's do Switzerland instead — I want mountains, not beaches."))
    for blob in make_noise(6, seed_text="TOOL_RESULT switzerland mountains trains"):
        turns.append(("tool", blob))
    turns.append(("user", "Summarize my trip plan so far."))

    mem = run_conversation(turns)
    ctx = mem.compressed_context()
    bot = respond("Summarize my trip plan so far.", mem)

    return [
        _assert_not_contains(bot, "Bali", name="C.bot_summary_has_zero_bali"),
        _assert_contains(bot, "Switzerland", name="C.bot_summary_mentions_switzerland"),
        # Compressed context can keep rejected memory, but should not pollute active plan.
        _assert_contains(ctx, "CITIES Switzerland", name="C.ctx_active_city_switzerland"),
    ]


def test_d_logistics_puzzle() -> List[CheckResult]:
    turns: List[Tuple[str, str]] = []
    turns.append(
        (
            "user",
            "I'm planning 6 days: 3 in Paris, 3 in Amsterdam. I have a meeting in Paris on Wednesday at 2pm near the Eiffel Tower.",
        )
    )
    for blob in make_noise(12, seed_text="TOOL_RESULT paris/amsterdam bookings"):
        turns.append(("tool", blob))
    turns.append(("user", "When should I take the train from Paris to Amsterdam?"))

    mem = run_conversation(turns)
    bot = respond("When should I take the train from Paris to Amsterdam?", mem)
    ctx = mem.compressed_context()

    return [
        _assert_contains(ctx, "meeting:Paris:Wednesday:2pm", name="D.ctx_keeps_meeting"),
        _assert_contains(bot, "Wednesday", name="D.bot_mentions_wednesday"),
        _assert_contains(bot, "Thursday", name="D.bot_suggests_thursday"),
    ]


def test_e_contradiction_detector() -> List[CheckResult]:
    turns: List[Tuple[str, str]] = []
    turns.append(("user", "I want a relaxing trip. No packed schedules. Max 2 activities per day."))
    # Simulate tool-driven research that surfaces many attractions.
    names = [f"Attraction{i}" for i in range(1, 16)]
    for blob in make_attraction_noise(names):
        turns.append(("tool", blob))
    # User later reveals trip length and requests booking everything.
    turns.append(("user", "OK book all of these for my 3-day trip"))

    mem = run_conversation(turns, extract_tool=True)
    ctx = mem.compressed_context()
    bot = respond("OK book all of these for my 3-day trip", mem)

    return [
        _assert_contains(ctx, "max_activities=2", name="E.ctx_keeps_max_activities"),
        _assert_contains(ctx, "trip_days=3", name="E.ctx_keeps_trip_days"),
        _assert_contains(ctx, "ACTIVITIES count=15", name="E.ctx_keeps_activity_count"),
        _assert_contains(bot, "conflicts", name="E.bot_pushes_back_on_conflict"),
    ]


TESTS: List[Tuple[str, Callable[[], List[CheckResult]]]] = [
    ("A - Forgotten Allergy", test_a_forgotten_allergy),
    ("B - Budget Anchor", test_b_budget_anchor),
    ("C - Pivot Invalidation", test_c_pivot_invalidation),
    ("D - Logistics Puzzle", test_d_logistics_puzzle),
    ("E - Contradiction Detector", test_e_contradiction_detector),
]


def main() -> None:
    all_results: List[CheckResult] = []
    for suite_name, fn in TESTS:
        results = fn()
        all_results.extend(results)
        ok = all(r.passed for r in results)
        print(f"\n[{ 'PASS' if ok else 'FAIL' }] {suite_name}")
        for r in results:
            print(f"  - {r.name}: {'PASS' if r.passed else 'FAIL'} ({r.details})")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    print(f"\nOverall: {passed}/{total} checks passed.")


if __name__ == "__main__":
    main()

