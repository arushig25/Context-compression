from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from graph_memory import GraphMemoryModule


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def _assert_contains(hay: str, needle: str, *, name: str) -> CheckResult:
    ok = needle.lower() in hay.lower()
    return CheckResult(name=name, passed=ok, details=("ok" if ok else f"missing {needle!r}"))


def run_turns(turns: List[str]) -> GraphMemoryModule:
    mem = GraphMemoryModule()
    for t in turns:
        mem.ingest_user_turn(t)
    return mem


def test_1_personal_evolution() -> List[CheckResult]:
    turns = [
        "I am a college student studying engineering.",
        "I like physics and math more than coding.",
        "I used to enjoy coding in C++ but it feels hard now.",
        "I am thinking of switching to data science.",
        "I started learning Python basics.",
        "I built a small calculator project.",
        "I struggled with data structures and algorithms.",
        "I feel more interested in AI now.",
        "My goal is to become an ML engineer.",
        "Summarize my journey so far.",
        "What are my current interests?",
        "What should I focus on next?",
    ]
    mem = run_turns(turns)
    ctx = mem.compressed_context()
    return [
        _assert_contains(ctx, "role=college student", name="T1.profile_college_student"),
        _assert_contains(ctx, "education=engineering", name="T1.education_engineering"),
        _assert_contains(ctx, "goal=ml engineer", name="T1.goal_ml_engineer"),
        _assert_contains(ctx, "USER PREFERENCES", name="T1.has_preferences_section"),
    ]


def test_2_health_noise_distillation() -> List[CheckResult]:
    turns = [
        "bro today was insane I woke up late missed class had coffee with friends talked about random stuff like movies memes cricket politics everything was chaotic and I forgot my assignment deadline but anyway I think I have lactose intolerance.",
        "later I went for a walk and it was nice.",
        "I also ate ice cream and my stomach felt weird again.",
        "I might stop drinking milk.",
        "Suggest me a drink.",
        "What should I avoid?",
        "Summarize my important health-related info.",
        "What pattern do you see in my diet?",
    ]
    mem = run_turns(turns)
    ctx = mem.compressed_context()
    return [
        _assert_contains(ctx, "intolerance=lactose intolerance", name="T2.keeps_lactose_intolerance"),
        _assert_contains(ctx, "dietary_avoid=dairy", name="T2.keeps_dairy_avoid"),
    ]


def test_3_multi_person_world_model() -> List[CheckResult]:
    turns = [
        "My name is Arjun and I am a student.",
        "My brother Rahul is working in finance.",
        "My friend Alex is a designer.",
        "I like football and pasta.",
        "Rahul likes chess and mountains.",
        "Alex likes basketball and sushi.",
        "I am planning a trip with Rahul.",
        "Alex is not joining us.",
        "I prefer beaches over mountains.",
        "Rahul prefers mountains.",
        "Summarize who likes what.",
        "Who is going on the trip?",
        "What are our conflicting preferences?",
    ]
    mem = run_turns(turns)
    ctx = mem.compressed_context()
    return [
        _assert_contains(ctx, "PEOPLE", name="T3.has_people_section"),
        _assert_contains(ctx, "Rahul(finance)", name="T3.rahul_finance"),
        _assert_contains(ctx, "likes=hobby:chess", name="T3.rahul_likes_chess"),
        _assert_contains(ctx, "Alex(designer)", name="T3.alex_designer"),
        _assert_contains(ctx, "not_joining=Alex", name="T3.alex_not_joining"),
    ]


def test_4_items_ambiguity() -> List[CheckResult]:
    turns = [
        "I bought a phone and a laptop last week.",
        "The phone is expensive and the laptop is new.",
        "The phone stopped working properly.",
        "I contacted customer support about it.",
        "They told me to wait 3 days.",
        "I am frustrated with it.",
        "I might return the phone.",
        "The laptop is fine.",
        "What is expensive?",
        "What is broken?",
        "What did I contact support for?",
        "Summarize my situation.",
        "What should I do next?",
    ]
    mem = run_turns(turns)
    ctx = mem.compressed_context()
    return [
        _assert_contains(ctx, "ITEMS", name="T4.has_items_section"),
        _assert_contains(ctx, "phone status=returning", name="T4.phone_returning"),
        _assert_contains(ctx, "props=expensive", name="T4.phone_expensive"),
        _assert_contains(ctx, "support_contact=phone", name="T4.support_contact_phone"),
        _assert_contains(ctx, "support_wait_days=3", name="T4.wait_days_3"),
    ]


def test_5_long_term_goal_shift() -> List[CheckResult]:
    turns = [
        "I want to learn programming from scratch.",
        "I started with Python basics.",
        "I learned loops, functions, and simple programs.",
        "I built a calculator and a to-do app.",
        "I struggled with DSA concepts.",
        "I started practicing LeetCode problems.",
        "I got stuck in recursion.",
        "I watched tutorials and improved a bit.",
        "Now I am interested in AI and ML.",
        "I want to join hackathons.",
        "I built a chatbot project.",
        "I feel confident in Python now.",
        "I want internship opportunities.",
        "Summarize my learning journey.",
        "What are my strengths and weaknesses?",
        "What should I focus on next?",
        "What kind of role suits me?",
    ]
    mem = run_turns(turns)
    ctx = mem.compressed_context()
    return [
        _assert_contains(ctx, "learning=python basics", name="T5.learning_python_basics"),
        _assert_contains(ctx, "project=chatbot", name="T5.project_chatbot"),
        _assert_contains(ctx, "difficulty=recursion", name="T5.difficulty_recursion"),
        _assert_contains(ctx, "USER PREFERENCES", name="T5.has_preferences_section"),
    ]


TESTS: List[Tuple[str, Callable[[], List[CheckResult]]]] = [
    ("T1 - Personal Evolution", test_1_personal_evolution),
    ("T2 - Health Distillation", test_2_health_noise_distillation),
    ("T3 - Multi-Person World Model", test_3_multi_person_world_model),
    ("T4 - Items + Ambiguity", test_4_items_ambiguity),
    ("T5 - Goal Shift + Planning", test_5_long_term_goal_shift),
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

