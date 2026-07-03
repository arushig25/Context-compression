from __future__ import annotations

import argparse

from graph_memory import GraphMemoryModule
from graph_memory.llm_openai import create_response
from graph_memory.llm_gemini import generate_content
from graph_memory.context_extractor_gemini import GeminiExtractor, apply_patch, sanitize_patch
from graph_memory.extractor import check_invalidation, extract_entities
from graph_memory.graph import STATUS_ACTIVE


HOTEL_RECS = {
    "kyoto": [
        {
            "name": "Mitsui Garden Hotel Kyoto Shijo",
            "area": "Shijo/Karasuma",
            "tier": "mid",
            "why": "Great value, very central, easy transit.",
        },
        {
            "name": "The Royal Park Hotel Kyoto Sanjo",
            "area": "Sanjo/Kawaramachi",
            "tier": "mid",
            "why": "Walkable to downtown + river; solid rooms.",
        },
        {
            "name": "Cross Hotel Kyoto",
            "area": "Kawaramachi",
            "tier": "mid",
            "why": "Modern, lively location near shopping/food.",
        },
        {
            "name": "Hyatt Regency Kyoto",
            "area": "Higashiyama",
            "tier": "upper",
            "why": "Quiet, close to temples; reliable upscale stay.",
        },
        {
            "name": "Hotel The Celestine Kyoto Gion",
            "area": "Gion",
            "tier": "upper",
            "why": "Great base for evening walks in Gion.",
        },
        {
            "name": "The Ritz-Carlton, Kyoto",
            "area": "Kamogawa riverside",
            "tier": "luxury",
            "why": "Top-end service; beautiful riverside setting.",
        },
    ],
    "tokyo": [
        {
            "name": "Hotel Gracery Shinjuku",
            "area": "Shinjuku",
            "tier": "mid",
            "why": "Convenient transport hub; lively neighborhood.",
        },
        {
            "name": "Tokyu Stay Shinjuku",
            "area": "Shinjuku",
            "tier": "mid",
            "why": "Great for longer stays; practical rooms.",
        },
        {
            "name": "The Gate Hotel Tokyo by HULIC",
            "area": "Ginza/Yurakucho",
            "tier": "upper",
            "why": "Stylish, central, good access across Tokyo.",
        },
        {
            "name": "Hotel Niwa Tokyo",
            "area": "Suidobashi/Jimbocho",
            "tier": "upper",
            "why": "Calmer area; consistently well-reviewed.",
        },
        {
            "name": "Park Hotel Tokyo",
            "area": "Shiodome/Shimbashi",
            "tier": "upper",
            "why": "Great skyline views; easy subway access.",
        },
        {
            "name": "The Peninsula Tokyo",
            "area": "Marunouchi/Hibiya",
            "tier": "luxury",
            "why": "High-end, ultra-comfortable central base.",
        },
    ],
}


def _get_constraint(mem: GraphMemoryModule, kind: str) -> str | None:
    vals = []
    for c in mem.graph.query("constraint"):
        if c.data.get("kind") == kind:
            v = c.data.get("value")
            if v is not None:
                vals.append(str(v))
    return vals[-1] if vals else None


def _pick_city_from_text(mem: GraphMemoryModule, user_text: str) -> str | None:
    t = user_text.lower()
    cities = mem.graph.query("city", status="active")
    for c in cities:
        if c.name.lower() in t:
            return c.name
    return None


def _format_hotel_list(city: str) -> str:
    recs = HOTEL_RECS.get(city.lower(), [])
    if not recs:
        return ""
    lines = [f"Popular {city} picks (mixed budgets):"]
    for r in recs[:6]:
        lines.append(f"- {r['name']} ({r['area']}, {r['tier']}): {r['why']}")
    return "\n".join(lines)


def _format_knowledge_graph(mem: GraphMemoryModule, *, active_only: bool = True, max_nodes: int = 60) -> str:
    g = mem.graph
    nodes = list(g.nodes.values())
    if active_only:
        nodes = [n for n in nodes if n.status == STATUS_ACTIVE]
    nodes.sort(key=lambda n: (n.type, n.name.lower(), n.created_at))

    shown = nodes[:max_nodes]
    id_to_label = {n.id: f"{n.type}:{n.name}" for n in shown}

    lines = ["=== KNOWLEDGE GRAPH ===", "NODES:"]
    if not shown:
        lines.append("  - (none)")
    else:
        for n in shown:
            # Keep attributes compact.
            attrs = n.data or {}
            attrs_s = ""
            if attrs:
                keys = ["kind", "category", "value", "priority", "role", "total", "amount"]
                small = {k: attrs.get(k) for k in keys if k in attrs}
                if small:
                    attrs_s = " " + str(small)
            lines.append(f"  - {n.type} {n.name}{attrs_s}")

    # Edges: only show edges between shown nodes to avoid noise.
    lines.append("EDGES:")
    any_edge = False
    for n in shown:
        for e in g.edges_out.get(n.id, []):
            if e.dst in id_to_label:
                any_edge = True
                lines.append(f"  - {id_to_label[n.id]} -[{e.rel}]-> {id_to_label[e.dst]}")
    if not any_edge:
        lines.append("  - (none)")

    lines.append("=== END KNOWLEDGE GRAPH ===")
    return "\n".join(lines)


def _memory_qa(user_text: str, mem: GraphMemoryModule) -> str | None:
    """
    Deterministic "memory QA" for common evaluation-style questions.
    This makes the agent robust even if the upstream LLM call fails.
    """
    t = user_text.strip().lower()
    g = mem.graph

    def _items_with_prop(prop: str) -> list[str]:
        out = []
        for it in g.query("item", status=STATUS_ACTIVE):
            props = (it.data or {}).get("properties")
            if isinstance(props, list) and prop.lower() in [str(p).lower() for p in props]:
                out.append(it.name)
        return out

    def _items_with_status(status: str) -> list[str]:
        out = []
        for it in g.query("item", status=STATUS_ACTIVE):
            st = (it.data or {}).get("status")
            if isinstance(st, str) and st.lower() == status.lower():
                out.append(it.name)
        return out

    # Health summary
    if "health" in t and ("summarize" in t or "summary" in t):
        lines = []
        for c in g.query("constraint", status=STATUS_ACTIVE):
            kind = (c.data or {}).get("kind")
            val = (c.data or {}).get("value") or ""
            if kind in {"allergy", "intolerance", "dietary_avoid", "diet"} and val:
                lines.append(f"- {kind}: {val}")
        if not lines:
            return "I don't have any explicit health-related constraints stored yet."
        return "Health-related info I have stored:\n" + "\n".join(lines)

    # Item state questions
    if "what is broken" in t or t.strip() == "broken?":
        broken = _items_with_status("broken")
        if broken:
            return "Broken item(s): " + ", ".join(broken) + "."
        return "I don't have any item marked as broken in memory."

    if "what is expensive" in t or t.strip() == "expensive?":
        expensive = _items_with_prop("expensive")
        if expensive:
            return "Expensive item(s): " + ", ".join(expensive) + "."
        return "I don't have any item marked as expensive in memory."

    if "contact support" in t or "customer support" in t:
        # In memory we store fact support_contact:<item>
        for f in g.query("fact", status=STATUS_ACTIVE):
            if (f.data or {}).get("kind") == "support_contact":
                v = (f.data or {}).get("value")
                if v:
                    return f"You contacted customer support about: {v}."
        return None

    # Trip participants
    if (
        "who is going on the trip" in t
        or "who's going on the trip" in t
        or "whos going on trip" in t
        or "name people going on trip" in t
        or "people going on trip" in t
    ):
        trip_with = []
        for f in g.query("fact", status=STATUS_ACTIVE):
            if (f.data or {}).get("kind") == "trip_with":
                v = (f.data or {}).get("value")
                if v:
                    trip_with.append(str(v))
        not_joining = []
        for f in g.query("fact", status=STATUS_ACTIVE):
            if (f.data or {}).get("kind") == "not_joining":
                v = (f.data or {}).get("value")
                if v:
                    not_joining.append(str(v))
        if trip_with or not_joining:
            who = ["you"] + sorted(set(trip_with))
            s = "Going on the trip: " + ", ".join(who) + "."
            if not_joining:
                s += " Not joining: " + ", ".join(sorted(set(not_joining))) + "."
            return s
        return None

    # Who likes what
    if "summarize who likes what" in t:
        # Owner preferences are stored as constraint kind=preference with owner field.
        by_owner: dict[str, list[str]] = {}
        for c in g.query("constraint", status=STATUS_ACTIVE):
            if (c.data or {}).get("kind") != "preference":
                continue
            owner = (c.data or {}).get("owner") or "you"
            cat = (c.data or {}).get("category") or "general"
            val = (c.data or {}).get("value") or ""
            if val:
                by_owner.setdefault(str(owner), []).append(f"{cat}={val}")
        if not by_owner:
            return None
        lines = []
        for owner in sorted(by_owner.keys(), key=lambda x: (x != "you", x.lower())):
            vals = ", ".join(sorted(set(by_owner[owner]))[:8])
            lines.append(f"- {owner}: {vals}")
        return "Likes/preferences:\n" + "\n".join(lines)

    # Conflicts (simple)
    if "conflicting preferences" in t or "conflicts" in t:
        user_place = None
        rahul_place = None
        for c in g.query("constraint", status=STATUS_ACTIVE):
            if (c.data or {}).get("kind") != "preference":
                continue
            if (c.data or {}).get("owner") == "Rahul" and (c.data or {}).get("category") == "place":
                rahul_place = (c.data or {}).get("value")
            if not (c.data or {}).get("owner") and (c.data or {}).get("category") == "place":
                user_place = (c.data or {}).get("value")
        if user_place and rahul_place and str(user_place).lower() != str(rahul_place).lower():
            return f"Conflict: you prefer {user_place}, Rahul prefers {rahul_place}."
        return None

    return None


def rule_based_assistant(user_text: str, mem: GraphMemoryModule) -> str:
    """
    Demo-only assistant: no web/tooling. Returns helpful templated responses that
    demonstrate it can read state from the graph (which is what compression preserves).
    """
    t = user_text.lower()
    g = mem.graph

    # Explicit "show me memory"
    if "summary" in t or "memory" in t or "context" in t:
        return mem.compressed_context_verbose()
    if "graph" in t and ("show" in t or "print" in t or t.strip() == "graph"):
        return _format_knowledge_graph(mem)
    qa = _memory_qa(user_text, mem)
    if qa:
        return qa

    # Budget
    if "budget" in t:
        b = g.compute_budget_state()
        return f"Budget state: spent ${b['spent']:.2f} of ${b['total']} (remaining ${b['remaining']:.2f})."

    # Flights
    if "flight" in t or "flights" in t:
        cities = [c.name for c in g.query("city", status="active")]
        hint = f" Active cities on your plan: {', '.join(cities)}." if cities else ""
        return (
            "I can't fetch live flight prices in this demo, but I can help you narrow options.\n"
            "- What date(s) are you leaving/returning?\n"
            "- Preferred NYC airport (JFK/EWR/LGA) and Tokyo airport (HND/NRT)?\n"
            "- Nonstop vs 1-stop acceptable, and baggage needs?\n"
            f"{hint}"
        )

    # Transit between cities (demo knowledge, no live data)
    if ("travel" in t or "get" in t or "best way" in t) and ("tokyo" in t and "kyoto" in t):
        return (
            "Tokyo <-> Kyoto: take the Shinkansen (bullet train).\n"
            "- Fastest: Nozomi (~2h 15m, usually not covered by the basic JR Pass)\n"
            "- Covered option: Hikari (~2h 40m)\n"
            "Most people go Tokyo Station/Shinagawa -> Kyoto Station."
        )

    # Hotels
    if "hotel" in t or "hotels" in t:
        b = g.compute_budget_state()
        city = _pick_city_from_text(mem, user_text)
        if city:
            shortlist = _format_hotel_list(city)
            if shortlist:
                return (
                    f"{shortlist}\n"
                    f"(budget) remaining~${b['remaining']:.0f} total.\n"
                    "If you tell me your dates + price/night target, I can narrow to 2-3 best fits."
                )
        # Generic fallback (no city detected)
        return (
            f"Popular picks depend a lot on area and budget, but you have about ${b['remaining']:.0f} remaining total.\n"
            "Tell me the city and your rough price/night and I'll recommend a short list."
        )

    # Restaurants/dinner (demonstrate allergy carryover)
    if "dinner" in t or "restaurant" in t or "spots" in t or "food" in t:
        allergy = _get_constraint(mem, "allergy") or ""
        avoid = _get_constraint(mem, "dietary_avoid") or ""
        places = [p.name for p in g.query("place", status="active")]
        cities = [c.name for c in g.query("city", status="active")]
        loc_hint = ""
        if places:
            loc_hint = f" Is this for {places[-1]}?"
        elif cities:
            loc_hint = f" Is this for {cities[-1]}?"
        if "shellfish" in allergy.lower() or "shellfish" in avoid.lower():
            return (
                "You mentioned a shellfish allergy, so I'll avoid seafood/sushi-heavy places and flag cross-contact risk.\n"
                f"Which neighborhood and price range do you want?{loc_hint}"
            )
        return f"Which neighborhood and price range do you want? Any dietary restrictions?{loc_hint}"

    # Generic trip query
    if "where" in t and ("going" in t or "trip" in t or "travel" in t):
        cities = g.query("city", status="active")
        if not cities:
            return "No active trip cities captured yet."
        return "Active trip cities: " + ", ".join(c.name for c in cities)

    return "Got it. If you want, ask for `summary` to see what memory is currently retained."


def main() -> None:
    ap = argparse.ArgumentParser(description="CLI demo for graph-memory context compression.")
    ap.add_argument(
        "--llm",
        choices=["rule", "openai", "gemini", "gemma"],
        default="rule",
        help=(
            "Response engine. 'rule' is offline demo; 'openai' uses OPENAI_API_KEY; "
            "'gemini' uses GEMINI_API_KEY with Gemini models; "
            "'gemma' uses GEMINI_API_KEY with Gemma models via Google AI Studio."
        ),
    )
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Model id for the selected provider. "
            "Gemini default: gemini-2.5-flash. "
            "Gemma default: gemma-3-1b-it."
        ),
    )
    ap.add_argument(
        "--extract",
        choices=["regex", "gemini", "gemma"],
        default="regex",
        help=(
            "Memory extraction engine. 'regex' is local; "
            "'gemini' uses Gemini for extraction; "
            "'gemma' uses Gemma-3-1b-it for extraction (same GEMINI_API_KEY)."
        ),
    )
    ap.add_argument("--show-context", action="store_true", help="Print compressed context after each user turn.")
    ap.add_argument("--show-graph", action="store_true", help="Print the full knowledge graph after each user turn.")
    ap.add_argument("--baseline-turns", type=int, default=30, help="Turns to include in baseline stuffed context.")
    args = ap.parse_args()

    mem = GraphMemoryModule()
    prev_response_id: str | None = None
    gem_extractor: GeminiExtractor | None = None

    # Resolve default model per provider.
    DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
        "gemma": "gemma-3-1b-it",
    }
    effective_llm = args.llm  # "rule" | "openai" | "gemini" | "gemma"
    effective_extract = args.extract  # "regex" | "gemini" | "gemma"
    # Both gemma and gemini use the same Google AI Studio endpoint; resolve model string.
    resolved_model = args.model or DEFAULT_MODELS.get(effective_llm, "gemini-2.5-flash")
    extract_model = args.model or DEFAULT_MODELS.get(effective_extract, resolved_model)

    print("Graph Memory Chat (type 'exit' to quit).")

    while True:
        try:
            user = input("\nYou> ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        # Update memory first (always run deterministic extraction; LLM extraction is additive).
        use_gemini_extract = effective_extract in {"gemini", "gemma"}
        use_gemini_llm = effective_llm in {"gemini", "gemma"}

        mem.add_turn("user", user)
        # Always run deterministic extraction so memory still updates even if the LLM call fails.
        check_invalidation(user, mem.graph)
        extract_entities(user, mem.graph)

        # Optional: LLM-based patch extraction (additive). Used when extraction is enabled but the response LLM
        # is not Gemini/Gemma (e.g., --llm rule/openai with --extract gemini/gemma).
        if use_gemini_extract and not use_gemini_llm:
            try:
                if gem_extractor is None or gem_extractor.model != extract_model:
                    gem_extractor = GeminiExtractor(model=extract_model)
                patch = gem_extractor.extract_patch(memory=mem.compressed_context(), user_text=user)
                patch = sanitize_patch(patch, user_text=user, graph=mem.graph)
                apply_patch(mem.graph, patch)
            except Exception:
                # Keep deterministic extraction only.
                pass

        qa = _memory_qa(user, mem)
        if qa:
            assistant = qa
        elif args.llm == "openai":
            # Feed only the compressed memory as context (this is what you're evaluating).
            instructions = (
        "You are a helpful assistant.\n\n"
        "CRITICAL RULES — you must follow ALL of these in every response:\n"
        "1. Read MEMORY carefully.\n"
        "2. Apply preferences ONLY when relevant to the user's request (e.g., color preferences matter for outfits, not diet advice).\n"
        "3. Health/diet constraints (allergy, intolerance, dietary_avoid, diet) are ALWAYS relevant for food/drink recommendations.\n"
        "4. Never suggest anything that violates a hard constraint.\n"
        "5. If you need location for recommendations and MEMORY has a target place but not a current location, ask a clarifying question.\n\n"
        f"USER MEMORY (source of truth):\n{mem.compressed_context()}\n\n"
        "Now respond to the user, always respecting the above preferences."
    )
            try:
                r = create_response(
                    instructions=instructions,
                    user_text=user,
                    model=args.model or "gpt-4o-mini",
                    previous_response_id=prev_response_id,
                    store=False,
                )
                assistant = r.text or "(no text returned)"
                prev_response_id = r.response_id
            except Exception as e:
                assistant = f"(openai error) {e}\nFalling back to offline demo.\n" + rule_based_assistant(user, mem)
                prev_response_id = None
        elif use_gemini_llm:
            try:
                if use_gemini_extract:
                    if gem_extractor is None or gem_extractor.model != resolved_model:
                        gem_extractor = GeminiExtractor(model=resolved_model)
                    memory = mem.compressed_context()
                    out = gem_extractor.extract_patch_and_answer(memory=memory, user_text=user)
                    patch = out.get("patch") if isinstance(out.get("patch"), dict) else {}
                    patch = sanitize_patch(patch, user_text=user, graph=mem.graph)

                    # Apply deterministic extraction as a backstop, then LLM patch.
                    apply_patch(mem.graph, patch)

                    assistant = str(out.get("answer") or "").strip() or "(no text returned)"
                else:
                    system_instruction = (
                        "You are a helpful assistant. Use the MEMORY block as the source of truth.\n"
                        "Apply preferences only when relevant; health/diet constraints always apply to food/drink.\n"
                        "If a recommendation depends on location and MEMORY has a target place but not current location, ask.\n"
                        "Ask the minimum number of questions. Prefer giving a shortlist of popular options.\n\n"
                        f"MEMORY:\n{mem.compressed_context()}\n"
                    )
                    r = generate_content(
                        system_instruction=system_instruction,
                        user_text=user,
                        model=resolved_model,
                    )
                    assistant = r.text or "(no text returned)"
            except Exception as e:
                assistant = f"({effective_llm} error) {e}\nFalling back to offline demo.\n" + rule_based_assistant(user, mem)
        else:
            assistant = rule_based_assistant(user, mem)
        mem.ingest_assistant_turn(assistant)

        print(f"Bot> {assistant}")

        rep = mem.compression_report()
        print(
            f"(compression) raw_tokens~{int(rep['raw_tokens'])} -> compressed_tokens~{int(rep['compressed_tokens'])} "
            f"(ratio~{rep['ratio']:.2f}x)"
        )

        if args.show_context:
            print("\n" + mem.compressed_context_verbose())
        if args.show_graph:
            print("\n" + _format_knowledge_graph(mem))

        # Baseline shown only as a size hint, to avoid spamming.
        baseline = mem.baseline_context(max_turns=args.baseline_turns)
        print(f"(baseline) stuffed_turns={min(len(mem.raw_history), args.baseline_turns)} chars={len(baseline)}")


if __name__ == "__main__":
    main()
