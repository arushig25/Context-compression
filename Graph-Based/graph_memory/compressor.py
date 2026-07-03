from __future__ import annotations

from typing import Dict, List, Literal

from .graph import KnowledgeGraph, STATUS_ACTIVE, STATUS_INVALIDATED


Style = Literal["dense", "verbose"]


def _fmt_money(x: float) -> str:
    # Avoid long floats in summaries.
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def build_compressed_context(graph: KnowledgeGraph, *, style: Style = "dense") -> str:
    """
    style="dense" is what you'd typically feed to a small-context model.
    style="verbose" is human-friendly (section headers, banners, warnings).
    """
    constraints = graph.query("constraint")  # "never expire" nodes (may include superseded history)
    # Only include constraints with a structured kind; bare-name constraints are too ambiguous/noisy.
    active_constraints = [
        c
        for c in constraints
        if c.status == STATUS_ACTIVE
        and c.data.get("kind")
        and c.data.get("kind") != "budget_total"
    ]

    budget = graph.compute_budget_state()

    active_cities = graph.query("city", status=STATUS_ACTIVE)
    invalid_cities = graph.query("city", status=STATUS_INVALIDATED)
    active_places = graph.query("place", status=STATUS_ACTIVE)
    invalid_places = graph.query("place", status=STATUS_INVALIDATED)

    events = graph.query("event", status=STATUS_ACTIVE)

    activities = graph.query("activity", status=STATUS_ACTIVE)

    profiles = graph.query("profile", status=STATUS_ACTIVE)
    facts = graph.query("fact", status=STATUS_ACTIVE)
    people = graph.query("person", status=STATUS_ACTIVE)
    items = graph.query("item", status=STATUS_ACTIVE)

    # Generic nodes for non-travel domains (e.g., task/goal/entity/fact).
    known_types = {"constraint", "city", "place", "person", "item", "event", "expense", "activity", "profile", "fact"}
    misc_active = [
        n for n in graph.nodes.values() if n.status == STATUS_ACTIVE and n.type not in known_types
    ]
    misc_active.sort(key=lambda n: (n.type, n.name.lower(), n.created_at))

    if style == "verbose":
        return _build_verbose(
            active_constraints,
            budget,
            active_cities,
            active_places,
            people,
            items,
            events,
            invalid_cities,
            invalid_places,
            activities,
            profiles,
            facts,
            misc_active,
            graph.warnings,
        )
    return _build_dense(
        active_constraints,
        budget,
        active_cities,
        active_places,
        people,
        items,
        events,
        invalid_cities,
        invalid_places,
        activities,
        profiles,
        facts,
        misc_active,
    )


def _build_dense(
    active_constraints,
    budget,
    active_cities,
    active_places,
    people,
    items,
    events,
    invalid_cities,
    invalid_places,
    activities,
    profiles,
    facts,
    misc_active,
) -> str:
    parts: List[str] = []

    # User profile/identity facts (who the user IS).
    if profiles or facts:
        pvals = []
        for p in profiles[:6]:
            val = p.data.get("value") or p.name
            kind = p.data.get("kind")
            pvals.append(f"{kind}={val}" if kind else str(val))
        for f in facts[:6]:
            val = f.data.get("value") or f.name
            kind = f.data.get("kind")
            pvals.append(f"{kind}={val}" if kind else str(val))
        parts.append("USER PROFILE\n" + "\n".join(f"  - {v}" for v in pvals))

    if active_constraints:
        # Sort chronologically so "latest" constraints win.
        ordered = sorted(active_constraints, key=lambda n: n.created_at)

        # Preferences (multi-value) vs other constraints (latest-value).
        pref_seen = set()
        pref_lines: List[str] = []
        latest_by_kind: Dict[str, str] = {}

        for c in ordered:
            attrs = c.data or {}
            kind = str(attrs.get("kind") or "").strip().lower()

            # Preference detection: either explicit kind or structured preference:<category>:<value> name.
            is_pref = kind == "preference" or c.name.lower().startswith("preference:")

            if is_pref:
                # Prefer structured name, but fall back to attributes.
                category = attrs.get("category")
                value = attrs.get("value")
                name_parts = c.name.split(":")
                if len(name_parts) == 3 and name_parts[0].lower() == "preference":
                    category = category or name_parts[1]
                    value = value or name_parts[2]
                category = str(category or "general").strip()
                value = str(value or "").strip()
                if not value:
                    continue

                key = (category.lower(), value.lower())
                if key in pref_seen:
                    continue
                pref_seen.add(key)

                priority = str(attrs.get("priority") or "low").strip().lower()
                pref_lines.append(f"{category}: {value} (priority={priority})")
                continue

            # Non-preference constraints: represent as kind=value.
            if not kind:
                # Fall back to "name:value" convention used by the regex extractor.
                if ":" in c.name:
                    kind = c.name.split(":", 1)[0].strip().lower()
            raw_value = attrs.get("value")
            if raw_value is None and ":" in c.name:
                raw_value = c.name.split(":", 1)[1].strip()
            if raw_value is None:
                continue
            latest_by_kind[kind or "constraint"] = str(raw_value).strip()

        if pref_lines:
            # High priority first for easy scanning.
            pref_lines.sort(key=lambda x: "priority=high" not in x)
            parts.append("USER PREFERENCES\n" + "\n".join(f"  - {v}" for v in pref_lines))

        if latest_by_kind:
            # Deterministic ordering for tests and stable diffs.
            lines = [f"{k}={v}" for k, v in sorted(latest_by_kind.items())]
            parts.append("CONSTRAINTS\n" + "\n".join(f"  - {v}" for v in lines))

    total = budget["total"]
    spent = budget["spent"]
    remaining = budget["remaining"]
    if total or spent:
        parts.append(f"BUDGET total={total} spent={_fmt_money(spent)} rem={_fmt_money(remaining)}")

    if active_cities:
        cities = []
        for c in active_cities[:8]:
            nights = c.data.get("nights")
            if isinstance(nights, int):
                cities.append(f"{c.name}({nights}n)")
            else:
                cities.append(c.name)
        parts.append("CITIES " + ", ".join(cities))

    if active_places:
        parts.append("PLACES " + ", ".join(p.name for p in active_places[:8]))

    if people:
        # Build owner->preferences map from constraints.
        owner_prefs: Dict[str, List[str]] = {}
        for c in active_constraints:
            if (c.data or {}).get("kind") != "preference":
                continue
            owner = (c.data or {}).get("owner")
            if not owner:
                continue
            cat = (c.data or {}).get("category") or "general"
            val = (c.data or {}).get("value") or ""
            if val:
                owner_prefs.setdefault(str(owner), []).append(f"{cat}:{val}")

        lines = []
        for p in people[:8]:
            if p.name.lower() == "user":
                # Prefer the most recent user name fact if present.
                display = (p.data or {}).get("display_name") or "user"
            else:
                display = p.name
            occ = (p.data or {}).get("occupation") or (p.data or {}).get("role") or ""
            pref_s = ""
            prefs = owner_prefs.get(p.name) or owner_prefs.get(display)
            if prefs:
                pref_s = " likes=" + ",".join(sorted(set(prefs))[:6])
            if occ:
                lines.append(f"{display}({occ}){pref_s}")
            else:
                lines.append(f"{display}{pref_s}")
        if lines:
            parts.append("PEOPLE " + "; ".join(lines))

    if items:
        lines = []
        for it in items[:8]:
            st = (it.data or {}).get("status")
            props = (it.data or {}).get("properties")
            bits = []
            if st:
                bits.append(f"status={st}")
            if isinstance(props, list) and props:
                bits.append("props=" + ",".join(str(p) for p in props[:6]))
            suffix = (" " + " ".join(bits)) if bits else ""
            lines.append(f"{it.name}{suffix}")
        if lines:
            parts.append("ITEMS " + "; ".join(lines))

    if events:
        evs = []
        for e in events[:8]:
            kind = e.data.get("kind") or "event"
            city = e.data.get("city") or "?"
            day = e.data.get("day") or "?"
            time_s = e.data.get("time") or "?"
            evs.append(f"{kind}:{city}:{day}:{time_s}")
        parts.append("EVENTS " + "; ".join(evs))

    if invalid_cities:
        names = ", ".join(c.name for c in invalid_cities[:8])
        suffix = " ..." if len(invalid_cities) > 8 else ""
        parts.append(f"REJECTED {names}{suffix}")

    if invalid_places:
        names = ", ".join(p.name for p in invalid_places[:8])
        suffix = " ..." if len(invalid_places) > 8 else ""
        parts.append(f"REJECTED_PLACES {names}{suffix}")

    if activities:
        parts.append(f"ACTIVITIES count={len(activities)}")

    if misc_active:
        by_type: Dict[str, List[str]] = {}
        for n in misc_active[:12]:
            by_type.setdefault(n.type, []).append(n.name)
        chunks = []
        for t, names in sorted(by_type.items()):
            shown = ", ".join(names[:3])
            suffix = " ..." if len(names) > 3 else ""
            chunks.append(f"{t}={shown}{suffix}")
        parts.append("MISC " + "; ".join(chunks))

    out = "\n".join(parts).strip()
    # Never return an empty memory block; a blank context makes downstream agents unstable.
    return out or "CONSTRAINTS\n  - (none)"


def _build_verbose(
    active_constraints,
    budget,
    active_cities,
    active_places,
    people,
    items,
    events,
    invalid_cities,
    invalid_places,
    activities,
    profiles,
    facts,
    misc_active,
    warnings,
) -> str:
    lines: List[str] = []
    lines.append("=== ACTIVE MEMORY (graph-compressed) ===")

    # User profile section (who they are).
    if profiles or facts:
        lines.append("USER PROFILE:")
        for p in profiles:
            val = p.data.get("value") or p.name
            kind = p.data.get("kind") or "role"
            lines.append(f"  - {kind}: {val}")
        for f in facts:
            val = f.data.get("value") or f.name
            kind = f.data.get("kind") or "fact"
            lines.append(f"  - {kind}: {val}")

    # Preferences + other constraints in separate sections to keep the memory legible.
    pref_lines: List[str] = []
    latest_by_kind: Dict[str, str] = {}
    if active_constraints:
        ordered = sorted(active_constraints, key=lambda n: n.created_at)
        pref_seen = set()
        for c in ordered:
            attrs = c.data or {}
            kind = str(attrs.get("kind") or "").strip().lower()
            is_pref = kind == "preference" or c.name.lower().startswith("preference:")

            if is_pref:
                category = attrs.get("category")
                value = attrs.get("value")
                name_parts = c.name.split(":")
                if len(name_parts) == 3 and name_parts[0].lower() == "preference":
                    category = category or name_parts[1]
                    value = value or name_parts[2]
                category = str(category or "general").strip()
                value = str(value or "").strip()
                if not value:
                    continue
                key = (category.lower(), value.lower())
                if key in pref_seen:
                    continue
                pref_seen.add(key)
                priority = str(attrs.get("priority") or "low").strip().lower()
                pref_lines.append(f"{category}: {value} (priority={priority})")
                continue

            if not kind and ":" in c.name:
                kind = c.name.split(":", 1)[0].strip().lower()
            raw_value = attrs.get("value")
            if raw_value is None and ":" in c.name:
                raw_value = c.name.split(":", 1)[1].strip()
            if raw_value is None:
                continue
            latest_by_kind[kind or "constraint"] = str(raw_value).strip()

    lines.append("USER PREFERENCES:")
    if not pref_lines:
        lines.append("  - (none)")
    else:
        pref_lines.sort(key=lambda x: "priority=high" not in x)
        for s in pref_lines:
            lines.append(f"  - {s}")

    lines.append("CONSTRAINTS:")
    if not latest_by_kind:
        lines.append("  - (none)")
    else:
        for k, v in sorted(latest_by_kind.items()):
            lines.append(f"  - {k}={v}")

    total = budget["total"]
    spent = budget["spent"]
    remaining = budget["remaining"]
    if total or spent:
        lines.append(f"BUDGET: ${_fmt_money(spent)} spent of ${total} total. Remaining: ${_fmt_money(remaining)}")

    lines.append("ENTITIES:")
    if not active_cities and not active_places:
        lines.append("  - (none)")
    else:
        for c in active_cities:
            nights = c.data.get("nights")
            nights_s = f"{nights} nights" if isinstance(nights, int) else "? nights"
            lines.append(f"  - city {c.name}: {nights_s}")
        for p in active_places:
            lines.append(f"  - place {p.name}")
    if people:
        lines.append("PEOPLE:")
        for p in people[:10]:
            occ = (p.data or {}).get("occupation") or (p.data or {}).get("role")
            if p.name.lower() == "user":
                display = (p.data or {}).get("display_name") or "user"
            else:
                display = p.name
            if occ:
                lines.append(f"  - {display} ({occ})")
            else:
                lines.append(f"  - {display}")

    if items:
        lines.append("ITEMS:")
        for it in items[:10]:
            st = (it.data or {}).get("status")
            props = (it.data or {}).get("properties")
            bits = []
            if st:
                bits.append(f"status={st}")
            if isinstance(props, list) and props:
                bits.append("props=" + ",".join(str(p) for p in props[:6]))
            suffix = (" " + " ".join(bits)) if bits else ""
            lines.append(f"  - {it.name}{suffix}")

    lines.append("EVENTS:")
    if not events:
        lines.append("  - (none)")
    else:
        for e in events:
            kind = e.data.get("kind") or "event"
            city = e.data.get("city") or "?"
            day = e.data.get("day") or "?"
            time_s = e.data.get("time") or "?"
            lines.append(f"  - {kind} in {city} on {day} at {time_s}")

    if invalid_cities:
        names = ", ".join(c.name for c in invalid_cities[:6])
        suffix = " ..." if len(invalid_cities) > 6 else ""
        lines.append(f"INVALIDATED CITIES (do not re-suggest): {names}{suffix}")

    if invalid_places:
        names = ", ".join(p.name for p in invalid_places[:6])
        suffix = " ..." if len(invalid_places) > 6 else ""
        lines.append(f"INVALIDATED PLACES (do not re-suggest): {names}{suffix}")

    if activities:
        lines.append(f"ACTIVITIES CAPTURED: {len(activities)}")

    if misc_active:
        lines.append("MISC:")
        for n in misc_active[:10]:
            lines.append(f"  - {n.type}: {n.name}")

    if warnings:
        lines.append("WARNINGS:")
        for w in warnings[-3:]:
            lines.append(f"  - {w}")

    lines.append("=== END COMPRESSED CONTEXT ===")
    return "\n".join(lines)
