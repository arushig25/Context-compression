from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .graph import KnowledgeGraph, STATUS_ACTIVE, STATUS_INVALIDATED, STATUS_SUPERSEDED
from .llm_gemini import generate_content


Patch = Dict[str, Any]


PATCH_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "upserts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "name": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [STATUS_ACTIVE, STATUS_INVALIDATED, STATUS_SUPERSEDED],
                    },
                    "attributes": {"type": "object"},
                },
                "required": ["type", "name"],
                "additionalProperties": False,
            },
        },
        "invalidations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["type", "name"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src_type": {"type": "string"},
                    "src_name": {"type": "string"},
                    "rel": {"type": "string"},
                    "dst_type": {"type": "string"},
                    "dst_name": {"type": "string"},
                },
                "required": ["src_type", "src_name", "rel", "dst_type", "dst_name"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

PATCH_AND_ANSWER_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "patch": PATCH_JSON_SCHEMA,
    },
    "required": ["answer", "patch"],
    "additionalProperties": False,
}

_LOW_SIGNAL_FACT_VALUES = {
    "nice",
    "cool",
    "okay",
    "ok",
    "fine",
    "good",
    "great",
}
_STOPWORDS = {
    "what",
    "who",
    "whos",
    "who's",
    "where",
    "when",
    "why",
    "how",
    "it",
    "this",
    "that",
    "stuff",
    "things",
    "anything",
    "everything",
    "someone",
    "somebody",
    "somewhere",
    "low",
    "high",
}
_ALLOWED_FACT_KINDS = {
    "goal",
    "plan",
    "education",
    "learning",
    "project",
    "difficulty",
    "location",
    "target_place",
    "name",
    "trip_with",
    "not_joining",
    "support_contact",
    "support_wait_days",
    "purchase_time",
    "return_plan",
    "state",
}


def sanitize_patch(patch: Patch, *, user_text: str, graph: KnowledgeGraph) -> Patch:
    """
    Gemma extraction can hallucinate. This sanitizer keeps only upserts that are:
      - already present in the graph (re-stating is fine), OR
      - strongly supported by the current user_text (substring / token match).
    Invalidations/relations are kept but validated for required fields.
    """
    if not isinstance(patch, dict):
        return {}

    user_lc = (user_text or "").lower()

    def _graph_has(type_: str, name: str) -> bool:
        try:
            return graph.find_by_type_name(type_, name) is not None
        except Exception:
            return False

    def _supported(value: str) -> bool:
        v = (value or "").strip().lower()
        if not v:
            return False
        if v in _LOW_SIGNAL_FACT_VALUES:
            return False
        if v in _STOPWORDS:
            return False
        # Very short tokens are often noise ("what", "ok", "3"); avoid unless explicitly special-cased.
        if len(v) <= 3 and v not in {"ai", "ml", "dsa"}:
            return False
        # Simple evidence: value substring occurs in message.
        if v in user_lc:
            return True
        # Token support: any 5+ char token present.
        toks = [t for t in re.split(r"[^a-z0-9]+", v) if len(t) >= 5]
        return any(t in user_lc for t in toks)

    cleaned: Patch = {"upserts": [], "invalidations": [], "relations": [], "notes": []}

    # Invalidations
    for inv in patch.get("invalidations", []) or []:
        if not isinstance(inv, dict):
            continue
        t = (inv.get("type") or "").strip()
        n = (inv.get("name") or "").strip()
        if t and n:
            cleaned["invalidations"].append({"type": t, "name": n, "reason": (inv.get("reason") or "").strip()})

    # Upserts
    allowed_types = {
        "constraint",
        "profile",
        "fact",
        "person",
        "place",
        "city",
        "event",
        "expense",
        "activity",
        "item",
        "state",
    }
    for u in patch.get("upserts", []) or []:
        if not isinstance(u, dict):
            continue
        t = (u.get("type") or "").strip()
        name = (u.get("name") or "").strip()
        if not t or not name or t not in allowed_types:
            continue

        attrs = u.get("attributes") if isinstance(u.get("attributes"), dict) else {}
        kind = str(attrs.get("kind") or "").strip().lower()
        value = attrs.get("value")
        if value is None and ":" in name:
            value = name.split(":", 1)[1]
        value_s = str(value or "").strip()

        # Drop noisy "fact" objects that just restate preferences/people; preferences should be constraints.
        if t == "fact":
            # Infer kind if missing from name.
            inferred_kind = kind or (name.split(":", 1)[0].strip().lower() if ":" in name else "")
            if inferred_kind in {"preference", "person", "hobby", "diet", "food", "fact"}:
                continue
            # If kind is present, enforce a small allowlist to keep the memory stable.
            if inferred_kind and inferred_kind not in _ALLOWED_FACT_KINDS:
                continue

        # Allow re-stating existing nodes without evidence.
        if _graph_has(t, name):
            cleaned["upserts"].append({"type": t, "name": name, "status": u.get("status"), "attributes": attrs})
            continue

        # Evidence gating for new nodes.
        if t == "constraint":
            # For preference constraints, require the value to be supported by this turn.
            if kind == "preference" or name.lower().startswith("preference:"):
                # Parse value from structured name if needed.
                parts = name.split(":")
                if len(parts) >= 3 and parts[0].lower() == "preference":
                    v = parts[-1]
                else:
                    v = value_s
                if _supported(v):
                    cleaned["upserts"].append({"type": t, "name": name, "status": u.get("status"), "attributes": attrs})
                continue

            # Health/diet constraints are critical: accept if supported by this message.
            if kind in {"allergy", "intolerance", "dietary_avoid", "diet"}:
                if _supported(value_s) or any(k in user_lc for k in ["allerg", "intoler", "lactose", "dairy", "gluten"]):
                    cleaned["upserts"].append({"type": t, "name": name, "status": u.get("status"), "attributes": attrs})
                continue

            # Other constraints: require evidence.
            if _supported(value_s) or _supported(name):
                cleaned["upserts"].append({"type": t, "name": name, "status": u.get("status"), "attributes": attrs})
            continue

        # Facts/profiles/people/places/items: require evidence.
        if _supported(value_s) or _supported(name):
            cleaned["upserts"].append({"type": t, "name": name, "status": u.get("status"), "attributes": attrs})

    # Relations: keep only if names/types are present and at least one endpoint exists or is mentioned.
    for rel in patch.get("relations", []) or []:
        if not isinstance(rel, dict):
            continue
        st = (rel.get("src_type") or "").strip()
        sn = (rel.get("src_name") or "").strip()
        rt = (rel.get("rel") or "").strip()
        dt = (rel.get("dst_type") or "").strip()
        dn = (rel.get("dst_name") or "").strip()
        if not (st and sn and rt and dt and dn):
            continue
        if _graph_has(st, sn) or _graph_has(dt, dn) or _supported(sn) or _supported(dn):
            cleaned["relations"].append(
                {"src_type": st, "src_name": sn, "rel": rt, "dst_type": dt, "dst_name": dn}
            )

    # Notes
    notes = patch.get("notes", [])
    if isinstance(notes, list):
        cleaned["notes"] = [str(n) for n in notes[:6] if isinstance(n, (str, int, float))]

    return cleaned


# ── Shared prompt constants ────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a high-precision memory extraction engine for long conversations.

Return ONLY valid JSON (no markdown, no explanations).

FORMAT:
{"upserts": [...], "invalidations": [...], "relations": [...], "notes": [...]}

----------------------------------------
GOAL
----------------------------------------
Capture ONLY stable, high-signal information that will matter later:
- health/dietary constraints (allergies, intolerances, diets)  [VERY IMPORTANT]
- user identity/profile (student, profession, etc.)
- goals/plans (e.g., becoming an ML engineer, studying abroad)
- long-lived preferences/interests (likes/dislikes, travel style, etc.)
- named people + relationships + their preferences (brother Rahul, friend Alex)
- concrete entities/items and their states (phone broken, laptop new)

IGNORE low-signal chatter:
- generic adjectives ("nice", "cool") unless clearly important
- random topics mentioned in passing (movies/memes/etc.) unless the user says they like/dislike them

----------------------------------------
RULES
----------------------------------------
1) ONLY extract what is explicitly stated. Never infer.
2) MULTI-VALUE: if multiple values are stated (A and B), create multiple upserts.
3) NEVER overwrite earlier facts; add new nodes instead.
4) NORMALIZE:
   - remove filler: "as well", "also", "really", "very"
   - lowercase simple values (colors/foods/etc.)
5) PRIORITY:
   - health constraints and hard requirements => priority="high"
   - soft preferences/interests => priority="low"

----------------------------------------
SCHEMA CONVENTIONS
----------------------------------------

PREFERENCES (user or other people):
- type="constraint"
- name:
  - user: preference:<category>:<value>
  - other person: preference:<owner>:<category>:<value>
- attributes:
  {"kind":"preference","category":"<category>","value":"<value>","priority":"high|low","owner":"<owner_optional>"}

HEALTH / DIET:
- type="constraint"
- name examples:
  - allergy:shellfish
  - intolerance:lactose intolerance
  - dietary_avoid:dairy
  - diet:vegan
- attributes:
  {"kind":"allergy|intolerance|dietary_avoid|diet","value":"<value>","priority":"high"}

PROFILE (who someone is):
- type="profile" or type="person"

FACTS / GOALS:
- type="fact"
- name: <kind>:<value>   (e.g., goal:become_ml_engineer, education:engineering, item_status:phone_broken)
- attributes: {"kind":"<kind>","value":"<value>"}

ITEMS:
- type="item" with attributes {"status": "...", "properties": ["..."]} when present.

RELATIONS:
- Use relations for stable links (user->brother->Rahul, trip participants, etc.)

----------------------------------------
EXAMPLES
----------------------------------------

Input: "I think I have lactose intolerance."
Output:
{
  "upserts":[
    {"type":"constraint","name":"intolerance:lactose intolerance","attributes":{"kind":"intolerance","value":"lactose intolerance","priority":"high"}},
    {"type":"constraint","name":"dietary_avoid:dairy","attributes":{"kind":"dietary_avoid","value":"dairy","priority":"high"}}
  ],
  "invalidations":[], "relations":[], "notes":[]
}

Input: "My brother Rahul likes chess and mountains."
Output:
{
  "upserts":[
    {"type":"person","name":"Rahul","attributes":{"kind":"person","role":"brother"}},
    {"type":"constraint","name":"preference:Rahul:hobby:chess","attributes":{"kind":"preference","owner":"Rahul","category":"hobby","value":"chess","priority":"low"}},
    {"type":"constraint","name":"preference:Rahul:place:mountains","attributes":{"kind":"preference","owner":"Rahul","category":"place","value":"mountains","priority":"low"}}
  ],
  "invalidations":[], "relations":[
    {"src_type":"person","src_name":"user","rel":"brother","dst_type":"person","dst_name":"Rahul"}
  ],
  "notes":[]
}
"""

_ANSWER_SYSTEM = """\
You are a personalized assistant.

IMPORTANT RULES:

1. You MUST use MEMORY to guide your answer.
2. If user preferences exist, they OVERRIDE general suggestions.
3. If multiple preferences exist:
   - prioritize "high" priority
   - otherwise prefer first mentioned
4. NEVER ignore preferences when relevant.

5. If user asks for recommendation:
   - first suggest using preferred values
   - then optionally give alternatives

6. Location-aware recommendations:
   - If the user asks for restaurants, local services, universities in a place, weather/clothing, or any advice that depends on location,
     first check MEMORY for a current location (e.g., location=...) or a target place (e.g., target_place=... / PLACES ...).
   - If a target place exists but current location is unclear, ask a clarifying question like:
     "Are you currently in <target_place>, or is this for a future plan?"

7. Relevance gating (CRITICAL):
   - Only apply preferences when they are relevant to the request.
     Example: color preferences matter for clothing/decor, NOT for diet/health advice.
   - Health/diet constraints (allergy, intolerance, dietary_avoid, diet) are ALWAYS relevant for food/drink advice.

MEMORY is the source of truth about the user.

Respond in natural language (no JSON).
"""

@dataclass
class GeminiExtractor:
    model: str = "gemma-3-1b-it"

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON robustly — strips markdown fences Gemma sometimes wraps output in."""
        if not text:
            return {}
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        try:
            return json.loads(cleaned)
        except Exception:
            return {"notes": ["non_json_extraction_response"], "raw_text": text}

    def extract_patch(self, *, memory: str, user_text: str) -> Patch:
        prompt = (
            f"EXISTING MEMORY:\n{memory}\n\n"
            f"USER MESSAGE:\n{user_text}\n\n"
            "Extract facts into JSON:"
        )
        r = generate_content(
            system_instruction=_EXTRACT_SYSTEM,
            user_text=prompt,
            model=self.model,
            temperature=0.0,
            max_output_tokens=400,
            response_mime_type="application/json",
            response_json_schema=PATCH_JSON_SCHEMA,
        )
        try:
            parsed = self._parse_json(r.text) if r.text else {}

            # Normalize preference constraints so the compressor can reliably read them.
            # Gemma sometimes emits "blue as well" or a wrong category; we clean and canonicalize.
            COLOR_WORDS = {
                "red",
                "blue",
                "green",
                "yellow",
                "black",
                "white",
                "purple",
                "pink",
                "orange",
                "brown",
                "gray",
                "grey",
            }

            for u in parsed.get("upserts", []) or []:
                attrs = u.get("attributes", {})
                if not isinstance(attrs, dict):
                    continue

                # Clean common filler words in extracted values.
                if "value" in attrs and isinstance(attrs["value"], str):
                    v = attrs["value"].strip()
                    v = re.sub(r"\b(as well|also|very|really)\b", "", v, flags=re.IGNORECASE).strip()
                    attrs["value"] = v.lower()

                # Canonicalize preference node names: preference:<category>:<value>
                if (u.get("type") or "").strip() == "constraint":
                    kind = str(attrs.get("kind") or "").strip().lower()
                    name = str(u.get("name") or "").strip()

                    is_pref = kind == "preference" or name.lower().startswith("preference:")
                    if is_pref:
                        cat = attrs.get("category")
                        val = attrs.get("value")

                        # If category/value are missing, try to parse from name.
                        parts = name.split(":")
                        if len(parts) == 3 and parts[0].lower() == "preference":
                            cat = cat or parts[1]
                            val = val or parts[2]

                        if isinstance(cat, str):
                            cat = cat.strip().lower()
                        if isinstance(val, str):
                            val = re.sub(r"\b(as well|also|very|really)\b", "", val, flags=re.IGNORECASE).strip().lower()

                        # If it looks like a color but category isn't color, fix category.
                        if isinstance(val, str) and val in COLOR_WORDS and cat not in {"color", "colour"}:
                            cat = "color"

                        if isinstance(cat, str) and isinstance(val, str) and cat and val:
                            attrs["kind"] = "preference"
                            attrs["category"] = cat
                            attrs["value"] = val
                            u["name"] = f"preference:{cat}:{val}"
            
            return parsed
        except Exception:
            return {"notes": ["non_json_extraction_response"], "raw_text": r.text}

    def extract_patch_and_answer(self, *, memory: str, user_text: str) -> Dict[str, Any]:
        """
        Two-call approach for models like Gemma that can't reliably do JSON+prose in one shot:
          1. extract_patch()  — JSON only, low temperature
          2. generate answer  — plain text, higher temperature
        Returns {"answer": str, "patch": Patch} matching the standard interface.
        """
        # Call 1: memory extraction (JSON)
        patch = self.extract_patch(memory=memory, user_text=user_text)

        # Call 2: natural language answer
        answer_prompt = (
            f"MEMORY (facts about this user):\n{memory}\n\n"
            f"USER: {user_text}"
        )
        r = generate_content(
            system_instruction=_ANSWER_SYSTEM,
            user_text=answer_prompt,
            model=self.model,
            temperature=0.7,
            max_output_tokens=600,
        )
        answer = (r.text or "").strip()
        return {"answer": answer, "patch": patch}


def apply_patch(graph: KnowledgeGraph, patch: Patch) -> None:
    """
    Applies an extraction patch to the graph.
    This is intentionally forgiving: unknown fields are ignored.
    """
    for inv in patch.get("invalidations", []) or []:
        t = (inv.get("type") or "city").strip()
        name = (inv.get("name") or "").strip()
        if not name:
            continue
        graph.invalidate_subtree(entity_name=name, root_type=t)

    for u in patch.get("upserts", []) or []:
        t = (u.get("type") or "").strip()
        name = (u.get("name") or "").strip()
        if not t or not name:
            continue
        status = u.get("status")
        attrs = u.get("attributes") if isinstance(u.get("attributes"), dict) else None

        # Preserve assignment rule: constraints don't expire unless superseded.
        if t == "constraint" and status == STATUS_INVALIDATED:
            status = None
        graph.upsert(t, name, status=status, data=attrs)
  
        # Budget convenience: allow extractor to upsert constraint kind=budget_total with total.
        if t == "constraint" and attrs and attrs.get("kind") == "budget_total" and "total" in attrs:
            try:
                graph.set_budget_total(int(attrs["total"]))
            except Exception:
                pass

    for rel in patch.get("relations", []) or []:
        st = (rel.get("src_type") or "").strip()
        sn = (rel.get("src_name") or "").strip()
        rt = (rel.get("rel") or "").strip()
        dt = (rel.get("dst_type") or "").strip()
        dn = (rel.get("dst_name") or "").strip()
        if not (st and sn and rt and dt and dn):
            continue
        s = graph.upsert(st, sn)
        d = graph.upsert(dt, dn)
        graph.add_edge(s.id, rt, d.id)
