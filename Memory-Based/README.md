# Context Compression for AI Agents

## Background

AI agents are increasingly expected to work across long conversations, large documents, tool chains, and multi-session workflows. But their active context remains finite, expensive, and vulnerable to noise. Simply expanding context windows is not enough: agents still struggle to consistently identify, preserve, and use the most relevant information at the right time.

This problem invites participants to build systems that help agents compress context without losing meaning, maintain memory across sessions, and retrieve only what matters for the next action. The goal is to make AI agents more reliable, efficient, and scalable under real-world constraints.

- Make every token count.
- Teach agents what to remember, what to forget, and what to re-inject.
- Go beyond long context: build efficient context compression for real AI agents.

## Problem Statement

Modern AI agents operate across long conversations, large documents, tool traces, and multi-session workflows. In these settings, simply passing more raw data to the model is often ineffective: uncurated histories, redundant tool outputs, and noisy retrieval can overwhelm the active context window, while long-context research shows that model performance still degrades as length and complexity rise.

Participants are challenged to build a **context compression system** that helps an agent preserve high-signal information, discard low-value context, and recover the right information at the right time so the agent remains coherent, accurate, and efficient over extended tasks.

## Core Objective

Design and prototype a system component that sits between an agent and its growing context and improves the agent’s ability to operate under constrained or noisy context conditions.

The artifact should be **reusable by agents**, not tailored as a one-off demo app. The system may:

- compress conversational history,
- summarize tool traces,
- maintain structured memory,
- prioritize retrieval,
- compact intermediate state, or
- manage multiple memory tiers.

This aligns with current platform guidance and research patterns such as session trimming/compression, structured notes across runs, persistent long-term memory, and hierarchical memory systems.

---

## What You Need to Do

### What participants should **not** build

This challenge is **not** about shipping a polished chatbot, enterprise dashboard, or consumer application. A UI is optional and only useful insofar as it helps demonstrate the system.

The judged artifact is the **compression system and its measured effect on agent behavior**, not the front-end experience.

### Required proof of value

Every team must demonstrate **two conditions on the same task or task family**:

#### 1. Baseline agent without context compression
Show failure, degradation, inefficiency, or instability organically when the agent relies on naive context handling such as:

- full-history stuffing,
- unfiltered retrieval, or
- raw tool traces.

#### 2. Agent with the proposed context-compression system
Show measurable improvement in one or more of the following:

- task success,
- factual retention,
- coherence across turns,
- tool-call quality,
- token efficiency,
- robustness across long horizons.

This before-vs-after requirement is important because long-context research shows that larger nominal windows do not guarantee reliable use of information. Relevant information buried in long contexts is often used poorly, and many long-context models show major drops as length and complexity increase.

---

## Success Definition

In practical terms, success means the system does at least some of the following:

- preserves critical facts, goals, and constraints across long interactions,
- reduces irrelevant or duplicated context,
- maintains or improves downstream task quality,
- supports continuity across sessions or windows,
- exposes interpretable memory artifacts or compression decisions, and
- demonstrates clear tradeoffs between compression and quality.

To keep the work **agent-centric rather than app-centric**, teams should build their context compression system around a use case involving:

- long dialogues,
- document understanding,
- multi-turn calls,
- user preferences, and
- evolving task state.

Refer to the **Use Case** section below.

---

## Minimum System Expectations

A valid submission should include:

- a clearly defined compression mechanism and architecture,
- an agent integration point showing where the system operates,
- a baseline comparison,
- an explanation of what information is preserved, transformed, or discarded,
- quantitative metrics as described below, and
- the full architecture showing the overall system workflow.

> **Recommended setup:** Preferably use an open-source LLM with a low context window (8K or 16K), for example the **smolLM2** family by Hugging Face.

## Minimum Evaluation Expectations

Participants should report quantitative or semi-quantitative evidence such as:

- token reduction or compression ratio,
- latency reduction,
- cost reduction,
- downstream task success rate,
- factual retention or recall,
- coherence over long turns,
- tool-call correctness,
- multi-session continuity, and
- omission or distortion rate introduced by compression.

---

## Non-Goals

Submissions should be considered weak if they are primarily:

- a simple chatbot wrapper with no actual compression logic,
- a pure RAG demo with no memory or compaction policy,
- a larger-context prompt stuffed with all available text,
- an app demo with no reproducible evidence,
- a manual summarization flow that cannot plausibly be reused by agents, or
- an agent framework feature that performs compression out of the box without meaningful original system design.

---

## Chat Bot Interface

Participants are free to use
1. Any open-source chatbot interface to re-use.  Examples include Open WebUI, or Gradio Chat Interface or any other re-usable interface.
2. Or build a chat bot interface from scratch

In any case, the deliverable will be the plug-and-play **Context Compression Module** evaluation will done based on the **Context Compression Module** itself.

---

## Use Case: Multi-City Trip Planning Agent

This is the sweet spot for this challenge.

Build a travel agent that helps a user plan a **10-12 day trip across multiple cities**, handling:

- flights,
- hotels,
- restaurants,
- activities,
- personal preferences,
- budgets, and
- logistics.

### Why this use case works well

The context grows in every dimension that matters:

- User preferences such as budget, dietary restrictions, mobility constraints, and travel style are stated once early and must persist across 20+ turns.
- Each tool call (web search for flights, hotel lookups, restaurant searches, weather checks) dumps large payloads into context.
- The user changes their mind mid-conversation, forcing the agent to reason over what is still valid and what has gone stale.
- The agent must cross-reference across distant turns, such as remembering that a restaurant suggestion conflicts with a dietary restriction mentioned much earlier.

### Suggested tools

Build the agent with these tools (you may add your own as well) make the tool definitions rule based (if you want), but the tools should return valid outputs:

- `web_search` — flights and general information
- `places_search` — hotels, restaurants, attractions
- `weather_fetch` — packing and activity decisions
- `budget_tracker` — a simple function that maintains a running spend tally

Wire the agent with a system prompt that instructs it to act as a **travel concierge** that:

- references prior preferences,
- tracks budget continuously, and
- proactively flags conflicts.

### Evaluation setup

Build a lightweight evaluation script that replays scripted multi-turn conversations and checks the agent’s final outputs against ground-truth assertions.

> The validations below are illustrative examples to give the flavor of evaluation. They are **not exhaustive**.

---

# Evaluation

## Layer 1: User-Stated Validation

### Test Conversation A — *The Forgotten Allergy*

| Turn | User Message |
|------|--------------|
| 1 | "I want to plan a 5-day trip to Tokyo and Kyoto. Budget is $3,000 total. I'm severely allergic to shellfish." |
| 2-15 | Research phase: flights, hotels, transit between cities (generates ~6-8 tool calls of context) |
| 16 | "Find me the best dinner spots in Tsukiji area" |

**Pass criteria:**  
The agent must either filter out sushi/seafood-heavy restaurants or explicitly warn about the shellfish allergy. If it recommends a seafood market restaurant without caveats, the context compression system lost the allergy information — **hard fail**.

---

### Test Conversation B — *The Budget Anchor*

| Turn | User Message |
|------|--------------|
| 1 | "Planning a trip to Italy — Rome, Florence, Amalfi Coast. 7 days, max budget $2,500, I'm a solo traveler." |
| 2-19 | Book flights (~$800), hotels in Rome (~$400), hotels in Florence (~$350) |
| 20 | "Find me a hotel on the Amalfi Coast" |

**Pass criteria:**  
The agent should know that roughly **$950 remains** and recommend accordingly. It should not suggest a **$500/night luxury resort** without flagging the budget tension. If it loses track of cumulative spend, the compression system dropped important numerical state.

---

## Layer 2: Stale Context Invalidation

**Question:** Does the agent forget what it *should* forget?

These tests verify whether the agent correctly stops using information the user has explicitly overridden.

### Test Conversation C — *The Pivot*

| Turn | User Message |
|------|--------------|
| 1 | "Plan me a beach vacation in Bali for next month." |
| 2-6 | Agent researches Bali: flights, beach resorts, surf lessons, temple tours — heavy context accumulation |
| 7 | "Actually, scratch Bali entirely. Let's do Switzerland instead — I want mountains, not beaches." |
| 8-25 | Agent researches Switzerland |
| 26 | "Summarize my trip plan so far." |

**Pass criteria:**  
The summary must contain **zero Bali references**. No leakage such as "as we discussed earlier about the beach resorts...". If compressed context retains Bali fragments that bleed into the Switzerland plan, that is a compression failure.

---

## Layer 3: Cross-Reference Reasoning

**Question:** Can the agent connect dots across distant turns?

This is the hardest test. The agent must synthesize information from multiple tool calls spread across many turns.

### Test Conversation D — *The Logistics Puzzle*

| Turn | User Message |
|------|--------------|
| 1 | "I'm planning 6 days: 3 in Paris, 3 in Amsterdam. I have a meeting in Paris on Wednesday at 2pm near the Eiffel Tower." |
| 2-9 | Book flights, hotel in Paris |
| 9-14 | Research Amsterdam hotels, Anne Frank House tickets, canal tours |
| 15 | "When should I take the train from Paris to Amsterdam?" |

**Pass criteria:**  
The agent must reason that the user’s **Wednesday 2pm meeting in Paris** means the train should be **Thursday morning** (or **Wednesday evening at the earliest**). This requires connecting turn 1 with turn 10 across eight turns of noisy tool-result context.

---

### Test Conversation E — *The Contradiction Detector*

| Turn | User Message |
|------|--------------|
| 1 | "I want a relaxing trip. No packed schedules. Max 2 activities per day." |
| 2-19 | Extensive research: agent finds 15+ attractions across the trip |
| 20 | "OK book all of these for my 3-day trip" |

**Pass criteria:**  
The agent should push back, for example:

> "That's 15 activities across 3 days, which conflicts with your preference for max 2 per day. Want me to help prioritize?"

If it blindly schedules everything, the compression system lost the relaxation constraint.

---

## Final Note

The evaluation scenarios above are **examples, not strict limits**. Participants are encouraged to think beyond these cases and design additional scenarios that better stress-test their context compression system across different forms of memory, retrieval, tool usage, and long-horizon reasoning.
