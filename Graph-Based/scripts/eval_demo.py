from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_memory import GraphMemoryModule
from graph_memory.metrics import estimate_tokens


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate baseline full-history stuffing vs graph-compressed context."
    )
    ap.add_argument(
        "--file",
        default=None,
        help="Path to a conversation JSON file (list of {role,text}). Defaults to data/demo_conversation.json.",
    )
    ap.add_argument(
        "--baseline-turns",
        type=int,
        default=0,
        help="If >0, baseline uses only last N turns; if 0, uses full history.",
    )
    ap.add_argument(
        "--print-baseline",
        action="store_true",
        help="Print the baseline stuffed context (can be long).",
    )
    args = ap.parse_args()

    if args.file:
        data_path = Path(args.file).expanduser().resolve()
    else:
        data_path = Path(__file__).resolve().parent.parent / "data" / "demo_conversation.json"

    convo = json.loads(data_path.read_text(encoding="utf-8"))

    mem = GraphMemoryModule()
    for turn in convo:
        role = turn["role"]
        text = turn["text"]
        if role == "user":
            mem.ingest_user_turn(text)
        else:
            mem.ingest_assistant_turn(text)

    baseline_turns = None if args.baseline_turns <= 0 else args.baseline_turns
    baseline = mem.baseline_context(max_turns=baseline_turns)
    compressed = mem.compressed_context()

    baseline_tokens = estimate_tokens(baseline)
    compressed_tokens = estimate_tokens(compressed)
    ratio = (baseline_tokens / compressed_tokens) if compressed_tokens else float("inf")

    print("Evaluation Demo (baseline stuffing vs compressed graph summary)")
    print(f"- Input file: {data_path}")
    print(f"- Turns: {len(mem.raw_history)}")
    print(f"- Baseline tokens (estimated): {baseline_tokens}")
    print(f"- Compressed tokens (estimated): {compressed_tokens}")
    print(f"- Compression ratio: {ratio:.2f}x")
    print("")
    if args.print_baseline:
        print("Baseline Context:")
        print(baseline)
        print("")
    print("Compressed Context:")
    print(compressed)
    print("")
    print("Tip: for a human-friendly view, run the chat CLI with --show-context.")


if __name__ == "__main__":
    main()
