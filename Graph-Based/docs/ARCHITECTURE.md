# Architecture: Graph Memory + Context Compression

## Data Flow

Raw conversation
  -> extractor (runs on every user turn)
     - entity extraction (constraints, cities, events, expenses)
     - invalidation triggers ("forget X", "scratch X", "let's do X instead")
     - graph updates (nodes/edges/status)
  -> compressor
     - builds compact "ACTIVE MEMORY" summary
  -> agent/model
     - uses compressed summary as context

## Key Design Decisions

- Constraints are persistent nodes (no TTL). They may be superseded (e.g., budget updated) but are not deleted.
- Cities can be invalidated (not deleted). This preserves "rejected" memory to avoid re-suggesting.
- Budget state is recomputed every time from expense nodes (active expenses only).
- Events carry temporal attributes (day/time) and link to city via edges.

## What Counts as "Baseline"

Baseline is naive "full-history stuffing":
- the entire chat transcript (or last N turns) is concatenated and fed into the model.

Compressed mode feeds:
- only the graph-compressed summary from `graph_memory/compressor.py`.

