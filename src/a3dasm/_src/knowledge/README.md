# Agentic Knowledge Base (DRAFT)

The canonical source of truth for **discretionary conventions and practices**
that help agentic workers do their jobs well — consultable on demand, instead
of bloating every system prompt with the same standing guidance.

## Status

Draft scaffold. Today it is a curated markdown corpus (`entries/`) with a
stable Python API (`KnowledgeBase`) and a transparent keyword search. The
search backend is meant to be swapped for semantic (embedding) retrieval once
the corpus is large enough to justify it — the `KBEntry` / `KnowledgeBase` API
is designed to stay stable across that change.

The intended end state is a worker-facing tool (e.g. `ConsultKnowledge(query)`)
wired into the agent tool set, so any node — implementer, datagenerator,
critic, lit reviewer, strategizer, and any future node — can pull the relevant
guidance at the moment it needs it.

## The line: enforce invariants, retrieve conventions

This is the load-bearing rule for the whole KB:

- **In the KB (retrieve):** discretionary conventions, idioms, how-tos, and
  gotchas — the long tail of "how we do things."
- **NOT in the KB (enforce):** mandatory invariants. Rules like *ground truth
  must go through `get_evaluator()`* or *the headline must reproduce from the
  store* are enforced by the **ScienceMonitor** and the **critic gate** — never
  left to optional retrieval. The KB may restate an invariant as guidance, but
  enforcement must never depend on an agent choosing to search.

A critical rule that lives only in a KB is a rule that silently fails.

## Authoring an entry

One markdown file per chunk under `entries/`, with frontmatter:

```markdown
---
id: short-kebab-id
title: One-line, action-oriented title
tags: [searchable, keywords, here]
audience: [implementer, strategizer, ...]
---
Body. Keep it self-contained — a worker should be able to act on this one
entry without reading the rest. Match the wording used in the prompts/nudges
(e.g. "canonical ExperimentData store", "get_evaluator()", "provenance") so
the corpus stays self-consistent.
```

## Curation

Node **retrospectives** are the natural feeder: when a worker reports friction
or a counterintuitive convention, that becomes a candidate entry. Curation is
**gated by a human** — never auto-ingest a retrospective into the KB. Keep
entries current; a stale standard actively misleads.
