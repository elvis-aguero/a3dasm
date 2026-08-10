# Customizing the graph

Everything in [Authoring a study](authoring-a-study.md) works with the built-in
graph: a **strategizer** delegating to **literature reviewer**, **data
generator**, **implementer**, and **critic**. That graph, and each shipped
agent's system prompt, are Python-level extension points, not `config.yaml`
settings. This page covers both, and where each stops being a supported,
tested surface.

## A custom graph

`AgenticRun` takes an optional `graph=` argument: an `a3dasm.Graph` built from
`a3dasm.Agent` subclasses and `a3dasm.Edge`s.

```python
from a3dasm import Agent, Edge, Graph, AgenticRun


class Strategist(Agent):
    role = "strategizer"
    description = "Decides what to try next."
    system_prompt = "You are the strategizer. Delegate to the implementer."


class Implementer(Agent):
    description = "Writes and runs the evaluation code."


graph = Graph(
    nodes={"strategizer": Strategist(), "implementer": Implementer()},
    edges=(Edge("strategizer", "implementer"),),
    entry="strategizer",
)

AgenticRun(study_dir="my_study", graph=graph).execute()
```

- **`nodes`** maps a name to an `Agent` instance. Every node needs a non-empty
  `description`; `Graph` raises `ValueError` if one is missing.
- **`edges`** are directed: `Edge("strategizer", "implementer")` lets the
  strategizer delegate to the implementer, and is what gives the implementer
  its `Delegate` tool. Both endpoints must be declared nodes, or `Graph`
  raises `ValueError`.
- **`entry`** is the node that receives the initial briefing; it must be a
  declared node.

This is the same shape the built-in graph uses (see `_default_graph()`): five
nodes, six edges, `entry="strategizer"`. There is no `config.yaml` key for
this: a custom graph is a constructor argument, full stop.

A bare `Agent` subclass starts from zero tools (`Agent.tools` defaults to
`frozenset()`) and zero epistemic machinery. The shipped agents
(`StrategizerAgent`, `LiteratureReviewAgent`, `DataGeneratorAgent`,
`F3dasmImplementerAgent`, `AdversarialCritiqueAgent`) wire up the hypothesis
ledger, the reproduction gate, and each other's delegation tools already.
Swapping in a hand-rolled `Agent` for one shipped role, while keeping the
rest of the built-in graph, isn't a pattern this project tests today, so
it isn't documented here as supported: build a full custom graph from
scratch instead, as above.

## A custom system prompt

`system_prompt` is a plain class attribute on `Agent` (default `""`); the
runtime reads `agent.system_prompt` whenever it builds that node's session.
Setting it is exactly what the example above does for `Strategist`. There is
no `system_prompt=` constructor keyword and no `config.yaml` key for it
either: subclassing and overriding the class attribute is the only lever.

One thing this does *not* give you: a3dasm always prepends its own run-paths
or workspace preamble to whatever you set in `system_prompt`. You can control
your agent's own instructions; you cannot suppress the preamble a3dasm adds
ahead of them.
