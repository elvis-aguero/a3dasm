# Customizing a run

Everything in [Authoring a study](authoring-a-study.md) works with the built-in
graph: a **strategizer** delegating to **literature reviewer**, **data
generator**, **implementer**, and **critic**, all driven by one backend. The
graph, each shipped agent's system prompt, and which backend/model drives
each agent are all Python-level extension points, not `config.yaml`
settings. This page covers all three, and where each stops being a
supported, tested surface.

## A custom graph

`AgenticRun` takes an optional `graph=` argument: an `a3dasm.Graph` built from
`a3dasm.Agent` subclasses and `a3dasm.Edge`s.

```python
from a3dasm import Agent, Edge, Graph, AgenticRun


class Strategist(Agent):
    role = "strategizer"  # the hub; other roles default to "worker"
    description = "Decides what to try next."  # required, or Graph() raises
    system_prompt = "You are the strategizer. Delegate to the implementer."


class Implementer(Agent):
    description = "Writes and runs the evaluation code."
    # no system_prompt override here: falls back to Agent's default ""


graph = Graph(
    nodes={"strategizer": Strategist(), "implementer": Implementer()},
    edges=(Edge("strategizer", "implementer"),),  # who may delegate to whom
    entry="strategizer",  # who gets the initial briefing
)

AgenticRun(study_dir="my_study", graph=graph).execute()  # swaps in this graph
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

## A different backend or model, per agent

When the runtime builds an agent's session, it resolves the backend and
model independently: `agent.backend or self._backend` and
`agent.model or self._model` (`agent_runtime.py`'s `_make_adapter`), where
`self._backend`/`self._model` are the run's own defaults, whatever
`config.yaml` or `AgenticRun`'s constructor set. But the two are set
differently, because `backend` and `model` aren't handled the same way on
`Agent`:

- **`backend`** is a plain class attribute, default `None`. Override it like
  `system_prompt`, by setting it on the subclass.
- **`model`** is constructor-only: `Agent.__init__` always does
  `self.model = model`, so a class-level `model = "..."` is silently
  shadowed by that assignment. Pass it when you instantiate instead.

```python
class Implementer(Agent):
    description = "Writes and runs the evaluation code."
    backend = "ollama"  # class attribute; every Implementer() gets this

implementer = Implementer(model="qwen2.5:7b")  # constructor arg, per instance
```

This agent alone runs local; everyone else in the graph still uses whatever
the run's default backend/model is. This is the same lever `system_prompt`
uses (a class attribute the runtime reads generically), so the same caveat
applies: no shipped agent sets a per-agent `backend`, and no test in this
project exercises a graph that mixes backends across agents. It's real
(`agent_runtime.py:1250`), but running most of your graph on Claude and one
specialist on a local model is a combination you'd be the first to try.

## The available backends

Whichever backend a run defaults to, whether set globally in `config.yaml`
or per agent above, all backends report token usage through the same
telemetry, so cost and throughput stay comparable.

### Claude CLI (default)

Uses the local `claude` CLI: see [Installation](installation.md) if you
haven't set it up yet. Once `claude` runs on its own in your terminal,
there's nothing else to configure beyond the model:

```yaml
backend: claude
model: claude-haiku-4-5-20251001
```

### Ollama

A local Ollama server. Point at it with `OLLAMA_BASE_URL` (defaults to
`http://localhost:11434`).

```yaml
backend: ollama
model: qwen2.5:7b
```

### OpenAI-compatible endpoints (OpenRouter, vLLM, others)

Any server that speaks the OpenAI API. The adapter resolves the base URL from an
explicit argument, then the relevant `*_BASE_URL` environment variable, then a
default.

```yaml
backend: openrouter        # or: vllm
model: meta-llama/llama-3.1-70b-instruct
```

```bash
export OPENROUTER_API_KEY=...        # or VLLM_BASE_URL=http://host:8000/v1
```

### A local model on a SLURM GPU node (vLLM)

a3dasm can own a model served on a separate SLURM GPU allocation for the whole
run: it submits the `vllm serve` job, waits for the node and a ready server,
points the backend at it over the cluster network, and cancels the job on every
exit path. Enable it with an `llm_slurm` block; a config-time throughput estimate
warns if the model/GPU choice is likely to be painfully slow.

```yaml
backend: vllm
llm_slurm:
  enabled: true
  model: gemma-4-27b-it
  gpu_model: a100            # enables the config-time speed check
  cluster:
    partition: gpu
    account: my_acct
    runner: "uv run python"
    env_setup: ["module load cuda", "module load vllm"]
```

The `llm_slurm` block also accepts resource overrides (`gres`, `mem`, `time`),
queue and serve timeouts, and a `tensor_parallel` size for sharding a large model
across GPUs.
