# Customizing a run

The [Quickstart](notebooks/quickstart.ipynb) ran the Branin problem with
a3dasm's own defaults: the built-in graph, the shipped agents' own prompts,
one backend for everything. This page reuses that same problem to show what
changes, and how, as you reach for something different.

## The starting point

```python
from pathlib import Path
from a3dasm import AgenticRun

study_dir = Path("studies/branin")
study_dir.mkdir(parents=True, exist_ok=True)
(study_dir / "PROBLEM_STATEMENT.md").write_text(
    "Minimise the 2D Branin function over its standard domain.\n"
    "Report the best design found and the objective value there.\n"
)

AgenticRun(study_dir=study_dir, model="claude-haiku-4-5-20251001").execute()
```

No `config.yaml` here; `model` is the only thing set, and there's no
`backend` at all (it defaults to Claude). This runs the built-in graph
(strategizer plus four specialists), every agent on the same backend, every
agent using its own shipped prompt.

## A different backend for the whole run: `config.yaml`

Say the run should go through Ollama instead. Drop a `config.yaml` next to
`PROBLEM_STATEMENT.md`:

```yaml
backend: ollama
model: qwen2.5:7b
```

Nothing else changes. Same graph, same prompts, same
`AgenticRun(study_dir=study_dir).execute()` call (`model`/`backend` now come
from `config.yaml`, so the constructor doesn't need them). Every agent in
the built-in graph now runs on Ollama, because none of the shipped agents
(`StrategizerAgent`, `LiteratureReviewAgent`, `DataGeneratorAgent`,
`F3dasmImplementerAgent`, `AdversarialCritiqueAgent`) sets its own
`backend`, so each one falls back to the run's default:
`agent.backend or self._backend` (`agent_runtime.py`'s `_make_adapter`,
line 1250).

## A different backend for one agent: this needs Python, not YAML

`config.yaml`'s `backend`/`model` are run-wide; there's no YAML key for
"just the implementer." Getting that means setting `backend`/`model` on one
node, which means that node is your own `Agent` subclass, which means
building the graph yourself: swapping a hand-rolled agent into the shipped
graph while keeping the rest isn't a pattern this project tests. So a custom
graph, a custom prompt, and a different backend for one node usually arrive
together, in one `Graph`:

```python
from a3dasm import Agent, Edge, Graph, AgenticRun


class Strategist(Agent):
    role = "strategizer"  # the hub; other roles default to "worker"
    description = "Decides what to try next."  # required, or Graph() raises
    system_prompt = "You are the strategizer. Delegate to the implementer."
    # backend/model unset: this node falls back to the run's own default


class Implementer(Agent):
    description = "Writes and runs the evaluation code."
    backend = "ollama"  # a class attribute override, only for this node


graph = Graph(
    nodes={
        "strategizer": Strategist(),
        "implementer": Implementer(model="qwen2.5:7b"),  # model is a
        # constructor arg, not a class attribute: Agent.__init__ always
        # does self.model = model, which would silently shadow a
        # class-level override
    },
    edges=(Edge("strategizer", "implementer"),),  # who may delegate to whom
    entry="strategizer",  # who gets the initial briefing
)

AgenticRun(study_dir=study_dir, graph=graph).execute()
```

Run this with no `config.yaml` (or one that just says `backend: claude`) and
the strategizer runs on Claude, the run's default, while the implementer
alone runs on Ollama with `qwen2.5:7b`. Nothing here reads `config.yaml` for
the implementer's backend at all; `Implementer(model=...)` and
`backend = "ollama"` are the only source of truth for that one node.

A bare `Agent` subclass starts from zero tools (`Agent.tools` defaults to
`frozenset()`) and zero epistemic machinery. The shipped agents wire up the
hypothesis ledger, the reproduction gate, and each other's delegation tools
already; a hand-rolled one, like `Strategist`/`Implementer` above, does not.
This is also why no shipped agent sets a per-agent `backend`, and no test in
this project exercises a graph that mixes backends: it's a real, working
lever (as shown above), but running most of a graph on Claude and one node
on a local model is a combination you'd be the first to try.

## Reference: the available backends

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
