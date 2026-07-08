# Configuring the backend

The backend is the language model that drives the agents. It is chosen in the
study's `config.yaml` and does not change the graph or the science. All backends
report token usage through the same telemetry, so cost and throughput are
comparable across them.

## Claude CLI (default)

Uses the local `claude` CLI (subscription auth preferred, `ANTHROPIC_API_KEY`
otherwise). Nothing to configure beyond the model.

```yaml
backend: claude
model: claude-haiku-4-5-20251001
```

## Ollama

A local Ollama server. Point at it with `OLLAMA_BASE_URL` (defaults to
`http://localhost:11434`).

```yaml
backend: ollama
model: qwen2.5:7b
```

## OpenAI-compatible endpoints (OpenRouter, vLLM, others)

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

## A local model on a SLURM GPU node (vLLM)

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

See the feature catalog for the full set of `llm_slurm` knobs (resource
overrides, queue and serve timeouts, tensor-parallel).
