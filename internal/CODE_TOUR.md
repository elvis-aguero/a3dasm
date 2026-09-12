# Code tour — where everything an agent sees actually lives

Written for the question "justify this instruction": for every word a model
reads and every rule that can stop it, this says which file to open.

For the interactive version — full assembled prompts, per-section `file:line`,
full-text search — regenerate and open the map:

```bash
make promptmap          # -> internal/promptmap.html
```

It is generated from the live `Graph`/`Agent` objects, never hand-typed, and
`tests/test_promptmap.py` fails if a prompt moves out from under its citation.

---

## 1. The shape of a run

`AgenticRun` (`src/a3dasm/_src/runtime/agent_runtime.py`) builds a LangGraph
StateGraph over a five-node topology defined in one place,
`src/a3dasm/_src/agents/_graphs.py`:

```
strategizer (entry) ──> literature_reviewer
                   ──> datagenerator ──> literature_reviewer
                   ──> implementer   ──> literature_reviewer
                   ──> critic
```

The strategizer is the only node that plans; the implementer is the only node
that evaluates designs. The single user input is
`<study_dir>/PROBLEM_STATEMENT.md`; the single deliverable is `pipeline.ipynb`.

Entry points: `AgenticRun(...).execute()` in-process, `python -m a3dasm
<study_dir>` on the command line (`src/a3dasm/__main__.py`), or
`python -m a3dasm._src.runtime.run` as the container entrypoint
(`runtime/run.py`). All three take the same `--model` and `--budget`.

## 2. What an agent sees, in assembly order

`agent_runtime.py` assembles four layers per node (around
`agent_runtime.py:1250-1340`). In order:

| # | Layer | Defined in | Notes |
|---|-------|-----------|-------|
| 1 | Run-paths preamble (entry) / workspace preamble (workers) | `prompts/agent_prompts.py` (`RUN_PATHS_PREAMBLE_TEMPLATE`, `WORKSPACE_PREAMBLE_TEMPLATE`) | Paths, resource stanza, handbook menu — substituted per run |
| 2 | The role's own system prompt | `agents/<role>.py`, inlined at module top | The bulk of the text |
| 3 | Notebook deliverable contract | `evaluation/notebook_exec.py::notebook_deliverable_spec` | Only `strategizer` / `implementer` / `critic`, and only while the `pipeline_deliverable` knob is true |
| 4 | `<tools>` catalog | `prompts/tool_catalog.py::render_tool_catalog` | **Generated** from the live closure dict — each entry is the tool's own docstring, so it cannot name a tool the agent lacks |

Two blocks are defined once and injected verbatim into more than one role, so
two agents can cite the same clause number without paraphrase drift:

- `knowledge/charter.py::FALSIFICATION_CHARTER` — the Popperian rules (§1–§6).
  Reaches the **strategizer** and the **critic**, and the verdict validator
  (`epistemics/verdict_validator.py`) judges against the same text.
  Wording is pinned by `tests/test_charter.py`.
- `knowledge/idioms.py::F3DASM_CORE_IDIOMS` — the f3dasm API surface. Reaches
  the **datagenerator** and the **implementer**.

Note `prompts/agent_prompts.py` is largely a **re-export surface**: the system
prompts themselves live next to their agent classes (deliberately — it keeps
them reachable via `inspect.getsource`). Search `agents/`, not `prompts/`.

### Text the runtime injects mid-run

Nudges, science-monitor drift, budget warnings, operator notes and Confer
messages do not go in the system prompt — they are prepended to the text of
whatever tool the agent calls next, wrapped in `<a3dasm-note>` markers so the
agent and the viewer can both tell runtime speech from tool output.
See `nodes/notices.py`.

## 3. Gating

Two kinds, and the distinction is the one to be precise about in review:

- **Hard** — refuses the action synchronously at the data boundary. The agent
  cannot proceed.
- **Soft** — speaks and lets the agent proceed. Advisory notes, not control.

The close sequence is a literal tuple in
`nodes/tools/routing/feedback.py::Done`; a gate that holds returns immediately
and nothing after it runs:

1. `_pending_refusal` — nothing closes while a delegation is in flight (soft)
2. `_capture_retrospective` — the exit interview
3. `_milestone_gate` — process backlog must be DONE or SKIPPED (hard; knob
   `milestones_enabled`; escape `MilestoneSkip(reason)`)
4. `_first_call_warning` — Done() is two-shot
5. `_must_reproduce` — the deliverable must execute before a critic turn is
   spent on it (hard, bounded)

then `_close` → `_critic_gate`: **a run closes only on a critic PASS.**

Elsewhere:

- **Reproduction gate** — `nodes/reproduction_gate.py`. `pipeline.ipynb` must
  execute cleanly in a hermetic sandbox copy of the store, add **zero** oracle
  rows and rewrite none. `_headline_consistency` additionally cross-checks the
  notebook's stated answer against its computed one, and is deliberately
  lenient (silent when either marker is absent).
- **Hypothesis data boundary** — `nodes/tools/routing/ledger.py`.
  `HypothesisUpdate` refuses a `SUPPORTED` with no falsification attempt, and
  refuses evidence citing a delegation that never ran.
- **Delegation** — `nodes/tools/routing/delegation.py`. A delegation must be
  anchored to hypotheses that exist; the implementer is unreachable while the
  milestone backlog is open.
- **Verdict validator (#9)** — `epistemics/verdict_validator.py`. An
  independent referee judges closing verdicts against the same charter.
  Advisory, and killable: `F3DASM_VERDICT_VALIDATOR=0`.
- **Science monitor** — `epistemics/science_monitor.py`. Only the
  `UNLEDGERED_EVALS` runtime rule remains; unstamped rows and duplicate
  evaluations are reported as in-band notices. Never blocks.
- **Problem-statement reviewer** — `epistemics/reviewer.py`. Advisory by
  design: always writes a report, never blocks a run.

## 4. Knobs

`runtime/settings.py` is the accessor and the list of known keys. Resolution
order per knob: `F3DASM_<KEY>` environment variable, then the study's
`config.yaml` `runtime:` block, then the default. An unrecognised key in
`runtime:` is reported rather than silently ignored — the classic
silent-config failure is designed out. The Config knobs view of the map lists
every one with its read site.

## 5. Where the rest of it is

| Concern | Directory |
|---|---|
| Model adapters (Claude, Ollama, OpenAI-compatible, OpenRouter, vLLM) | `backends/` |
| Hypothesis ledger, milestones, charter enforcement | `epistemics/` |
| Oracle resolution, instrumented evaluation, notebook execution | `evaluation/` |
| Delegation log, container runner, SLURM, telemetry, workspace VCS | `infra/` |
| Charter, f3dasm idioms, the handbook agents can consult | `knowledge/` |
| Paper search + corpus embedding | `literature/`, `agents/literature_tools/` |
| Node lifecycle, tool routing, both gates | `nodes/` |
| Live read-only run viewer (Starlette + SSE) | `viewer/` |
| Design rationale, one file per change | `internal/specs/` |

`runtime/run_diagram.py` renders the run's actual topology as SVG from the live
graph objects — the same anti-drift contract this map follows.
