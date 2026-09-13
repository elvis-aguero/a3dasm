# Running a study

`AgenticRun` is the entry point. It reads `PROBLEM_STATEMENT.md` from the study
directory, builds the agent graph, and runs the loop to a gated deliverable.

::: a3dasm.AgenticRun

::: a3dasm.AgenticRunError

::: a3dasm.DEFAULT_MODEL

## Using a run as an f3dasm optimizer

`AgenticOptimizerAdapter` wraps a whole agentic run behind the standard
f3dasm `Optimizer` interface, so it can be dropped in anywhere a regular
optimizer is used.

::: a3dasm.AgenticOptimizerAdapter
