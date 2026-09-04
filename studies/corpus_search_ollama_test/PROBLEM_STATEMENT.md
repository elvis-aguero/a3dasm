# CorpusAdd/CorpusSearch verification on a local model (Ollama)

This is not a science problem. It exists solely to verify BACKLOG #32's fix
(`CorpusSearch`/`CorpusAdd`/`CorpusGetPaper` missing type annotations, which
broke tool-schema generation on OpenAI-compatible backends) via a real
`literature_reviewer` delegation running on a locally served model, not just
the unit tests already added.

## Task

Delegate to `literature_reviewer`:

1. Call `CorpusAdd` on `workspace/excerpt.md` (an excerpt of a real paper,
   "Drop rebound at low Weber number") with `title="Drop rebound at low Weber
   number"`, `year="2025"`.
2. Call `CorpusSearch` with `query="kinematic match model contact pressure"`,
   `top_k=3`, and report the paper(s) returned and their matched excerpt(s).
3. Call `CorpusGetPaper` on the paper id returned by step 2 and report its
   stored metadata.
4. Reply with a short summary of what all three calls returned. Do not
   fabricate results — quote exactly what the tools returned.

There is no pipeline, no evaluator, no deliverable notebook. Call `Done()`
once the summary above has been reported.
