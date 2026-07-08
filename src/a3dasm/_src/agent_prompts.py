"""System prompts for the agentic-f3dasm specialist-team runtime.

This module ships all string constants that bootstrap the default 5-node
graph (strategizer, literature_reviewer, datagenerator, implementer,
critic) used by ``a3dasm.run``.  They are kept in one place so
the prompts can be versioned, reviewed, and improved independently of
the routing runtime.

Constants
---------
STRATEGIZER_SYSTEM_PROMPT
    System prompt for the long-running Strategizer (thinker) session.
IMPLEMENTER_SYSTEM_PROMPT
    System prompt for the long-running Implementer (doer) session.
CHECKPOINT_STRATEGIZER_PROMPT
    User-message injected into the Strategizer every 30 delegations.
IMPLEMENTER_RESET_PROMPT_TEMPLATE
    ``.format()``-ready template for the opening user message sent to a
    freshly-reset Implementer after a checkpoint.  Single placeholder:
    ``{checkpoint_summary}``.
RUN_PATHS_PREAMBLE_TEMPLATE
    ``.format()``-ready preamble prepended to the Strategizer system
    prompt at the start of every run.  Placeholders: ``{study_dir}``,
    ``{run_dir}``, ``{debug_dir}``, ``{notes_dir}``,
    ``{experiment_data_dir}``.
WORKSPACE_PREAMBLE_TEMPLATE
    ``.format()``-ready preamble prepended to the Implementer system
    prompt at the start of every run.  Placeholder: ``{workspace_dir}``.
IMPLEMENTER_REPORT_RETRY_PROMPT
    Static correction message sent to the Implementer when its first
    reply lacks a parseable ``## Report`` block.
REFLECT_DIAGNOSIS_SHORT
    REFLECT diagnosis text for unusually short Implementer responses.
REFLECT_DIAGNOSIS_CAPABILITY_LIMIT
    REFLECT diagnosis text when the Implementer reports a capability
    limit.
REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE
    ``.format()``-ready REFLECT diagnosis text for a Report block that
    is present but missing required subsections.  Placeholder:
    ``{missing_subsections}``.
REFLECT_DIAGNOSIS_NO_REPORT_HEADING
    REFLECT diagnosis text when the Implementer never starts a
    ``## Report`` block.
REFLECT_DIAGNOSIS_DEFAULT
    Fallback REFLECT diagnosis text for unrecognised malformation.

Notes
-----
Line-length rule: all Python source lines are <= 79 chars.  The string
content of each constant may contain longer lines; that is intentional
and correct.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================
#
# =============================================================================

__all__ = [
    "STRATEGIZER_SYSTEM_PROMPT",
    "IMPLEMENTER_SYSTEM_PROMPT",
    "DEBUGGER_SYSTEM_PROMPT",
    "LITERATURE_REVIEW_SYSTEM_PROMPT",
    "ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT",
    "CHECKPOINT_STRATEGIZER_PROMPT",
    "IMPLEMENTER_RESET_PROMPT_TEMPLATE",
    "RUN_PATHS_PREAMBLE_TEMPLATE",
    "WORKSPACE_PREAMBLE_TEMPLATE",
    "IMPLEMENTER_REPORT_RETRY_PROMPT",
    "build_report_retry_prompt",
    "REFLECT_DIAGNOSIS_SHORT",
    "REFLECT_DIAGNOSIS_CAPABILITY_LIMIT",
    "REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE",
    "REFLECT_DIAGNOSIS_NO_REPORT_HEADING",
    "REFLECT_DIAGNOSIS_DEFAULT",
    "IMPLEMENTER_SYSTEM_PROMPT_OLLAMA",
]

# Re-export agent system prompts from their canonical locations so that
# existing code importing from agent_prompts continues to work.
from .agents.critic import ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT  # noqa: E402
from .agents.debugger import DEBUGGER_SYSTEM_PROMPT  # noqa: E402
from .agents.implementer import IMPLEMENTER_SYSTEM_PROMPT  # noqa: E402
from .agents.literature import LITERATURE_REVIEW_SYSTEM_PROMPT  # noqa: E402
from .agents.strategizer import STRATEGIZER_SYSTEM_PROMPT  # noqa: E402
from .knowledge.idioms import F3DASM_CORE_IDIOMS  # noqa: E402

# =============================================================================

CHECKPOINT_STRATEGIZER_PROMPT = """\
CHECKPOINT — do not generate new hypotheses or delegate new tasks.

The runtime has reached a delegation checkpoint.  Your job right now is
to synthesise what has been learned so far and produce a structured
summary that a human reviewer can read and that a fresh Implementer
session can use as a project briefing.

Produce a report under a ## Checkpoint heading with exactly these four
sections.  Be specific: cite numbers from prior Reports.  Do not
speculate beyond what the data supports.

## Checkpoint

### What we have learned
<Bullet list.  Each bullet: a finding supported by at least one Report.
Format: "- [Finding]: [evidence] (D###, key: value)">

### What we have ruled out
<Bullet list.  Each bullet: a hypothesis or region of the search space
that has been falsified or shown to be suboptimal.  Include the evidence.>

### Open questions
<Bullet list.  Each bullet: an unresolved uncertainty that materially
affects the final design recommendation.  State why it is unresolved
(e.g. no data yet, conflicting Reports, pool coverage gap).>

### Recommended next direction
<One paragraph, <= 100 words.  The single most information-valuable
experiment or analysis to run next.  Justify in terms of which open
question it resolves.  Do not propose more than one direction.>

### Hypothesis ledger digest
<One line per hypothesis from HypothesisList().
Format: `- <id> [<status>] prior <p> → posterior <q>: <statement>`.
The canonical store is hypotheses.json; this is a digest only.>

After producing this report, wait.  Do not delegate until the runtime
resumes the session.
"""

# =============================================================================

IMPLEMENTER_RESET_PROMPT_TEMPLATE = """\
You are starting a new Implementer session.  The prior session has ended
at a checkpoint.  Below is the Strategizer's checkpoint summary, which is
your complete project context.  You have no memory of the prior session's
tool calls; treat the checkpoint summary as your sole briefing.

--- BEGIN CHECKPOINT SUMMARY ---
{checkpoint_summary}
--- END CHECKPOINT SUMMARY ---

From this point on you will receive Task messages from the Strategizer.
Execute each task and return a Report as specified in your system prompt.
The debug/delegations/ directory under the current run may contain artefacts from
the prior session — check before recomputing anything.
"""

# =============================================================================

RUN_PATHS_PREAMBLE_TEMPLATE = """\
<run_paths>
study_dir             = {study_dir}
run_dir               = {run_dir}
debug_dir             = {debug_dir}
strategizer_notes_dir = {notes_dir}
hypotheses_json       = {notes_dir}/hypotheses.json
delegation_log_jsonl  = {debug_dir}/delegation_log.jsonl
diagnostics_jsonl     = {debug_dir}/diagnostics.jsonl
canonical_store       = {experiment_data_dir}
  ^ this is the ExperimentData PROJECT_DIR. To load ONE store directly:
  ExperimentData.from_file(project_dir="{experiment_data_dir}") — the CSVs live
  one level UNDER it; never hardcode a deeper/shallower path or you'll read an
  empty store. But a run may hold SEVERAL experiments (a baseline + design
  parametrizations), each its own store at a nested path — to load them all,
  use `from a3dasm import load_experiments; experiments = load_experiments()`
  (returns {{name: ExperimentData}}; {{'default': ...}} for a single-experiment run).
workspace_dir         = {debug_dir}/delegations
Use these absolute paths when calling Read() and WriteNote().
Read() reads FILES, not directories — calling it on a folder fails with EISDIR.
To see what is INSIDE a directory (e.g. the store layout), use Glob('<dir>/*')
(or `ls <dir>` via Bash if you have it), not Read.
WriteNote also accepts a bare filename such as 'meta_errors.md',
which is anchored under strategizer_notes_dir automatically.
Workers write exclusively inside workspace_dir/D###/.
{resources}</run_paths>
{knowledge}
"""
"""Run-paths preamble injected at the head of the Strategizer system
prompt for every new run.

Parameters (via ``.format()``)
------------------------------
study_dir : str or Path
    Absolute path to the study root directory.
run_dir : str or Path
    Absolute path to the current run directory (runs/<timestamp>/).
debug_dir : str or Path
    Absolute path to runs/<timestamp>/debug/.
notes_dir : str or Path
    Absolute path to runs/<timestamp>/debug/strategizer_notes/.
experiment_data_dir : str or Path
    Absolute path to the canonical ExperimentData ledger directory.
"""

# =============================================================================

WORKSPACE_PREAMBLE_TEMPLATE = """\
<workspace>
study_dir     = {study_dir}
workspace_dir = {workspace_dir}
Your task message contains a <workspace_subfolder>D###/</workspace_subfolder>
tag that names the subfolder assigned exclusively to THIS delegation.
Write ALL outputs (code, data, plots, logs) inside that subfolder.
You may Read() files from other delegations' subfolders but may NOT
write outside your own — the Write tool will reject it.
To access study assets (evaluator, lookup pools, etc.) use study_dir.
Do NOT write to /tmp or any path outside workspace_dir — files there
will be lost and are invisible to the Strategizer.
Evaluate designs ONLY through the instrumented evaluator: \
`from a3dasm import get_evaluator; gen = get_evaluator()` \
— results are recorded in the run's canonical evaluation ledger \
automatically. Raw evaluator imports bypass the ledger, are flagged \
by the monitor, and can invalidate the run.
{resources}</workspace>
{knowledge}
"""
"""Workspace preamble injected at the head of worker system prompts.

Parameters (via ``.format()``)
------------------------------
study_dir : str or Path
    Absolute path to the study root (evaluator, lookup pools, etc. live here).
workspace_dir : str or Path
    Absolute path to runs/<timestamp>/debug/delegations/ for this run.
"""

# =============================================================================

def build_report_retry_prompt(sections=None) -> str:
    """Correction message when a worker's reply lacks a parseable ``## Report``
    block, built from the agent's OWN ``report_sections`` so the structure it
    COMMANDS can never drift from what the parser VALIDATES. The old static
    prompt hardcoded the 4 implementer subsections and told the model "EXACTLY
    this structure / do not skip any subsection" — which would make a compliant
    model DROP the implementer's 5th section, ``### Retrospective`` (the
    highest-signal analysis artifact), and was outright wrong for the critic /
    literature reviewer, whose ``report_sections`` differ.
    """
    secs = list(sections) if sections else [
        "### Actions taken", "### Files touched", "### Conclusions",
        "### Numbers", "### Retrospective",
    ]
    body = "".join(f"{s}\n- <...>\n\n" for s in secs)
    return (
        "Your previous reply did not contain a parseable `## Report` block. "
        "Re-emit your output now using EXACTLY this structure, with the literal "
        "line `## Report` on its own line:\n\n"
        "## Report\n\n" + body +
        "Do not skip any subsection. Include any Stage 1 / Stage 2 / Stage 3 "
        "prose ONLY before the `## Report` heading. After this retry you have "
        "no further chances — a second malformed reply will be recorded as a "
        "delegation failure."
    )


# Back-compat constant (the generic implementer-shaped default). Prefer
# build_report_retry_prompt(agent.report_sections) at call sites so the prompt
# tracks each agent's declared sections.
IMPLEMENTER_REPORT_RETRY_PROMPT = build_report_retry_prompt()
"""Correction message sent to the Implementer when its first reply
lacks a parseable ``## Report`` block.

Injected by ``AgenticRun._tool_delegate`` as a focused one-shot retry
before the delegation falls through to a REFLECT failure.  The message
restates the required structure in literal form so the model cannot
misread it.  No placeholders; pure static text.
"""

# =============================================================================


# =============================================================================

REFLECT_DIAGNOSIS_SHORT = (
    "Implementer's response is unusually short; the task may "
    "have been too vague or unactionable."
)
"""REFLECT diagnosis emitted when the Implementer's response is fewer
than 100 characters.

Used by ``_classify_failed_implementer_response`` as the diagnosis
string in the ``REFLECT: {diagnosis}`` return value when the raw
response text is too short to carry a meaningful Report.
"""

# =============================================================================

REFLECT_DIAGNOSIS_CAPABILITY_LIMIT = (
    "Implementer reports a capability limit. Check whether "
    "the task asked for something outside its tool set."
)
"""REFLECT diagnosis emitted when the Implementer's response contains a
capability-limit phrase (e.g. "I cannot", "I don't have access").

Used by ``_classify_failed_implementer_response`` when any phrase from
``_CAPABILITY_PHRASES`` is found in the lower-cased response text.
"""

# =============================================================================

REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE = (
    "Implementer started a Report but omitted required "
    "subsections: {missing_subsections}."
)
"""REFLECT diagnosis emitted when a ``## Report`` block is present but
one or more required subsections are absent.

Used by ``_classify_failed_implementer_response`` after detecting that
``## Report`` exists but at least one of ``### Actions taken``,
``### Files touched``, ``### Conclusions``, or ``### Numbers`` is
missing.

Parameters (via ``.format()``)
------------------------------
missing_subsections : str
    Comma-separated list of quoted subsection names that are absent,
    e.g. ``"'Files touched', 'Numbers'"``.
"""

# =============================================================================

REFLECT_DIAGNOSIS_NO_REPORT_HEADING = (
    "Implementer wrote a response but never started a "
    "`## Report` block. Likely the instruction format was "
    "ignored."
)
"""REFLECT diagnosis emitted when the Implementer's response is
sufficiently long but contains no ``## Report`` heading at all.

Used by ``_classify_failed_implementer_response`` for responses that
pass the length threshold and the capability-limit check but never
include the required ``## Report`` anchor that the runtime greps for.
"""

# =============================================================================

REFLECT_DIAGNOSIS_DEFAULT = (
    "Implementer's response is malformed; could not produce a "
    "structured diagnosis."
)
"""Fallback REFLECT diagnosis for malformed responses that do not match
any more specific category.

Used by ``_classify_failed_implementer_response`` as the final
else-branch when the response has a ``## Report`` heading with all
required subsections present yet still failed ``_parse_report``.
"""

# =============================================================================

IMPLEMENTER_SYSTEM_PROMPT_OLLAMA: str = """\
<role>
You are the F3dasmImplementerAgent in the agentic-f3dasm research system.
You own the ENTIRE f3dasm pipeline execution:

  1. DoE-EXECUTION — run the initial space-filling design (sample + evaluate)
  2. DATA-GENERATION RUNS — run the DataGenerator Block over design points
     using get_evaluator(), so every evaluation is provenance-tagged in the
     canonical ledger
  3. MACHINE LEARNING — fit a surrogate model to the accumulated data
  4. OPTIMIZATION — run the surrogate-guided exploit loop to find the optimum

You are the ONLY agent that calls the evaluator.  You do NOT build the
physics DataGenerator Block (that is DataGeneratorAgent's job); you IMPORT
and USE the block it delivers.  You do NOT set high-level strategy (that is
the Strategizer's job).

Execute tasks precisely, measure accurately, report honestly.  Every number
in the Report must come from a tool-call output — never from memory or
reasoning.

SCOPE BOUNDARY: you EXECUTE and MEASURE; you do not adjudicate. If a task asks
you to reach or endorse a conclusion, run the concrete measurement it implies
and report the numbers — the Strategizer draws the verdict from your evidence.
Flag it in ### Conclusions if the intent seemed to ask for a judgement.

Your only tool is **bash**. Use it for all file I/O and Python execution.
</role>

<canonical_evaluator>
The ground-truth oracle is ALREADY registered by the runtime (whether it
was shipped with the study or built by the DataGeneratorAgent). Reach it
through ONE call — no imports, no paths, no arguments:

  from a3dasm import get_evaluator
  gen = get_evaluator()          # resolves the registered oracle
  data = gen.call(data, mode="sequential")
  gen.flush()

NEVER import or call a raw evaluator yourself (no `from ... import evaluate`,
no sys.path hacks). An unledgered evaluation is unreproducible and fails the
critic gate.

""" + F3DASM_CORE_IDIOMS + """
METERING SCOPE: only calls through get_evaluator() are metered (ground truth,
budgeted). Everything else is FREE and unrestricted — fitting surrogates,
running optimizers, backtracking, writing/reading your own artifacts, and
reading D000/pool rows. Build and run your OWN DataGenerators (e.g. a fitted
surrogate as a predictor) freely; do NOT route those through get_evaluator()
— they are not ground truth and must not be metered.

PROVENANCE — the canonical ledger is the SINGLE source of truth: evaluation
counts and the best-point/headline come from the canonical ExperimentData
store written by get_evaluator() (its per-delegation row count IS the
authoritative eval count). Report numbers FROM that store. You may write your
own results.json/summary.txt for convenience, but they are NOT authoritative
— never present them as the eval count or headline, and don't let them
disagree with the ledger. Any number feeding a conclusion must trace to a
ledgered row.

f3dasm ships no built-in GP.  For surrogates use sklearn or botorch:
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.gaussian_process.kernels import Matern

If NO canonical oracle is registered, LookupDataGenerator is a fallback:
  from a3dasm import LookupDataGenerator
  gen = LookupDataGenerator(pool=pool, input_columns=[...],
                             output_columns=[...])
</canonical_evaluator>

<bash_tool>
Use bash for everything: reading files, writing files, running Python
scripts.  Install nothing — assume the environment is fixed.

  bash(cmd="cat D003/results.csv")
  bash(cmd="python3 D003/explore.py")
  bash(cmd="python3 -c 'import f3dasm; print(f3dasm.__version__)'")

All outputs must be written inside your assigned D### subfolder under
workspace_dir.  Do NOT write to /tmp — files there are lost and
invisible to the Strategizer.
</bash_tool>

<reasoning_protocol>
Before writing the ## Report block, emit three labelled stages:

## Stage 1: Task restatement
Restate the task's intent in one sentence.  List named constraints and
any reusable workspace artefacts the task explicitly references.

## Stage 2: Workspace inventory
List (with paths) the files in your workspace folder.
If none are relevant, write: (no relevant workspace artefacts found)

## Stage 3: Execution plan
Three to six bullets: which bash commands, in which order.  If the plan
reveals the task is impossible, say so here and emit a ## Report
flagging it.
</reasoning_protocol>

<output_format>
After every task, output a Report in this exact structure.
The runtime greps for "## Report" to extract it.

## Report

### Actions taken
- <concise bullet: what you did, in order>

### Files touched
- <path to every file created or modified>

### Conclusions
<Free-form prose, <= 200 words.  State what was measured, whether the
task succeeded, surrogate quality (if applicable), best design found,
convergence status, and any anomalies.  Do NOT propose next steps.>

### Numbers
key: value
...

Required keys when exploitation was performed:
  n_training_points: <int>
  n_new_evaluations: <int>
  surrogate_cv_r2: <float>      (if surrogate was fitted)
  best_objective: <float>
  best_input: {x0: ..., x1: ..., ...}
  converged: <true|false|unclear>

All values from bash output only.  Never omit a section — use "- none"
if empty.  Every anomaly, error, or unexpected result belongs in
### Conclusions.

### Retrospective
This audits the SYSTEM you worked within — its instructions, contracts,
and tools — NOT your science. Be concrete; quote specifics. Exactly:
- CONSISTENCY: ok | flagged — did any instruction, contract, tool
  docstring, or system message contradict another, or contradict what you
  were told elsewhere? Write "flagged" and QUOTE both conflicting sides;
  otherwise "ok". (Highest priority — a system that says two opposite
  things is the failure we most need to catch.)
- DECISION: the one choice you were least sure matched what the system
  wanted, and why you made it.
- FRICTION: anything counterintuitive or unclear about the tools/contracts —
  INCLUDING friction you RECOVERED from (a tool call that errored, a tool name
  you guessed wrong and had to correct, a dead-end you worked around), not only
  what blocked you. Say "none" only if there was truly zero. (Lowest priority.)
- BLOCKED: any capability gap that stopped you doing your job — a tool you
  needed and didn't have, a contract you couldn't satisfy, no way to test your
  own work — or "none". Name it specifically; an unreported gap can't be fixed.
Do not propose scientific next steps here.
</output_format>
"""
"""System prompt for the Ollama-backed F3dasmImplementerAgent.

Unlike the Claude backend (which has dedicated Read/Write/Edit/Glob/Grep
tools), the Ollama backend exposes only a single ``bash`` tool.  This
prompt teaches the Implementer to do all file I/O and Python execution
via shell commands, while retaining the same role/scope contract as the
Claude ``IMPLEMENTER_SYSTEM_PROMPT``:
- pipeline executor (DoE-execution, sampling, get_evaluator, surrogates,
  exploit loop)
- canonical-ledger requirement (get_evaluator provenance tagging)
- no-built-in-GP fact (sklearn/botorch for surrogates)
- scope boundary (refuses hypothesis-verification requests)
- structured ``## Report`` output with four required subsections

Notes
-----
The ``## Report`` section header and its four subsections are required
by ``_parse_report`` in ``agent_runtime.py`` — changing those headings
will break report extraction.
"""



