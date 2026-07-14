# Agentic backlog — specs

Evidence-grounded specs for the backlog items in `../BACKLOG.md`. Each is written
from **primary evidence** (file:line citations + run IDs), follows **TDD**
(failing tests named first) and **DRY** (explicit reuse of existing helpers /
seams), and ends with a **KPI "done when"** (a measured outcome, not "it should
work" — per the no-KPIs-no-science rule).

Priority order (highest first) and how they compose:

| Spec | Item | Priority | Status |
|---|---|---|---|
| [01](01-reconcile-cancelled-completed.md) | Reconcile cancelled-but-completed delegations | **highest** | spec |
| [02](02-typed-escalation-comms.md) | Typed blocker/escalation comms (delegator levers) | high | spec |
| [03](03-problem-definer-agent.md) | ProblemDefinerAgent intake pre-pass | medium | spec |
| [06](06-stuck-delegation-detection.md) | Stuck-delegation detection + push | medium | **detection done; push+levers remain** |
| [04](04-single-notebook-deliverable.md) | Single `.ipynb` deliverable | low | spec |
| [05](05-slurm-execution-kb.md) | SLURM execution KB (+ shared-FS invariant) | lowest | spec |
| [07](07-duplicate-evaluation-detection.md) | Duplicate/redundant design-point evaluation detection | medium | **DONE (shipped simplified)** |

**Dependency graph (build order matters):**
- **01 + 02 + 06 are one cluster.** 06 *detects* a stuck/slow delegation; 02
  supplies the *actuators* (`grant_budget`/`abort`) to respond; 01 *reconciles*
  whatever a cancelled/aborted-but-completed delegation already stamped. Building
  02 first gives 01 a clean `abort` and 06 a real lever; build **02 → 01 → 06**.
- **03, 04, 05, 07 are independent** of that cluster and of each other.

Every spec separates the **mechanism claim** (tests pass) from the **behavioral
claim** (KPI improves on a re-run) and says which is which.
