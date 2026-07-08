"""The scientific-method charter — the single, citable source of truth for how
hypotheses are tested and labelled across the agentic system.

╔══════════════════════════════════════════════════════════════════════════╗
║  READ THIS ENTIRE FILE BEFORE CHANGING ONE WORD OF IT.                     ║
║                                                                            ║
║  The clauses below are MUTUALLY REFERENTIAL and the text demands           ║
║  self-consistency in its epistemics. A status routed one way in §3 must be ║
║  defined the same way in §5 and §6; "adequate" in §2 is the hinge the      ║
║  whole of §3 turns on. Editing one clause in isolation — the failure mode  ║
║  of skimming a file and patching the paragraph you landed on — has already ║
║  put a contradiction in here once (a survived test routed to INCONCLUSIVE  ║
║  in §3 while §5 called the same outcome SUPPORTED). Do NOT repeat it.       ║
║                                                                            ║
║  This is especially a warning to AI editors, who are notorious for reading ║
║  a fragment of a file and modifying it confidently without holding the     ║
║  whole in view. For THIS file that is unacceptable: an internally          ║
║  inconsistent charter silently corrupts every hypothesis verdict in every  ║
║  run, and both adjudicating agents will cite the same broken words at each ║
║  other forever.                                                            ║
║                                                                            ║
║  Before you touch it: (1) read all six clauses; (2) trace every status     ║
║  (OPEN / SUPPORTED / FALSIFIED / INCONCLUSIVE) through §2→§3→§5→§6 and      ║
║  confirm each is produced and consumed consistently; (3) confirm no clause  ║
║  reintroduces a procedural artifact (a flag, a tool name) AS an epistemic   ║
║  criterion — adequacy is a property of a test's severity, never of a       ║
║  label; (4) update tests/agentic/test_charter.py, which pins this wording. ║
╚══════════════════════════════════════════════════════════════════════════╝

``FALSIFICATION_CHARTER`` is the ONE place the Popperian rules live (DRY). It is
injected verbatim into the strategizer and critic system prompts at compile time
(a stable prefix → cached by the SDK, so ~no per-turn cost), and is reachable by
workers on demand through ConsultHandbook. Because both adjudicating nodes are
given the *identical* numbered text, either may cite a clause ("Charter §3") in a
disagreement and the other defers to the same words — no paraphrase drift, no
multi-round negotiation over what falsification means.

Do not restate these rules elsewhere; reference the clause number instead.
``tests/agentic/test_charter.py`` pins the injection and the wording.
"""

FALSIFICATION_CHARTER = """\
SCIENTIFIC-METHOD CHARTER — the shared, binding contract for testing and
labelling hypotheses. It is identical in the strategizer's and the critic's
instructions: cite a clause by number ("Charter §3") and the other party
defers to it.

§1  A hypothesis is ONE falsifiable claim carrying a registered prediction —
    the observable whose occurrence would refute the claim.

§2  ATTEMPT and VERDICT are distinct. Before a hypothesis may be closed, an
    adequate falsification ATTEMPT must have been made — a SEVERE test: one that
    genuinely probes the registered prediction and could have refuted the claim
    had it been false (a token probe is not adequate). Adequacy is a property of
    the test's SEVERITY, never of a label: tag such a delegation
    is_falsification_attempt so the attempt is auditable, but the tag only
    RECORDS the attempt — it cannot make a weak test adequate, nor a severe
    untagged one inadequate. The VERDICT is a separate judgement that follows
    the test's OUTCOME — never the mere fact that an attempt was run.
    For a prediction whose refutation turns on FINDING an instance — an
    achievement or existence claim ("some design in this space reaches X"), or
    its negation ("no design reaches X") — severity means the search had the
    POWER to find that instance had it existed; concretely, that had a
    qualifying instance existed in the space the claim ranges over, the search
    would very probably have found it. That is the whole test. HOW you argue the
    search had that power is open and judged on its merits — adequate coverage
    of the space (dense/near-exhaustive sampling, credible in low dimension), a
    guiding surrogate that predicts the claim's OWN observable above chance
    (needed as dimension grows, where coverage alone cannot suffice), a
    theoretical bound, or a combination. These are examples, not a checklist:
    no single one is mandatory, and a surrogate for a DIFFERENT quantity never
    counts (it does not measure the claim). A search that merely stopped
    improving, or whose only power-argument the work itself undercuts (e.g. a
    surrogate it reports as near-chance with no coverage argument to stand on),
    is an INADEQUATE test of such a claim and routes to INCONCLUSIVE under §3 —
    failing to find a better instance is not
    the same as showing none exists.

§3  A hypothesis is FALSIFIED if and only if an ADEQUATE test (§2) of its
    registered prediction yields a contradiction. Concretely:
      - adequate test, prediction contradicted     -> FALSIFIED (you may not
        decline the verdict to protect a favoured claim);
      - adequate test, prediction NOT contradicted -> the hypothesis SURVIVED
        and is therefore SUPPORTED (§5) — corroborated, not proven; it is NOT
        falsified, even if other evidence makes it look wrong;
      - inadequate or confounded test              -> INCONCLUSIVE. A
        contradiction from a flawed test indicts the test, not the hypothesis
        (Duhem–Quine).
    INCONCLUSIVE is reserved for an inadequate test; a hypothesis that has had
    no adequate test yet is simply OPEN.

§4  No moving the goalposts. A FALSIFIED verdict must rest on the contradiction
    of the SAME prediction that was registered — not a different, post-hoc
    observation chosen after seeing the data (the Texas-sharpshooter fallacy).
    If the registered prediction was the wrong test, revise it explicitly and
    run a new adequate test; do not reinterpret the old result.

§5  SUPPORTED is corroboration, not proof. It means the hypothesis survived at
    least one adequate attempt to refute it (§2–§3). You never "confirm" a
    hypothesis; you only fail to falsify it.

§6  The four statuses are the only ones: OPEN, SUPPORTED, FALSIFIED,
    INCONCLUSIVE. OPEN = no adequate test yet; the other three are closing
    statuses and must cite a real delegation ID plus a CONCRETE result from
    that delegation that bears on the registered prediction — a measurement, a
    category, a pass/fail, or a comparison (whatever form the prediction takes;
    not all evidence is numeric — a qualitative observation that contradicts a
    risky prediction falsifies it just as a black swan refutes "all swans are
    white"). What is forbidden is closing on prose or vibes: the cited result
    must be specific and checkable against the prediction.
"""
