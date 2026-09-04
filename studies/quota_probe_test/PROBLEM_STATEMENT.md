# Quota probe — is Anthropic usage blocked at all right now?

This is not a science problem. It exists solely to test whether the account
can make ANY successful Claude API round trip right now, as cheaply as
possible, after a prior run (`supercompressible-material/20260903T233207`
on Oscar) died immediately with "You've hit your org's monthly spend
limit." All-Haiku, two nodes (strategizer + critic), run locally — no
Oscar, no SLURM, no local model, nothing that could itself fail and be
mistaken for the thing under test.

## Task

Say "OK" and call `Done()` immediately. There is nothing to solve, no
data, no pipeline, no deliverable. The only thing this run is checking is
whether the strategizer's turn and the critic's gate check both complete
without an API-level block. Whatever the critic decides (pass or reject)
is irrelevant to the purpose of this run — do not retry more than once if
rejected; a single completed round trip either way already answers the
question this run exists to ask.
