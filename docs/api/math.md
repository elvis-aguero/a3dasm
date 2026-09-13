# Symbolic derivations

`Workspace` is the vocabulary a derivation is written against: a sequential
list of typed steps, each either a real SymPy computation with a verdict
(`CONFIRMED` / `REFUTED` / `INCONCLUSIVE`) or an explicitly recorded
assumption (`ASSERTED`).

The split is the point. Anything SymPy can adjudicate is checked mechanically
and carries its verdict; anything it cannot — an ansatz, a physical claim, an
unverified relation — is recorded as an assumption rather than blurred into
the derivation. A reader of the rendered result can always tell which is
which.

This is what `MathExpertAgent` writes when a run needs an analytical result
rather than a measured one.

::: a3dasm.Workspace
