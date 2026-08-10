# Installation

Two things, in order: the Python package, and a way to actually talk to a
model. Both are one command each.

## 1. The package

Requires Python 3.10+. [f3dasm](https://github.com/bessagroup/f3dasm) comes
along automatically from PyPI.

```bash
pip install "a3dasm @ git+https://github.com/elvis-aguero/a3dasm.git"
```

(Not on PyPI yet, so it installs straight from the repository.)

## 2. A model to drive the agents

By default a3dasm talks to models through the **Claude CLI**, a separate
tool, not part of the `pip install` above.

```bash
npm install -g @anthropic-ai/claude-code
claude   # first run prompts you to log in (subscription or API key)
```

That's it: once `claude` works on its own from your terminal, a3dasm can use
it. No Node.js? Or already have an API key and want to skip the login prompt?

```bash
export ANTHROPIC_API_KEY=sk-...
```

Don't have Claude access at all, or want to use a different model? a3dasm
also drives Ollama, any OpenAI-compatible endpoint, and vLLM: see
[Customizing a run](customizing-a-run.md#the-available-backends).

## Optional extras

- `a3dasm[extra]` adds `docling` for layout-aware PDF parsing in the literature
  reviewer (pulls torch; excluded on Intel macOS).
- `a3dasm[docs]` installs the documentation toolchain.
- `a3dasm[tests]` installs the test toolchain.
- `a3dasm[dev]` installs pre-commit and ruff.

## Next

[Run the Quickstart](notebooks/quickstart.ipynb): a couple of minutes to
your first result.
