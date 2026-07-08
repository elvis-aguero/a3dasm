# Installation

a3dasm requires Python 3.10 or newer and depends on
[f3dasm](https://github.com/bessagroup/f3dasm).

While the repository is private, install from Git:

```bash
pip install "a3dasm @ git+ssh://git@github.com/bessagroup/a3dasm.git@v0.1.0"
```

## Optional extras

- `a3dasm[extra]` adds `docling` for layout-aware PDF parsing in the literature
  reviewer (pulls torch; excluded on Intel macOS).
- `a3dasm[docs]` installs the documentation toolchain.
- `a3dasm[tests]` installs the test toolchain.
- `a3dasm[dev]` installs pre-commit and ruff.

## Backends

a3dasm can drive the agents with the Claude CLI (default), an OpenAI-compatible
endpoint, Ollama, OpenRouter, or a vLLM server. See the feature catalog for the
backend matrix.
