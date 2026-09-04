"""``python -m a3dasm.viewer`` — see ``a3dasm._src.viewer.__main__`` for the
real CLI implementation."""
from __future__ import annotations

import sys

from .._src.viewer.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
