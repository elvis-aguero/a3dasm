"""One-time setup: generate and compile the 3-D black-box evaluator.

Run once from the study root before starting the agentic run:

    python generate_evaluator.py

Produces workspace/evaluator.so (Linux) or workspace/evaluator.dylib
(macOS) and workspace/evaluator.py.  The C source is written, compiled,
and immediately deleted — the binary is the only persistent record of
the function.

Design
------
A sum of negative Gaussian wells over [-5, 5]^3:

    f(x) = -sum_i w_i * exp(-||x - c_i||^2 / (2 sigma_i^2))

One designated *global* well has weight 1.0; the rest are shallower
"distractor" wells whose depths sit just below the global one, so a greedy
local search is easily trapped in a near-global local minimum.  The whole
landscape is then **scaled so its true global minimum is exactly -1.0** —
the optimum location is randomised in the domain (seeded) and is NOT
exposed to the solver.  3-D keeps it tractable (10-20 min) while the small
global-vs-distractor depth gap keeps it deceptive.
"""

#                                                                       Modules
# =============================================================================

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

#                                                               Study constants
# =============================================================================

_SEED = 20260616
_DIM = 3
_BOUNDS = (-5.0, 5.0)
_N_DISTRACTORS = 12
_SIGMA_BOUNDS = (0.5, 0.9)
_GLOBAL_SIGMA = 0.7
_DISTRACTOR_WEIGHT_BOUNDS = (0.40, 0.85)   # all strictly below the global 1.0
_MIN_SEPARATION = 2.5                       # keep wells from merging into one basin

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# -----------------------------------------------------------------------------
# Parameter generation
# -----------------------------------------------------------------------------

def _sample_separated_centers(rng, n: int) -> np.ndarray:
    """n+1 centers in the domain, pairwise >= _MIN_SEPARATION apart.

    Index 0 is the global well; 1.. are distractors.
    """
    lo, hi = _BOUNDS
    centers: list[np.ndarray] = []
    while len(centers) < n + 1:
        cand = rng.uniform(lo, hi, _DIM)
        if all(np.linalg.norm(cand - c) >= _MIN_SEPARATION for c in centers):
            centers.append(cand)
    return np.asarray(centers)


def _generate_params():
    rng = np.random.default_rng(_SEED)
    centers = _sample_separated_centers(rng, _N_DISTRACTORS)
    weights = np.empty(_N_DISTRACTORS + 1)
    sigmas = np.empty(_N_DISTRACTORS + 1)
    weights[0] = 1.0                                   # global well
    sigmas[0] = _GLOBAL_SIGMA
    weights[1:] = rng.uniform(*_DISTRACTOR_WEIGHT_BOUNDS, _N_DISTRACTORS)
    sigmas[1:] = rng.uniform(*_SIGMA_BOUNDS, _N_DISTRACTORS)
    return centers, weights, sigmas


# -----------------------------------------------------------------------------
# Reference landscape + global-minimum search (for exact -1.0 normalisation)
# -----------------------------------------------------------------------------

def _f_raw(X, centers, weights, sigmas):
    """Vectorised raw landscape. X: (..., DIM) -> (...,)."""
    X = np.asarray(X, dtype=float)
    diff = X[..., None, :] - centers           # (..., n_wells, DIM)
    sq = np.sum(diff * diff, axis=-1)          # (..., n_wells)
    contrib = weights * np.exp(-sq / (2.0 * sigmas * sigmas))
    return -np.sum(contrib, axis=-1)


def _find_global_min(centers, weights, sigmas):
    """Locate the true global minimum by coarse grid + nested refinement.

    No scipy dependency: a shrinking-box grid search converges quickly on a
    smooth Gaussian landscape and gives the min value to ~1e-9.
    """
    lo, hi = _BOUNDS
    # Coarse grid over the whole domain.
    g = np.linspace(lo, hi, 41)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, _DIM)
    vals = _f_raw(grid, centers, weights, sigmas)
    best_i = int(np.argmin(vals))
    best_x = grid[best_i].copy()

    # Nested refinement: shrink a box around the incumbent and re-grid.
    half = (hi - lo) / 40.0
    for _ in range(40):
        axes = [np.linspace(best_x[d] - half, best_x[d] + half, 11)
                for d in range(_DIM)]
        sub = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, _DIM)
        sub = np.clip(sub, lo, hi)
        v = _f_raw(sub, centers, weights, sigmas)
        j = int(np.argmin(v))
        best_x = sub[j].copy()
        half *= 0.5
    return best_x, float(_f_raw(best_x, centers, weights, sigmas))


# -----------------------------------------------------------------------------
# C source construction
# -----------------------------------------------------------------------------

def _c_literal(v: float) -> str:
    return f"{v:.17g}"


def _c_array_1d(name: str, values) -> str:
    vals = ", ".join(_c_literal(v) for v in values)
    return f"static const double {name}[{len(values)}] = {{{vals}}};"


def _c_array_2d(name: str, rows) -> str:
    n, m = len(rows), len(rows[0])
    inner = ",\n    ".join(
        "{" + ", ".join(_c_literal(v) for v in row) + "}"
        for row in rows
    )
    return f"static const double {name}[{n}][{m}] = {{\n    {inner}\n}};"


def _build_c_source(centers, weights, sigmas, scale) -> str:
    n_wells = len(weights)
    return textwrap.dedent(f"""\
        #include <math.h>
        #include <stddef.h>

        #define N_WELLS {n_wells}
        #define DIM     {_DIM}

        {_c_array_2d("centers", centers)}
        {_c_array_1d("weights", weights)}
        {_c_array_1d("sigmas",  sigmas)}
        static const double SCALE = {_c_literal(scale)};

        double evaluate(const double *x) {{
            double result = 0.0;
            int i, d;
            for (i = 0; i < N_WELLS; i++) {{
                double sq_dist = 0.0;
                for (d = 0; d < DIM; d++) {{
                    double diff = x[d] - centers[i][d];
                    sq_dist += diff * diff;
                }}
                result -= weights[i]
                          * exp(-sq_dist
                                / (2.0 * sigmas[i] * sigmas[i]));
            }}
            return SCALE * result;
        }}
    """)


# -----------------------------------------------------------------------------
# Python wrapper generation
# -----------------------------------------------------------------------------

def _write_wrapper(workspace: Path) -> None:
    src = textwrap.dedent(f"""\
        \"\"\"ctypes wrapper for the compiled 3-D black-box evaluator.\"\"\"

        import ctypes
        from pathlib import Path

        _here = Path(__file__).parent
        _candidates = ("evaluator.dylib", "evaluator.so")
        _lib = None
        for _name in _candidates:
            _p = _here / _name
            if _p.exists():
                _lib = ctypes.CDLL(str(_p))
                break
        if _lib is None:
            raise FileNotFoundError(
                f"No compiled evaluator found in {{_here}}. "
                "Run generate_evaluator.py first."
            )

        _lib.evaluate.restype = ctypes.c_double
        _lib.evaluate.argtypes = [ctypes.POINTER(ctypes.c_double)]


        def evaluate(x):
            \"\"\"Evaluate the black-box function at x.

            Parameters
            ----------
            x : sequence of {_DIM} float
                Point in [-5, 5]^{_DIM}.

            Returns
            -------
            float
                Objective value (minimise).
            \"\"\"
            arr = (ctypes.c_double * {_DIM})(*x)
            return float(_lib.evaluate(arr))


        def evaluate_kw(**kwargs):
            \"\"\"Evaluate via keyword arguments x1, x2, x3.\"\"\"
            x = [kwargs[f"x{{i}}"] for i in range(1, {_DIM} + 1)]
            return evaluate(x)
    """)
    (workspace / "evaluator.py").write_text(src)


# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------

def _compile(c_path: Path, workspace: Path) -> Path:
    if sys.platform == "darwin":
        lib_name = "evaluator.dylib"
        cmd = [
            "gcc", "-O2", "-dynamiclib",
            "-o", str(workspace / lib_name),
            str(c_path), "-lm",
        ]
    else:
        lib_name = "evaluator.so"
        cmd = [
            "gcc", "-O2", "-shared", "-fPIC",
            "-o", str(workspace / lib_name),
            str(c_path), "-lm",
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    c_path.unlink()

    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed.\n\nstderr:\n{result.stderr}"
        )

    return workspace / lib_name


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    # This generator lives OUTSIDE the study dir (agents must not read the
    # secret it embeds — the optimum and well structure). It writes the oracle
    # artifacts into the study's workspace/.
    study_root = Path(__file__).resolve().parent.parent / "agentic_black_box_3d"
    workspace = study_root / "workspace"
    workspace.mkdir(exist_ok=True)

    centers, weights, sigmas = _generate_params()

    # True global minimum of the raw landscape, then scale so it is exactly -1.
    x_opt, m_raw = _find_global_min(centers, weights, sigmas)
    scale = -1.0 / m_raw                       # m_raw < 0  ->  scale > 0

    c_source = _build_c_source(centers, weights, sigmas, scale)
    c_path = workspace / "_evaluator_src.c"
    c_path.write_text(c_source)
    lib_path = _compile(c_path, workspace)
    _write_wrapper(workspace)

    # Sanity: the compiled, scaled optimum value should be -1.0.
    sys.path.insert(0, str(workspace))
    import importlib
    ev = importlib.import_module("evaluator")
    importlib.reload(ev)
    f_opt = ev.evaluate(list(x_opt))

    print(f"Compiled  : {lib_path}")
    print(f"Wrapper   : {workspace / 'evaluator.py'}")
    print(f"Seed      : {_SEED}   wells: {len(weights)} (1 global + "
          f"{_N_DISTRACTORS} distractors)")
    print(f"Optimum x : ({', '.join(f'{v:.4f}' for v in x_opt)})   "
          "[hidden from the solver]")
    print(f"f(x_opt)  : {f_opt:.10f}   (target -1.0)")
    print()
    print("Study is ready. Start the agentic run with:  python run.py")


if __name__ == "__main__":
    main()
