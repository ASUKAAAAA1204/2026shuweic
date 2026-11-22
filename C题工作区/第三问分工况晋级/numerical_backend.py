"""Choose the numerical backend once per process, outside the frozen policy code."""
from functools import lru_cache
import os
import shutil


@lru_cache(maxsize=1)
def selected_backend():
    requested = os.environ.get("SHUWEICUP_Q3_BACKEND", "auto").lower()
    if requested not in {"auto", "cpp", "numpy"}:
        raise ValueError("SHUWEICUP_Q3_BACKEND must be auto, cpp or numpy")
    native = os.name == "nt" and shutil.which("g++") is not None
    if requested == "cpp" and not native:
        raise RuntimeError("The C++ backend requires Windows and MinGW g++ on PATH")
    if requested == "auto":
        return "cpp" if native else "numpy"
    return requested


@lru_cache(maxsize=1)
def _rolling():
    from numpy_kernels import rolling_numpy
    return rolling_numpy


@lru_cache(maxsize=1)
def _bellman():
    from numpy_kernels.bellman_numpy import ArrayBackend
    return ArrayBackend()


def configure(rolling, routes=None, bellman=None):
    """Bind kernel factories before constructing any policy or Bellman table."""
    if selected_backend() == "numpy":
        rolling.library = _rolling
        if routes is not None and not getattr(routes.beam_library, "_numpy_routes", False):
            original = routes.beam_library

            def library(stochastic=False):
                if stochastic:
                    from numpy_kernels import stochastic_numpy
                    return stochastic_numpy
                # Deterministic beam is an unpublished research candidate.
                return original(False)

            library._numpy_routes = True
            routes.beam_library = library
        if bellman is not None:
            bellman.library = _bellman
    return selected_backend()
