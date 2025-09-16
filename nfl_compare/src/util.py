from __future__ import annotations

from contextlib import contextmanager
import io
import sys


@contextmanager
def silence_stdout_stderr():
    """Temporarily silence prints to stdout and stderr (e.g., noisy dependency logs).

    Usage:
        with silence_stdout_stderr():
            call_noisy_function()
    """
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
