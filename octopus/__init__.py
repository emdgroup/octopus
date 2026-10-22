"""Octopus - AutoML for small Datasets."""

import os
import sys

if not sys.version_info >= (3, 12):
    raise RuntimeError("Minimum required Python version is 3.12")

# Ray detects drivers started via `uv run` and launches every worker with its own
# `uv run` inside a copy of the working directory. Since Ray 2.56 that copy excludes
# `.git`, so building octopus from a source checkout fails (setuptools-scm cannot
# determine the version) and the workers never start. Workers should instead reuse
# the driver's interpreter. Ray reads this variable on import, so it must be set
# before `ray` is imported. setdefault keeps an explicit user setting intact.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

del os, sys
