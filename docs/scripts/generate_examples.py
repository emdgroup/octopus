"""Convert all example scripts into markdown files for inclusion in the docs."""

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_log = logging.getLogger(Path(__file__).name)
_log.setLevel(logging.INFO)

if not _log.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(name)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    _log.addHandler(handler)

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
TARGET_DIR = Path(__file__).parent.parent / "examples"
EXCLUDED_FILES = {"__init__.py"}
FORCE = False

_log.debug(f"Converting all scripts in {EXAMPLES_DIR} to markdown files in {TARGET_DIR}")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

for example_script in EXAMPLES_DIR.glob("*.py"):
    if example_script.name in EXCLUDED_FILES:
        continue

    _log.info(f"Converting {example_script.name}")

    target_file = TARGET_DIR / f"{example_script.stem}.md"

    if target_file.exists():
        if not FORCE:
            _log.debug(f"Skipping existing file {target_file}")
            continue
        else:
            _log.debug(f"Removing existing file {target_file}")
            target_file.unlink()

    env = os.environ | {"ALWAYS_OVERWRITE_STUDY": "yes"}

    with tempfile.NamedTemporaryFile(suffix=".py") as temp:
        marimo_script = Path(temp.name)

        # TODO: rather convert to a jupyer notebook, execute and convert to markdown afterwards to also have results, etc.
        # convert file to marimo notebook (inplace - is a no-op if the file already is a marimo notebook)
        proc = subprocess.run(
            [sys.executable, "-m", "marimo", "convert", example_script, "-o", marimo_script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            _log.error(f"✗ Failed to convert {example_script} to marimo notebook format.\n\tstderr:\n{proc.stderr}")
            proc.check_returncode()

        if "File is already a valid marimo notebook." in proc.stdout:
            _log.debug(f"{example_script.name} is already a marimo notebook.")
            marimo_script = example_script

        # execute marimo notebook and export to markdown
        proc = subprocess.run(  # TODO: markdown conversion does not seem to run/store results
            [sys.executable, "-m", "marimo", "export", "md", marimo_script, "-o", target_file],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            _log.error(f"✗ Failed to export {example_script} to markdown format.\n\tstderr:\n{proc.stderr}")
            proc.check_returncode()

    # CLEANUP
    # We wrap lines which are too long as long as they do not contain a link.
    # To discover whether a line contains a link, we check if the string "]("
    # is contained.
    with open(target_file, encoding="UTF-8") as markdown_file:
        content = markdown_file.read()
        wrapped_lines = []
        ignored_substrings = (
            r"!\[svg\]",
            r"!\[png\]",
            r"<Figure size",
            r"it/s",
            r"s/it",
            r"^---$",
            r"^marimo-version:",
            r"^width: ",
        )

        # Make a regex that matches if any of our regexes match.
        ignored_patterns = "(" + ")|(".join(ignored_substrings) + ")"

        for line in content.splitlines():
            if re.search(ignored_patterns, line):
                continue

            line = line.replace("title: ", "# ")  # noqa: PLW2901
            line = line.replace("```python {.marimo}", '```py linenums="1"')  # noqa: PLW2901

            wrapped_lines.append(line)

        lines = [line + "\n" for line in wrapped_lines]

    # Rewrite the file
    with open(target_file, "w", encoding="UTF-8") as markdown_file:
        markdown_file.writelines(lines)

    _log.debug(f"Converted {example_script.name} to {target_file.name}")
