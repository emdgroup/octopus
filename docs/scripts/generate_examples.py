"""Convert all example scripts into markdown files for inclusion in the docs."""

import logging
import os
import re
import subprocess
import sys
from itertools import chain
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

for example_script in chain.from_iterable([EXAMPLES_DIR.glob("*.py"), EXAMPLES_DIR.glob("*.ipynb")]):
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

    if example_script.suffix == ".py":
        # Convert python script to notebook first
        temp_notebook = TARGET_DIR / f"{example_script.stem}.ipynb"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupytext",
                "--to",
                "notebook",
                "--output",
                str(temp_notebook),
                str(example_script),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            _log.error(f"✗ Failed to convert {example_script} to notebook format.\n\tstderr:\n{proc.stderr}")
            proc.check_returncode()

        example_script = temp_notebook  # noqa: PLW2901

    # execute jupyter notebook and export to markdown
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            # TODO: add "--execute", in case we want to try including results
            "-y",
            "--to",
            "markdown",
            "--embed-images",
            "--output-dir",
            str(target_file.parent),
            "--output",
            str(target_file.stem),
            str(example_script),
        ],
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
        )

        # Make a regex that matches if any of our regexes match.
        ignored_patterns = "(" + ")|(".join(ignored_substrings) + ")"

        for line in content.splitlines():
            if re.search(ignored_patterns, line):
                continue

            line = line.replace("title: ", "# ")  # noqa: PLW2901

            wrapped_lines.append(line)

        lines = [line + "\n" for line in wrapped_lines]

    # Rewrite the file
    with open(target_file, "w", encoding="UTF-8") as markdown_file:
        markdown_file.writelines(lines)

    _log.debug(f"Converted {example_script.name} to {target_file.name}")
