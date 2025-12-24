"""CLI entry point for Octopus."""

import argparse
import os
import subprocess
import sys
from itertools import chain
from pathlib import Path


def _get_example_files(path: Path):
    try:
        return [file.relative_to(path) for file in chain.from_iterable([path.glob("*.py"), path.glob("*.ipynb")])]
    except FileNotFoundError as e:
        raise RuntimeError("Examples directory not found.") from e


EXAMPLES_DIR = Path(__file__).parent.parent.absolute() / "examples"
EXAMPLES_FILES = _get_example_files(EXAMPLES_DIR)


def run_jupyter_notebook(target: Path):
    """Run a Jupyter notebook server for the specified target path."""
    # Jupyter has no proper notion of the current working directory.
    # Thus we inject a proper default location for the studies directory.
    env = os.environ | {"STUDIES_PATH": str(Path.cwd() / "studies")}

    try:
        subprocess.run([sys.executable, "-m", "jupyter", "notebook", str(target)], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(
            "Failed to open Jupyter notebook. "
            "Please ensure Jupyter is installed (e.g. 'pip install jupyter') and "
            "that the 'jupyter' command is available on your PATH.\n"
            f"Underlying error: {e}",
            file=sys.stderr,
        )
        sys.exit(e.returncode or 1)


def open_all_examples_in_jupyter():
    """Open the examples directory in a Jupyter notebook server."""
    run_jupyter_notebook(EXAMPLES_DIR)


def open_example_in_jupyter(which: str):
    """Open a specific example in a Jupyter notebook server."""
    try:
        example_number = int(which)
    except ValueError:
        example_path = EXAMPLES_DIR / which
    else:
        if 0 <= example_number < len(EXAMPLES_FILES):
            example_path = EXAMPLES_DIR / EXAMPLES_FILES[example_number]
        else:
            print(f"Example number {example_number} is out of range.")
            return

    if example_path.is_file():
        run_jupyter_notebook(example_path)
    else:
        print(f"Example file {example_path} does not exist.")


def list_examples():
    """List all example files in the examples directory."""
    print("Available example scripts/notebooks:")
    for ifile, file in enumerate(EXAMPLES_FILES):
        print(f"\t({ifile:2d}) - {file}")


def cli_main(argv=None):
    """Main cli function."""
    parser = argparse.ArgumentParser(prog="octopus", description="Octopus CLI")
    subparsers = parser.add_subparsers(dest="command")

    examples_parser = subparsers.add_parser("examples", help="Interact with example notebooks/scripts")
    examples_parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Example index or filename. Omit to open the examples directory.",
    )
    examples_parser.add_argument(
        "--list",
        action="store_true",
        help="List available examples",
    )

    args = parser.parse_args(argv)

    if args.command == "examples":
        if args.list:
            list_examples()
        elif args.target:
            open_example_in_jupyter(args.target)
        else:
            open_all_examples_in_jupyter()
    else:
        parser.print_help()


if __name__ == "__main__":
    cli_main()
