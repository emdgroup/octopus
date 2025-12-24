import subprocess
import sys
from pathlib import Path


def run_marimo():
    """Run an interactive marimo session in the examples directory."""
    examples_dir = Path(__file__).parent.absolute()
    subprocess.run([sys.executable, "-m", "marimo", "edit", examples_dir], check=True)


if __name__ == "__main__":
    run_marimo()
