import os
import subprocess
import sys
from pathlib import Path

import pytest

_examples_dir = Path(__file__).parent.parent / "examples"
_all_examples = sorted(
    [f for f in _examples_dir.glob("*.py") if f.name not in {"__init__.py", "main.py"}],
    key=lambda p: p.name,
)


_all_examples_basic = sorted([f for f in _all_examples if "basic_" in f.name], key=lambda p: p.name)
_all_examples_others = sorted(
    [f for f in _all_examples if "basic_" not in f.name],
    key=lambda p: p.name,
)


def run_example(example_path: Path):
    result = subprocess.run([sys.executable, str(example_path)], check=False, capture_output=True, text=True)

    try:
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        pytest.fail(f"""Example {example_path.name} failed with error: {e}

STDOUT:
{result.stdout}

STDERR:
{result.stderr}""")


@pytest.fixture(scope="module")
def temporary_working_dir(tmp_path_factory):
    """Fixture to create and switch to a temporary working directory."""
    temp_dir = tmp_path_factory.mktemp("temp_working_dir")
    oldCWD = os.getcwd()
    try:
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(oldCWD)


@pytest.mark.slow
@pytest.mark.usefixtures("temporary_working_dir")
@pytest.mark.parametrize("example_path", _all_examples_basic, ids=lambda p: p.name)
def test_basic_examples(example_path: Path):
    """Test that each basic example script runs without error.

    All examples run inside the same temporary working directory created by a fixture
    because the base* examples generate data files that are used by other examples.
    """
    run_example(example_path)


@pytest.mark.slow
@pytest.mark.order(after="test_basic_examples")
@pytest.mark.usefixtures("temporary_working_dir")
@pytest.mark.parametrize("example_path", _all_examples_others, ids=lambda p: p.name)
def test_example(example_path: Path):
    """Test that each other example script runs without error.

    All examples run inside the same temporary working directory created by a fixture
    because the base* examples generate data files that are used by other examples.
    """
    run_example(example_path)
