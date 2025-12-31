"""Utility for creating the examples."""

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

from tqdm import tqdm

# TODO full rebuild option


def build_examples(destination_directory: Path, dummy: bool, remove_dir: bool, force: bool):
    """Create the documentation version of the examples files.

    Note that this deletes the destination directory if it already exists.

    Args:
        destination_directory: The destination directory.
        dummy: Only build a dummy version of the files.
        remove_dir: Remove the examples directory if it already exists.
        force: Force-build the steps, do not abort on errors.

    Raises:
        OSError: If the directory already exists but should not be removed.
        subprocess.CalledProcessError: If a subprocess call fails and force is not set.
    """
    # if the destination directory already exists it is deleted
    if destination_directory.is_dir():
        if remove_dir:
            shutil.rmtree(destination_directory)
        else:
            raise OSError("Destination directory exists but should not be removed.")

    # Copy the examples folder in the destination directory
    shutil.copytree("examples", destination_directory)

    # For the toctree of the top level example folder, we need to keep track of all
    # folders. We thus write the header here and populate it during the execution of the
    # examples
    ex_file = """# Examples
This page contains all examples provided with Octopus.
```{toctree}
"""

    try:
        # list all .py files in the directory that need to be converted
        excluded_files = {"__init__.py"}
        py_files = [f for f in destination_directory.glob("**/*.py") if f.name not in excluded_files]

        # Iterate through the individual example files
        for file in (pbar := tqdm(py_files, leave=False)):
            # Include the name of the file to the toctree
            # Format it by replacing underscores and capitalizing the words
            file_name = file.stem

            formatted = " ".join(word.capitalize() for word in file_name.split("_"))

            ex_file += formatted + f"<{file_name}>\n"

            # If we ignore the examples, we do not want to actually execute or convert
            # anything. Still, due to existing links, it is necessary to construct a
            # dummy file and then continue.
            if dummy:
                markdown_path = file.with_suffix(".md")
                # Rewrite the file
                with open(markdown_path, "w", encoding="UTF-8") as markdown_file:
                    markdown_file.writelines("# DUMMY FILE")
                continue

            # Set description for progress bar
            pbar.set_description(f"Progressing {file}")

            # Create the Markdown file:
            markdown_path = file.with_suffix(".md")

            # This is only done if we decide not to ignore the examples.
            # The creation of the files themselves and converting them to markdown still
            # happens since we need the files to check for link integrity.
            env = os.environ | {"PYTHONPATH": os.getcwd(), "ALWAYS_OVERWRITE_STUDY": "yes"}

            # convert file to marimo notebook (inplace - is a no-op if the file already is a marimo notebook)
            proc = subprocess.run(
                ["marimo", "convert", file, "-o", file],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                print(f"✗ Failed to convert {file} to marimo notebook format.")
                print(f"  stderr: {proc.stderr}")
                proc.check_returncode()

            # execute marimo notebook and export to markdown
            proc = subprocess.run(  # TODO: markdwon conversion does not seem to run/store results
                ["marimo", "export", "md", file, "-o", markdown_path],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                print(f"✗ Failed to export {file} to markdown format.")
                print(f"  stderr: {proc.stderr}")
                proc.check_returncode()

            # CLEANUP
            # We wrap lines which are too long as long as they do not contain a link.
            # To discover whether a line contains a link, we check if the string "]("
            # is contained.
            with open(markdown_path, encoding="UTF-8") as markdown_file:
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
                )

                # Make a regex that matches if any of our regexes match.
                ignored_patterns = "(" + ")|(".join(ignored_substrings) + ")"

                for line in content.splitlines():
                    # Skip formatter control lines so they don't appear in docs
                    if "fmt: off" in line or "fmt: on" in line:
                        continue

                    if re.search(ignored_patterns, line):
                        continue

                    line = line.replace("title: ", "# ")  # noqa: PLW2901

                    if len(line) > 88 and "](" not in line:
                        wrapped = textwrap.wrap(line, width=88)
                        wrapped_lines.extend(wrapped)
                    else:
                        wrapped_lines.append(line)

            # Add a manual new line to each of the lines
            lines = [line + "\n" for line in wrapped_lines]
            # Delete lines we do not want to have in our documentation
            # lines = [line for line in lines if "![svg]" not in line]
            # We check whether pre-built light and dark plots exist. If so, we append
            # corresponding lines to our markdown file for including them.
            # If not, we check if a single plot version exists and append it
            # regardless of light/dark mode.
            light_figure = Path(file_name + "_light.svg")
            dark_figure = Path(file_name + "_dark.svg")
            figure = Path(file_name + ".svg")
            if light_figure.is_file() and dark_figure.is_file():
                lines.append(f"```{{image}} {file_name}_light.svg\n")
                lines.append(":align: center\n")
                lines.append(":class: only-light\n")
                lines.append("```\n")
                lines.append(f"```{{image}} {file_name}_dark.svg\n")
                lines.append(":align: center\n")
                lines.append(":class: only-dark\n")
                lines.append("```\n")
            elif figure.is_file():
                lines.append(f"```{{image}} {file_name}.svg\n")
                lines.append(":align: center\n")
                lines.append("```\n")

            # Rewrite the file
            with open(markdown_path, "w", encoding="UTF-8") as markdown_file:
                markdown_file.writelines(lines)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while building the examples: {e}")
        if not force:
            raise e

    finally:
        # Write last line of top level toctree file and write the file
        ex_file += "```"
        with open(destination_directory / f"{destination_directory.name}.md", "w", encoding="UTF-8") as f:
            f.write(ex_file)

        # Remove remaining files and subdirectories from the destination directory
        # Remove any not markdown files
        for file in destination_directory.glob("**/*"):
            if file.is_file() and file.suffix not in (".md", ".svg"):
                file.unlink(missing_ok=True)
