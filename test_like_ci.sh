#!/bin/bash

# Script to replicate GitHub Actions CI environment locally
# This helps test if changes will pass in CI before pushing

set -e  # Exit on error

echo "=========================================="
echo "Replicating GitHub Actions CI Environment"
echo "=========================================="
echo ""

# get the directory where this script resides (i.e. the repository root)
PROJECT_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd "$PROJECT_ROOT"

# Create a temporary test environment
TEST_ENV_DIR="/tmp/octopus_ci_test_$RANDOM"
export UV_PROJECT_ENVIRONMENT="${TEST_ENV_DIR}/.venv"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OCTOPUS="0.1.dev+`git describe --tags --always`"

echo "Creating fresh test environment at: $TEST_ENV_DIR"
mkdir -p "$TEST_ENV_DIR"

# Copy project to test directory (excluding .venv and other build artifacts)
# Note: We keep .git because setuptools-scm needs it for version detection
echo "Copying project files..."
rsync -av \
  --filter=':- .gitignore' \
  --exclude='.git' \
  "$PROJECT_ROOT/" "$TEST_ENV_DIR/"


pushd "$TEST_ENV_DIR"

echo ""
echo "Installing dependencies with 'uv sync --extra test' (like GitHub Actions)..."
# Set the same env var as GitHub Actions
uv sync --extra test

echo ""
echo "Running pytest with coverage (like GitHub Actions)..."
echo "=========================================="
uv run pytest --cov=octopus

TEST_RESULT=$?

echo ""
echo "=========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ Tests PASSED - your changes should pass in CI!"
else
    echo "❌ Tests FAILED - fix these issues before pushing to CI"
fi
echo "=========================================="

popd

# Cleanup
echo ""
echo "Cleaning up test environment..."
rm -rf "$TEST_ENV_DIR"

popd
exit $TEST_RESULT
