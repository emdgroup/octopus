#!/bin/bash
# Script to replicate GitHub Actions CI environment locally
# This helps test if changes will pass in CI before pushing

set -e  # Exit on error

echo "=========================================="
echo "Replicating GitHub Actions CI Environment"
echo "=========================================="
echo ""

# Store original directory
ORIGINAL_DIR=$(pwd)
PROJECT_ROOT="/home/ec2-user/octopus"

cd "$PROJECT_ROOT"

# Create a temporary test environment
TEST_ENV_DIR="/tmp/octopus_ci_test_$$"
echo "Creating fresh test environment at: $TEST_ENV_DIR"
mkdir -p "$TEST_ENV_DIR"

# Copy project to test directory (excluding .venv and other build artifacts)
# Note: We keep .git because setuptools-scm needs it for version detection
echo "Copying project files..."
rsync -av \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='*.egg-info' \
  --exclude='dist' \
  --exclude='build' \
  --exclude='studies' \
  --exclude='datasets_local' \
  --exclude='AutogluonModels' \
  "$PROJECT_ROOT/" "$TEST_ENV_DIR/"

cd "$TEST_ENV_DIR"

echo ""
echo "Installing dependencies with 'uv sync --extra test' (like GitHub Actions)..."
export SMOKE_TEST="true"  # Set the same env var as GitHub Actions
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

# Cleanup
echo ""
echo "Cleaning up test environment..."
cd "$ORIGINAL_DIR"
rm -rf "$TEST_ENV_DIR"

exit $TEST_RESULT
