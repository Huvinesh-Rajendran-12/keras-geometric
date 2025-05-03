#!/bin/bash
# Script to verify editable installation works correctly using uv

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first."
    echo "Installation instructions: https://github.com/astral-sh/uv"
    exit 1
fi

# Create a temporary virtual environment using uv
echo "Creating temporary virtual environment with uv..."
uv venv test_venv

# Activate the virtual environment
source test_venv/bin/activate

# Install dependencies required for editable install with uv
echo "Installing build dependencies with uv..."
uv pip install hatchling hatch-vcs editables

# Install backend dependencies required for keras-geometric
echo "Installing backend dependencies..."
uv pip install tensorflow

# Install the package in editable mode using uv
echo "Installing keras-geometric in editable mode with uv..."
uv pip install -e .

# Test that the package can be imported
echo "Testing import..."
python -c "import keras_geometric; print(f'Successfully imported keras_geometric version: {keras_geometric.__version__}')"

# Deactivate and clean up
deactivate
echo "Cleaning up..."
rm -rf test_venv

echo "Verification complete!"
