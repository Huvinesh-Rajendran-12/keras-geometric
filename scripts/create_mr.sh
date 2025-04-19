#!/bin/bash
# Script to create a merge request to the main branch in GitLab

# Check if the gitlab CLI is installed
if ! command -v glab &> /dev/null; then
    echo "Error: GitLab CLI (glab) is not installed."
    echo "Please install it: https://gitlab.com/gitlab-org/cli"
    exit 1
fi

# Get the current branch
CURRENT_BRANCH=$(git branch --show-current)

if [ "$CURRENT_BRANCH" = "main" ]; then
    echo "Error: You are currently on the main branch. Please switch to a feature branch."
    exit 1
fi

# Run pre-commit hooks
echo "Running pre-commit checks..."
if ! pre-commit run --all-files; then
    echo "Error: Pre-commit checks failed. Please fix the issues and try again."
    exit 1
fi

# Run tests
echo "Running tests..."
uv run -m pytest tests/

if [ $? -ne 0 ]; then
    echo "Error: Tests failed. Please fix the tests and try again."
    exit 1
fi

# Prompt for merge request details
echo "Creating a merge request from $CURRENT_BRANCH to main"
echo "Please provide the following information:"
echo -n "Title: "
read TITLE

echo -n "Description (press Enter then Ctrl+D when done): "
DESCRIPTION=$(cat)

# Create the merge request
glab mr create --title "$TITLE" \
    --description "$DESCRIPTION" \
    --source-branch "$CURRENT_BRANCH" \
    --target-branch "main" \
    --remove-source-branch \
    --labels "feature"

echo "Merge request created successfully!"
echo "Once merged, you can create a release by running:"
echo "scripts/create_release.sh [version]"
