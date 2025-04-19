#!/bin/bash
# Script to create a new release and deploy to PyPI

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.0"
    exit 1
fi

VERSION=$1

# Validate semantic version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must follow semantic versioning (X.Y.Z)"
    exit 1
fi

# Check if we're on the main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Error: You must be on the main branch to create a release."
    exit 1
fi

# Make sure the working directory is clean
if ! git diff-index --quiet HEAD --; then
    echo "Error: Working directory is not clean. Please commit or stash changes."
    exit 1
fi

# Pull latest changes
echo "Pulling latest changes from main..."
git pull origin main

# Verify tests pass
echo "Running tests..."
uv run -m pytest tests/

if [ $? -ne 0 ]; then
    echo "Error: Tests failed. Cannot create release."
    exit 1
fi

# Create a tagged release
echo "Creating release v$VERSION..."
git tag -a "v$VERSION" -m "Release v$VERSION"
git push origin "v$VERSION"

# The actual PyPI deployment will be triggered by the CI/CD pipeline
# when it sees the new tag

echo "Release v$VERSION created and pushed to GitLab!"
echo "The CI/CD pipeline will automatically build and deploy to PyPI."
echo "You can check the status at: https://gitlab.com/<your-username>/keras-geometric/-/pipelines"
