# Development Scripts for Keras Geometric

This directory contains helpful scripts for development and release management.

## Prerequisites

1. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. Install GitLab CLI for merge request creation:
   ```bash
   # For macOS
   brew install glab

   # For Linux
   curl -s https://raw.githubusercontent.com/profclems/glab/trunk/scripts/install.sh | sudo bash

   # Authentication
   glab auth login
   ```

## Creating a Merge Request

1. Create a new feature branch from main:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them.

3. Run the merge request script:
   ```bash
   ./scripts/create_mr.sh
   ```

This script will:
- Run pre-commit hooks to check code quality
- Run tests to ensure everything works
- Create a merge request on GitLab from your branch to main

## Creating a Release

After your changes are merged to main:

1. Switch to the main branch:
   ```bash
   git checkout main
   git pull
   ```

2. Run the release script with the semantic version:
   ```bash
   ./scripts/create_release.sh 1.0.0
   ```

This script will:
- Verify you're on the main branch
- Ensure all tests pass
- Create and push a version tag
- The CI/CD pipeline will automatically build and deploy to PyPI

## Setting up PyPI Deployment

To enable automatic PyPI deployment, you need to set up CI/CD variables in GitLab:

1. Go to your GitLab repository: Settings > CI/CD > Variables
2. Add the following variables:
   - `PYPI_USERNAME`: Your PyPI username
   - `PYPI_PASSWORD`: Your PyPI password or API token (mark as Protected and Masked)

## Development Workflow

1. Create a feature branch
2. Make changes and test locally with `uv run -m pytest tests/`
3. Commit changes (pre-commit hooks will run automatically)
4. Create a merge request with `./scripts/create_mr.sh`
5. After merge request is approved and merged, create a release with `./scripts/create_release.sh X.Y.Z`
