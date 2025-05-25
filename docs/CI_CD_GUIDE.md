# CI/CD Pipeline Usage Guide

This guide explains how to use the GitHub Actions CI/CD pipeline for Keras Geometric development, testing, and deployment.

## 🏗️ Pipeline Overview

The CI/CD pipeline consists of several automated workflows that ensure code quality, run tests, and handle deployments:

### Main CI/CD Workflow (`ci-cd.yml`)
- **Trigger**: Push to main, PRs, tags, manual dispatch
- **Jobs**: Filter → Lint → Security → Test → Build → Publish → Release

### Supporting Workflows
- **PR Labeler**: Automatically labels PRs based on changed files
- **Performance**: Weekly benchmarks and performance regression detection
- **Cleanup**: Weekly cache cleanup to optimize storage
- **Dependabot**: Automated dependency updates

## 🚀 Getting Started

### Prerequisites
```bash
# Install development dependencies
pip install -e ".[dev]"

# Verify installation
./scripts/verify_editable_install.sh
```

### Local Development Workflow
```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Make changes and test locally
ruff format .                # Format code
ruff check .                 # Lint code
pyrefly check               # Type check
python -m pytest tests/ -v  # Run tests

# 3. Commit and push
git add .
git commit -m "feat: implement new feature"
git push origin feature/my-new-feature

# 4. Create PR (triggers CI automatically)
gh pr create --title "Add new feature" --body "Description"
```

## 📊 Pipeline Jobs Explained

### 1. Filter Job
- **Purpose**: Detects if Python files changed to skip unnecessary runs
- **Files Watched**: `src/**/*.py`, `tests/**/*.py`, `pyproject.toml`, `uv.lock`
- **Performance**: Saves ~5-10 minutes on non-Python changes

### 2. Lint Job
- **Tools**: Ruff (format check + linting) + Pyrefly (type checking)
- **Runtime**: ~2-3 minutes
- **Checks**:
  ```bash
  ruff format --check .  # Code formatting
  ruff check .          # Code quality
  pyrefly check         # Type safety
  ```

### 3. Security Job
- **Tools**: pip-audit + safety
- **Runtime**: ~1-2 minutes
- **Scans**: Known vulnerabilities in dependencies

### 4. Test Matrix Job
- **Backends**: TensorFlow, PyTorch, JAX
- **Python Versions**: 3.9, 3.10, 3.11, 3.12
- **Total**: 12 test combinations
- **Runtime**: ~15-20 minutes (parallel execution)

### 5. Build Job (Tags Only)
- **Trigger**: Version tags (v1.0.0, v1.0.0-beta1, etc.)
- **Output**: Source distribution + wheel
- **Uses**: hatch-vcs for version from git tags

### 6. Publish Jobs (Tags Only)
- **PyPI**: Stable releases (v1.0.0)
- **TestPyPI**: Pre-releases (v1.0.0-beta1, v1.0.0-alpha1, v1.0.0-rc1)
- **Method**: GitHub trusted publishing (no API keys needed)

### 7. GitHub Release (Tags Only)
- **Auto-generates**: Release notes from git commits
- **Attaches**: Distribution files
- **Marks**: Pre-releases appropriately

## 🏷️ Automatic PR Labeling

PRs are automatically labeled based on changed files:

| Files Changed | Labels Applied |
|---------------|----------------|
| `src/keras_geometric/layers/**` | `core` |
| `src/keras_geometric/datasets/**` | `datasets` |
| `tests/**` | `tests` |
| `*.md`, `docs/**` | `documentation` |
| `pyproject.toml`, `.github/**` | `build`, `ci/cd` |
| Branch: `fix/*`, `bugfix/*` | `bug` |
| Branch: `feature/*`, `feat/*` | `enhancement` |
| Branch: `breaking/*` | `breaking` |

## 📈 Performance Monitoring

### Weekly Benchmarks
- **Schedule**: Monday 2 AM UTC
- **Tests**: All GNN layers (GCNConv, GINConv, GATv2Conv, SAGEConv)
- **Metrics**: Memory usage, execution time
- **Backends**: TensorFlow, PyTorch

### Manual Performance Testing
```bash
# Trigger performance workflow manually
gh workflow run performance.yml

# View results
gh run list --workflow=performance.yml
```

## 🔄 Dependency Management

### Dependabot Configuration
- **Schedule**: Weekly updates on Monday 9 AM UTC
- **Limits**: 5 pip updates, 3 GitHub Actions updates
- **Auto-assigns**: To repository owner
- **Ignores**: Major version updates for TensorFlow, PyTorch, Keras

### Manual Dependency Updates
```bash
# Update specific dependency
uv pip install --upgrade tensorflow

# Update all dependencies
uv pip install --upgrade -e ".[dev]"

# Check for security issues
pip-audit
safety check
```

## 🚀 Release Process

### 1. Automated Release (Recommended)
```bash
# Create and push tag - CI handles the rest
./scripts/create_release.sh 1.2.3

# This triggers:
# 1. Full test suite across all backends/Python versions
# 2. Build source + wheel distributions
# 3. Publish to PyPI (stable) or TestPyPI (pre-release)
# 4. Create GitHub release with auto-generated notes
```

### 2. Manual Release Steps
```bash
# 1. Ensure clean main branch
git checkout main
git pull origin main

# 2. Run tests locally
python -m pytest tests/

# 3. Create tag
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3

# 4. Monitor CI/CD pipeline
gh run watch
```

### Release Types and Destinations

| Version Pattern | Destination | Example |
|----------------|-------------|---------|
| `v1.2.3` | PyPI | Stable release |
| `v1.2.3-beta1` | TestPyPI | Beta testing |
| `v1.2.3-alpha1` | TestPyPI | Alpha testing |
| `v1.2.3-rc1` | TestPyPI | Release candidate |

## 🔧 Troubleshooting

### Common CI Failures

**Linting Failures**
```bash
# Fix formatting
ruff format .

# Fix linting issues
ruff check --fix .

# Check types
pyrefly check
```

**Test Failures**
```bash
# Run specific test
python -m pytest tests/test_gcn_conv.py::TestGCNConvComprehensive::test_refactored_initialization -v

# Run with specific backend
KERAS_BACKEND=tensorflow python -m pytest tests/ -v
```

**Build Failures**
```bash
# Test build locally
python -m build

# Check package
twine check dist/*
```

### Pipeline Debugging

**View detailed logs**
```bash
# List recent runs
gh run list

# View specific run
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

**Manual workflow triggers**
```bash
# Trigger CI manually
gh workflow run ci-cd.yml

# Trigger performance tests
gh workflow run performance.yml

# Trigger cache cleanup
gh workflow run cleanup.yml
```

## 📚 Environment Variables

### CI Environment Variables
- `KERAS_BACKEND`: tensorflow/torch/jax (set automatically)
- `UV_CACHE_DIR`: UV package cache location
- `PIP_CACHE_DIR`: Pip cache location
- `UV_GLOBAL_CACHE`: Enable UV global caching

### Secrets Required
- `GITHUB_TOKEN`: Automatically provided by GitHub
- No PyPI tokens needed (uses trusted publishing)

## 🎯 Best Practices

### Development
1. **Run checks locally** before pushing
2. **Create focused PRs** with clear descriptions
3. **Use conventional commits** (feat:, fix:, docs:, etc.)
4. **Test across backends** for compatibility

### Releases
1. **Use semantic versioning** (major.minor.patch)
2. **Test pre-releases** on TestPyPI first
3. **Write clear commit messages** (used in release notes)
4. **Coordinate breaking changes** with major versions

### Performance
1. **Monitor benchmark results** weekly
2. **Profile locally** for performance-critical changes
3. **Use caching effectively** in development
4. **Keep dependencies minimal** in core library

## 📞 Getting Help

**Documentation**
- CI/CD Guide: This document
- Development Guide: `CLAUDE.md`
- Project README: `README.md`

**GitHub Features**
- Actions: https://github.com/Huvinesh-Rajendran-12/keras-geometric/actions
- Releases: https://github.com/Huvinesh-Rajendran-12/keras-geometric/releases
- Issues: https://github.com/Huvinesh-Rajendran-12/keras-geometric/issues

**Commands Reference**
```bash
# Development
pip install -e ".[dev]"
python -m pytest tests/ -v
ruff check .
pyrefly check

# Release
./scripts/create_release.sh 1.2.3
gh run watch

# Monitoring
gh run list
gh workflow list
```
