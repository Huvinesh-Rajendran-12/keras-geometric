# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Keras Geometric Development Guide

## Build/Test Commands

- Install: `pip install -e .` or `pip install -e ".[dev]"`
- Run all tests: `uv run pytest tests/`
- Run specific test: `uv run pytest tests/test_gcn_conv.py::TestGCNConvComprehensive::test_refactored_initialization`
- Run with verbose output: `uv run pytest -v tests/`
- Lint with Ruff: `ruff check .` or `ruff check src/`
- Format with Ruff: `ruff format .`
- Type check with Pyrefly: `pyrefly check`
- Verify editable installation: `./scripts/verify_editable_install.sh`
- Create a release: `./scripts/create_release.sh <version>`

## Type Checking Configuration

- Type checking uses Pyrefly, configured in `pyrefly.toml`
- Disable specific error types using the `[errors]` section (e.g., `bad-argument-type = false`)
- Inline ignores can be added with `# pyrefly: ignore` comments in the code
- For more complex ignores, use `# pyrefly: ignore # error-code` to target specific errors

### Known Type Issues

The codebase has a few types of intentionally ignored type errors:

1. **implicitly-defined-attribute**: Attributes like `self.mlp`, `self.kernel`, and `self.bias` that are defined during the `build()` method rather than in `__init__`. This is standard in Keras layers and can be ignored with `# pyrefly: ignore  # implicitly-defined-attribute`.

2. **bad-return**: In some cases, especially in the `SAGEConv` layer, there might be return type mismatches that Pyrefly doesn't correctly infer with complex tensor operations. These can be ignored with `# pyrefly: ignore  # bad-return`.

3. **bad-argument-type**: Errors related to tensor operations that Pyrefly doesn't correctly understand. This is disabled globally in the `pyrefly.toml` configuration.

When adding new code, follow the existing patterns for handling these known type issues. For Keras layers, declare attributes as `None` in `__init__` before they're assigned values in `build()`.

## Architecture Overview

Keras Geometric is a library built on Keras 3+ for geometric deep learning with a focus on Graph Neural Networks (GNNs). The core architecture follows these principles:

1. **MessagePassing Base Layer**: The foundational abstraction in `layers/message_passing.py` that handles the message passing paradigm of graph neural networks. All GNN layers inherit from this base class.

2. **GNN Layer Implementations**:

   - `GCNConv`: Graph Convolutional Network layer (Kipf & Welling, 2017)
   - `GINConv`: Graph Isomorphism Network layer (Xu et al., 2019)
   - `GATv2Conv`: Graph Attention Network v2 layer (Brody et al., 2021)
   - `SAGEConv`: GraphSAGE layer (Hamilton et al., 2017)

3. **Dataset Handlers**:

   - Base Dataset class in `datasets/base.py`
   - Citation networks (Cora) in `datasets/citation.py` and `datasets/cora.py`

4. **Backend Compatibility**: Uses Keras 3's backend-agnostic approach, allowing the library to work with TensorFlow, PyTorch, or JAX backends through the use of `keras.ops` for tensor operations.

## Code Style Guidelines

- **Imports**: Group imports by standard library → third-party → local packages
- **Typing**: Use type hints for function parameters (e.g., `def __init__(self, output_dim: int, use_bias: bool = True)`)
- **Docstrings**: Use docstrings with Args/Returns sections for all public methods
- **Error Handling**: Use assertions for internal validation, explicit errors with messages for APIs
- **Backend Compatibility**: Use `keras.ops` for tensor operations (not TensorFlow-specific operations)
- **Layer Structure**: Follow Keras Layer conventions with `build`, `call`, `get_config` methods
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Testing**: Include unit tests for all functionality with clear test case organization

## Testing Strategy

- Tests are organized by layer type (`test_gcn_conv.py`, `test_gatv2_conv.py`, etc.)
- Each test file includes comprehensive tests for initialization, forward pass, shape validation, and serialization
- When PyTorch backend is available, numerical comparison tests are run against PyTorch Geometric implementations
- Backend-agnostic code is ensured through the use of `keras.ops.convert_to_numpy()` for tensor conversion

## Development Workflow

1. Install the package in development mode: `pip install -e ".[dev]"`
2. Make code changes in the appropriate files
3. Run tests to verify changes: `python -m pytest tests/`
4. Lint and format code: `ruff check .` and `ruff format .`
5. Type check: `pyrefly check`
6. Create a pull request with your changes

## Release Process

1. Ensure all tests pass: `python -m pytest tests/`
2. Create a new release tag: `./scripts/create_release.sh X.Y.Z`
3. The CI/CD pipeline will automatically build and deploy to PyPI when a new tag is detected
