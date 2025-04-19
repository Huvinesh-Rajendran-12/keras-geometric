# Keras Geometric Development Guide

## Build/Test Commands
- Install: `pip install -e .` or `pip install -e ".[dev]"`
- Run all tests: `python -m pytest tests/`
- Run specific test: `python -m pytest tests/test_gcn_conv.py::TestGCNConvComprehensive::test_refactored_initialization`
- Run with verbose output: `python -m pytest -v tests/`

## Code Style Guidelines
- **Imports**: Group imports by standard library → third-party → local packages
- **Typing**: Use type hints for function parameters (e.g., `def __init__(self, output_dim: int, use_bias: bool = True)`)
- **Docstrings**: Use docstrings with Args/Returns sections for all public methods
- **Error Handling**: Use assertions for internal validation, explicit errors with messages for APIs
- **Backend Compatibility**: Use `keras.ops` for tensor operations (not TensorFlow-specific operations)
- **Layer Structure**: Follow Keras Layer conventions with `build`, `call`, `get_config` methods
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Testing**: Include unit tests for all functionality with clear test case organization
