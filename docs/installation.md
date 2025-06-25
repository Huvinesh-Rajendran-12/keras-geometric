# Installation

## Requirements

- Python 3.9 or later
- Keras 3.0 or later
- One of the following backends:
  - TensorFlow 2.19.0 or later
  - PyTorch 2.6.0 or later
  - JAX 0.4.30 or later

## Install from PyPI

### Basic Installation

The base package includes only the core dependencies:

```bash
pip install keras-geometric
```

### Backend-Specific Installation

Install with your preferred backend:

```bash
# For TensorFlow backend
pip install keras-geometric[tensorflow]

# For PyTorch backend
pip install keras-geometric[pytorch]

# For JAX backend
pip install keras-geometric[jax]
```

### Additional Features

```bash
# Dataset utilities
pip install keras-geometric[datasets]

# Development tools
pip install keras-geometric[dev]

# All features
pip install keras-geometric[all]
```

## Install from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/Huvinesh-Rajendran-12/keras-geometric.git
cd keras-geometric
pip install -e ".[dev]"
```

## Backend Configuration

Set your preferred backend before importing Keras:

```bash
# Environment variable
export KERAS_BACKEND=tensorflow  # or torch, jax

# Or in Python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
```

## Verify Installation

```python
import keras_geometric
print(keras_geometric.__version__)

# Check backend
import keras
print(f"Backend: {keras.backend.backend()}")
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you have installed the appropriate backend:

```bash
# Check installed packages
pip list | grep -E "tensorflow|torch|jax"
```

### Backend Issues

If the backend isn't detected correctly:

1. Set the `KERAS_BACKEND` environment variable explicitly
2. Ensure the backend package is installed
3. Restart your Python session after changing backends

### macOS Metal Support

For GPU acceleration on Apple Silicon Macs:

```bash
pip install keras-geometric[macos-metal]
```
