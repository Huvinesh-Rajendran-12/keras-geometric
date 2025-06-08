"""
Test configuration for keras-geometric.

Sets up consistent backend configuration and handles PyTree registration conflicts.
"""

import os
import sys

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set default backend before any imports
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# Clear any existing Keras modules to avoid PyTree conflicts
modules_to_clear = [name for name in sys.modules.keys() if name.startswith("keras")]
for module in modules_to_clear:
    sys.modules.pop(module, None)

# PyTree registration conflicts will be handled at test level
