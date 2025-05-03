# Keras Geometric

[![PyPI version](https://badge.fury.io/py/keras-geometric.svg)](https://badge.fury.io/py/keras-geometric)
[![Build Status](https://github.com/Huvinesh-Rajendran-12/keras-geometric/workflows/Keras-Geometric%20CI/CD/badge.svg)](https://github.com/Huvinesh-Rajendran-12/keras-geometric/actions)
[![Python Versions](https://img.shields.io/pypi/pyversions/keras-geometric.svg)](https://pypi.org/project/keras-geometric/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Keras Geometric is a library built on Keras (version 3+) designed for geometric deep learning, with a primary focus on Graph Neural Networks (GNNs). It provides modular building blocks to easily create and experiment with GNN architectures within the Keras ecosystem.

The core philosophy is to offer a flexible and intuitive API, leveraging the power and simplicity of Keras for building complex graph-based models. Key components include a versatile [`MessagePassing`](src/keras_geometric/layers/message_passing.py) base layer and implementations of popular graph convolution layers like `GCNConv`, `GINConv`, and `GATv2Conv`.

## Features

- **Flexible Message Passing:** A core `MessagePassing` layer that handles the fundamental logic of neighborhood aggregation, allowing for easy customization of message creation, aggregation, and update steps. Supports various aggregation methods (e.g., 'sum', 'mean', 'max').
- **Standard Graph Convolutions:** Ready-to-use implementations of popular graph convolution layers:
  - `GCNConv`: Graph Convolutional Network layer from Kipf & Welling (2017).
  - `GINConv`: Graph Isomorphism Network layer from Xu et al. (2019).
  - `GATv2Conv`: Graph Attention Network v2 layer from Brody et al. (2021), providing dynamic attention for better expressiveness.
  - `SAGEConv`: GraphSAGE layer from Hamilton et al. (2017), for inductive representation learning.
- **Data Handling:** Built-in `GraphData` class and utilities for managing graph-structured data and batching multiple graphs together.
- **Benchmark Datasets:** Standard citation network datasets (Cora, CiteSeer, PubMed) for node classification tasks.
- **Seamless Keras Integration:** Designed as standard Keras layers, making them easy to integrate into `keras.Sequential` or functional API models.
- **Backend Agnostic:** Leverages Keras 3, allowing compatibility with different backends like TensorFlow, PyTorch, and JAX (ensure backend compatibility with sparse operations if needed).
- **Example Models:** Comprehensive examples for various graph learning tasks including node classification, graph classification, and molecular property prediction.

## Installation

1.  **Prerequisites:**

    - Python 3.12 or later.
    - Keras 3 (version 3.9.0 or later). You can install/update it using pip:
      ```sh
      pip install --upgrade keras>=3.9.0
      ```
    - A Keras backend (TensorFlow, PyTorch, or JAX). Install your preferred backend if you haven't already (e.g., `pip install tensorflow`).

2.  **Install Keras Geometric:**

    There are multiple ways to install Keras Geometric:

    **Option 1: Install from PyPI (Recommended)**
    ```sh
    # Install the latest stable version
    pip install keras-geometric

    # Install with additional features
    pip install keras-geometric[dev]  # Development dependencies
    pip install keras-geometric[test] # Testing dependencies
    pip install keras-geometric[macos-metal] # Metal acceleration for macOS
    ```

    **Option 2: Install from Source**
    ```sh
    # Clone the repository
    git clone https://github.com/Huvinesh-Rajendran-12/keras-geometric.git
    cd keras-geometric

    # Install the package
    pip install .
    # Or, for development mode (changes in source code reflect immediately)
    pip install -e .
    # With development dependencies
    pip install -e ".[dev]"
    ```

    **Option 3: Using the uv Package Manager**
    ```sh
    # Install the latest stable version
    uv pip install keras-geometric

    # For development mode
    uv pip install -e .
    ```

## Core Concepts: Graph Neural Networks & Message Passing

Graph Neural Networks (GNNs) are a class of neural networks designed to operate directly on graph-structured data. They learn representations (embeddings) of nodes, edges, or entire graphs by leveraging the graph's topology.

**The Message Passing Paradigm:**

Many GNN layers can be understood through the lens of **message passing**. This is a general framework where nodes iteratively update their representations by exchanging and aggregating information with their neighbors. A typical message passing iteration involves three steps:

1.  **Message Computation:** Each node computes messages to send to its neighbors, often based on its own features and the features of the sending node.
2.  **Aggregation:** Each node aggregates the incoming messages from its neighbors. Common aggregation functions include sum, mean, or max.
3.  **Update:** Each node updates its own representation (embedding) based on its aggregated messages and its previous representation.

The `MessagePassing` layer in Keras Geometric encapsulates this process. Specific layers like `GCNConv` and `GINConv` inherit from `MessagePassing` and implement these steps according to their respective mathematical formulations.

**Graph Convolutional Networks (GCN):**

GCN layers (Kipf & Welling, 2017) perform a spectral-based convolution on graphs. A simplified view is that they update a node's representation by taking a weighted average of its own features and the features of its neighbors, followed by a linear transformation and non-linearity. The weights are often derived from the graph's adjacency matrix, typically normalized.

**Graph Isomorphism Networks (GIN):**

GIN layers (Xu et al., 2019) were designed to be maximally expressive GNNs, theoretically as powerful as the Weisfeiler-Lehman graph isomorphism test. They use a learnable function (often a small Multi-Layer Perceptron - MLP) to combine a node's features with the aggregated features of its neighbors.

**Graph Attention Networks v2 (GATv2):**

GATv2 layers (Brody et al., 2021) address a theoretical limitation in the original GAT architecture by using a more expressive attention mechanism that enables dynamic attention. This allows the model to assign different importance to different neighbors based on their features and the current task, improving performance on various graph learning tasks.

**Graph SAGE (GraphSAGE):**

GraphSAGE (Hamilton et al., 2017) is designed for inductive representation learning on large graphs. It learns a function that can generate embeddings for previously unseen nodes by sampling and aggregating features from a node's local neighborhood. GraphSAGE supports various aggregation functions (mean, max, LSTM) to capture different structural properties of the graph.

## Quick Start: Using `GCNConv`

Here's a basic example of how to use the `GCNConv` layer within a Keras functional model:

```python
import keras
import numpy as np
# Assuming keras_geometric is installed and importable
from keras_geometric import GCNConv

# --- 1. Prepare Graph Data ---
# Example: A simple graph with 4 nodes and 5 edges
# Node features (e.g., 3 features per node)
node_features = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]
], dtype=np.float32)

# Edge index (COO format: [senders, receivers])
# Edges: 0->1, 0->2, 1->2, 2->3, 3->0
edge_index = np.array([
    [0, 0, 1, 2, 3],  # Senders
    [1, 2, 2, 3, 0]   # Receivers
], dtype=np.int32)

num_nodes = node_features.shape[0]

# --- 2. Define the GNN Model ---
# Input layers
node_input = keras.Input(shape=(node_features.shape[1],), name="node_features")
edge_input = keras.Input(shape=(2, None), dtype="int32", name="edge_index") # Shape (2, num_edges)

# Apply GCN layer
# output_dim: Dimensionality of the output node embeddings
# activation: Activation function
gcn_layer = GCNConv(output_dim=16, activation='relu')

# The GCNConv layer expects inputs as a list or tuple: [node_features, edge_index]
node_embeddings = gcn_layer([node_input, edge_input])

# Create the Keras model
model = keras.Model(inputs=[node_input, edge_input], outputs=node_embeddings)

model.summary()

# --- 3. Use the Model (Example: Get embeddings) ---
# Get the node embeddings
output_embeddings = model.predict([node_features, edge_index])

print("Input Node Features Shape:", node_features.shape)
print("Edge Index Shape:", edge_index.shape)
print("Output Node Embeddings Shape:", output_embeddings.shape)
# Expected output shape: (num_nodes, units) -> (4, 16)
```

## Example Models for Graph Learning Tasks

In the `examples/` directory, you'll find a collection of example models for various graph learning tasks:

### Basic Examples

- `basic_gcn_example.py`: A minimal example showing how to use GCN for node classification on a synthetic graph
- `simple_gatv2_example_fixed.py`: A simple example demonstrating the usage of `GATv2Conv` with a fixed graph structure. This file is also present in the root directory for quick access.

### Node Classification

- `node_classification/gcn_citation.py`: Node classification on the Cora citation network using GCN

### Molecular Property Prediction

- `molecular_property_prediction/gin_molecule_classification.py`: Using Graph Isomorphism Networks (GIN) for molecular property prediction

### Graph Classification

- `graph_classification/multi_gnn_graph_classification.py`: A multi-branch GNN model that combines GCN, GAT and GraphSAGE for graph classification

These examples demonstrate how to implement various graph learning tasks and can serve as templates for your own applications. See the `examples/README.md` file for more details about each example.

## Working with Datasets

Keras Geometric provides built-in dataset loaders for common benchmark datasets:

```python
from keras_geometric.datasets import Cora

# Load the Cora citation network dataset
dataset = Cora(root="data")

# Get the single graph
graph = dataset[0]

# Access graph components
x = graph.x  # Node features
y = graph.y  # Node labels
edge_index = graph.edge_index  # Edge connectivity

# Use with your GNN model
model.fit(
    [x, edge_index],
    y,  # Target node labels
    epochs=200,
    batch_size=1  # Process the entire graph as one batch
)
```

## Development and Testing

For development, use these commands:

```sh
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_gcn_conv.py::TestGCNConvComprehensive::test_refactored_initialization

# Run with verbose output
python -m pytest -v tests/
```

## Contributing

We welcome contributions to Keras Geometric! Here's how you can contribute:

1. **Fork the repository** and clone it locally.
2. **Create a new branch** for your feature or bugfix.
3. **Implement your changes** following the project's coding style.
4. **Run the tests** to ensure your changes don't break existing functionality:
   ```sh
   python -m pytest
   ```
5. **Add tests** for your new features to ensure they work as expected.
6. **Submit a pull request** with a clear description of the changes and any relevant documentation.

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration and delivery:

- All pull requests and pushes to the main branch are automatically tested.
- When a new version tag (e.g., `v0.2.0`) is pushed, the package is:
  1. Built and tested across multiple Python versions and backends
  2. Published to TestPyPI for verification
  3. Published to PyPI for general availability
  4. A GitHub release is created automatically

### Versioning

The project follows [Semantic Versioning](https://semver.org/). Version numbers are derived from Git tags and automatically applied during the build process.

## Citation

If you use this library in your research, please cite the respective papers for the GNN methods you use:

- GCN: Kipf & Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
- GIN: Xu et al., [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (ICLR 2019)
- GATv2: Brody et al., [How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) (ICLR 2022)
- GraphSAGE: Hamilton et al., [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NeurIPS 2017)
