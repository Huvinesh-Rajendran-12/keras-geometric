# Keras Geometric

Keras Geometric is a library built on Keras (version 3+) designed for geometric deep learning, with a primary focus on Graph Neural Networks (GNNs). It provides modular building blocks to easily create and experiment with GNN architectures within the Keras ecosystem.

The core philosophy is to offer a flexible and intuitive API, leveraging the power and simplicity of Keras for building complex graph-based models. Key components include a versatile [`MessagePassing`](src/keras-geometric/message_passing.py) base layer and implementations of popular graph convolution layers like `GCNConv` and `GINConv`.

## Features

-   **Flexible Message Passing:** A core `MessagePassing` layer that handles the fundamental logic of neighborhood aggregation, allowing for easy customization of message creation, aggregation, and update steps. Supports various aggregation methods (e.g., 'sum', 'mean', 'max').
-   **Standard Graph Convolutions:** Ready-to-use implementations of popular graph convolution layers:
    -   `GCNConv`: Graph Convolutional Network layer from Kipf & Welling (2017).
    -   `GINConv`: Graph Isomorphism Network layer from Xu et al. (2019).
-   **Seamless Keras Integration:** Designed as standard Keras layers, making them easy to integrate into `keras.Sequential` or functional API models.
-   **Backend Agnostic:** Leverages Keras 3, allowing compatibility with different backends like TensorFlow, PyTorch, and JAX (ensure backend compatibility with sparse operations if needed).

## Installation

1.  **Prerequisites:**
    *   Python 3.12 or later.
    *   Keras 3 (version 3.9.0 or later). You can install/update it using pip:
        ```sh
        pip install --upgrade keras>=3.9.0
        ```
    *   A Keras backend (TensorFlow, PyTorch, or JAX). Install your preferred backend if you haven't already (e.g., `pip install tensorflow`).

2.  **Install Keras Geometric:**
    Currently, you can install the library directly from the source repository:
    ```sh
    # Clone the repository (if you haven't already)
    git clone <your-repo-url> # Replace <your-repo-url> with the actual URL
    cd keras-geometric

    # Install the package
    pip install .
    # Or, for development mode (changes in source code reflect immediately)
    # pip install -e .
    ```
    *(Note: If you plan to publish to PyPI later, update this section accordingly.)*

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
# units: Dimensionality of the output node embeddings
# activation: Activation function
gcn_layer = GCNConv(units=16, activation='relu')

# The GCNConv layer expects inputs as a list or tuple: [node_features, edge_index]
# It also needs the number of nodes to correctly handle potential isolates
# and determine the output shape.
# Note: Passing num_nodes might vary depending on the final layer implementation details.
# Check the layer's call signature. Assuming it's passed like this for now:
node_embeddings = gcn_layer([node_input, edge_input], num_nodes=num_nodes)

# Optional: Add more Keras layers (e.g., another GCN layer, Dense layers)
# node_embeddings = GCNConv(units=8, activation='relu')([node_embeddings, edge_input], num_nodes=num_nodes)
# node_embeddings = keras.layers.Dense(4)(node_embeddings)

# Create the Keras model
model = keras.Model(inputs=[node_input, edge_input], outputs=node_embeddings)

model.summary()

# --- 3. Use the Model (Example: Get embeddings) ---
# Prepare inputs for the model
# Note: Keras functional API often expects inputs as a dictionary or list
# matching the Input layers.
input_data = {
    "node_features": node_features,
    "edge_index": edge_index
}
# Or as a list: [node_features, edge_index]

# Get the node embeddings
output_embeddings = model.predict(input_data)

print("Input Node Features Shape:", node_features.shape)
print("Edge Index Shape:", edge_index.shape)
print("Output Node Embeddings Shape:", output_embeddings.shape)
# Expected output shape: (num_nodes, units) -> (4, 16)
```

