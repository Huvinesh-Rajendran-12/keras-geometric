# Getting Started with Keras Geometric

Welcome to Keras Geometric! This tutorial will guide you through the basics of building Graph Neural Networks (GNNs) using our library.

## Overview

Keras Geometric is a library for building and training Graph Neural Networks using Keras 3. It provides:

- **Graph Neural Network layers**: GCN, GIN, GAT, SAGE, and more
- **Pooling operations**: Global pooling, attention pooling, Set2Set
- **Backend compatibility**: Works with TensorFlow, PyTorch, and JAX backends
- **Easy integration**: Seamlessly integrates with existing Keras workflows

## Installation

First, install the package and its dependencies:

```bash
pip install keras-geometric

# For development installation:
pip install -e ".[dev]"
```

## Basic Concepts

### Graphs in Machine Learning

Graphs consist of:
- **Nodes** (vertices): Entities with features
- **Edges**: Connections between nodes
- **Node features**: Feature vectors for each node
- **Edge indices**: Pairs of node indices defining connections

### Graph Representation

In Keras Geometric, graphs are represented as:
- `node_features`: Tensor of shape `[num_nodes, num_features]`
- `edge_indices`: Tensor of shape `[2, num_edges]` containing source and target node indices

## Your First GNN Model

Let's build a simple Graph Convolutional Network (GCN) for node classification:

```python
import numpy as np
import keras
from keras_geometric.layers import GCNConv

# Set your preferred backend
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # or "torch", "jax"

# Create sample graph data
num_nodes = 100
num_features = 16
num_edges = 300

# Random node features
node_features = np.random.randn(num_nodes, num_features).astype(np.float32)

# Random edge connections (ensuring valid indices)
edge_indices = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int32)

# Convert to Keras tensors
node_features = keras.ops.convert_to_tensor(node_features)
edge_indices = keras.ops.convert_to_tensor(edge_indices)

# Build a simple GCN model
hidden_dim = 32
output_dim = 7  # Number of classes

# Create GCN layers
gcn1 = GCNConv(hidden_dim, use_bias=True)
gcn2 = GCNConv(output_dim, use_bias=True)

# Forward pass
x = gcn1([node_features, edge_indices])
x = keras.ops.relu(x)
x = gcn2([x, edge_indices])
predictions = keras.ops.softmax(x)

print(f"Input shape: {node_features.shape}")
print(f"Output shape: {predictions.shape}")
```

## Node Classification Example

Here's a complete example for node classification on a graph:

```python
import numpy as np
import keras
from keras_geometric.layers import GCNConv

# Create synthetic graph data
def create_synthetic_graph(num_nodes=200, num_features=16, num_classes=3):
    \"\"\"Create a synthetic graph with communities.\"\"\"
    # Create node features
    node_features = np.random.randn(num_nodes, num_features).astype(np.float32)

    # Create labels (3 communities)
    labels = np.random.randint(0, num_classes, size=num_nodes)

    # Create edges with higher probability within communities
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Higher connection probability within same community
            same_community = labels[i] == labels[j]
            prob = 0.1 if same_community else 0.01

            if np.random.random() < prob:
                edges.append([i, j])
                edges.append([j, i])  # Undirected graph

    edge_indices = np.array(edges).T.astype(np.int32)

    return node_features, edge_indices, labels

# Create data
node_features, edge_indices, labels = create_synthetic_graph()

# Convert to Keras tensors
node_features = keras.ops.convert_to_tensor(node_features)
edge_indices = keras.ops.convert_to_tensor(edge_indices)
labels_categorical = keras.utils.to_categorical(labels, num_classes=3)

print(f"Graph: {node_features.shape[0]} nodes, {edge_indices.shape[1]} edges")

# Build multi-layer GCN
class NodeClassificationGCN:
    def __init__(self, hidden_dim=64, num_classes=3, dropout=0.5):
        self.gcn1 = GCNConv(hidden_dim, use_bias=True)
        self.gcn2 = GCNConv(hidden_dim, use_bias=True)
        self.classifier = keras.layers.Dense(num_classes, activation="softmax")
        self.dropout = keras.layers.Dropout(dropout)

    def __call__(self, node_features, edge_indices, training=False):
        # First GCN layer
        x = self.gcn1([node_features, edge_indices])
        x = keras.ops.relu(x)
        x = self.dropout(x, training=training)

        # Second GCN layer
        x = self.gcn2([x, edge_indices])
        x = keras.ops.relu(x)
        x = self.dropout(x, training=training)

        # Classification
        predictions = self.classifier(x)
        return predictions

# Create model
model = NodeClassificationGCN(hidden_dim=32, num_classes=3)

# Forward pass
predictions = model(node_features, edge_indices, training=False)

# Compute accuracy
predicted_labels = keras.ops.argmax(predictions, axis=1)
true_labels = keras.ops.convert_to_tensor(labels)
accuracy = keras.ops.mean(keras.ops.equal(predicted_labels, true_labels))

print(f"Random baseline accuracy: {keras.ops.convert_to_numpy(accuracy):.3f}")
```

## Graph Classification Example

For graph-level tasks, we need pooling to create graph representations:

```python
import numpy as np
import keras
from keras_geometric.layers import GINConv
from keras_geometric.layers.pooling import GlobalPooling

# Create multiple graphs (batch)
def create_graph_batch():
    \"\"\"Create a batch of small graphs.\"\"\"
    graphs = []

    for _ in range(3):  # 3 graphs in batch
        num_nodes = np.random.randint(10, 20)
        num_features = 8

        # Random features and edges
        node_features = np.random.randn(num_nodes, num_features).astype(np.float32)

        # Create some random edges
        num_edges = num_nodes * 2
        edge_indices = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int32)

        graphs.append((node_features, edge_indices))

    return graphs

# Create graph classification model
class GraphClassificationGIN:
    def __init__(self, hidden_dim=32, num_classes=2):
        self.gin1 = GINConv(hidden_dim, aggregator="sum")
        self.gin2 = GINConv(hidden_dim, aggregator="sum")
        self.pooling = GlobalPooling(pooling="mean")
        self.classifier = keras.layers.Dense(num_classes, activation="softmax")

    def __call__(self, node_features, edge_indices):
        # Graph convolutions
        x = self.gin1([node_features, edge_indices])
        x = keras.ops.relu(x)

        x = self.gin2([x, edge_indices])
        x = keras.ops.relu(x)

        # Global pooling to get graph representation
        graph_repr = self.pooling(x)

        # Classification
        predictions = self.classifier(graph_repr)
        return predictions

# Test on individual graphs
graphs = create_graph_batch()
model = GraphClassificationGIN(hidden_dim=16, num_classes=2)

for i, (node_features, edge_indices) in enumerate(graphs):
    node_features = keras.ops.convert_to_tensor(node_features)
    edge_indices = keras.ops.convert_to_tensor(edge_indices)

    predictions = model(node_features, edge_indices)
    print(f"Graph {i+1}: {node_features.shape[0]} nodes -> prediction shape: {predictions.shape}")
```

## Advanced: Attention-based Models

Let's build a Graph Attention Network (GAT) with attention pooling:

```python
from keras_geometric.layers import GATv2Conv
from keras_geometric.layers.pooling import AttentionPooling

class AdvancedGraphModel:
    def __init__(self, hidden_dim=32, num_heads=4, num_classes=3):
        # Multi-head attention layers
        self.gat1 = GATv2Conv(hidden_dim, heads=num_heads, use_bias=True)
        self.gat2 = GATv2Conv(hidden_dim, heads=2, use_bias=True)

        # Attention-based pooling
        self.attention_pool = AttentionPooling(attention_dim=16)

        # Classification
        self.classifier = keras.layers.Dense(num_classes, activation="softmax")

    def __call__(self, node_features, edge_indices, training=False):
        # First GAT layer (multi-head)
        x = self.gat1([node_features, edge_indices])
        x = keras.ops.elu(x)

        # Second GAT layer
        x = self.gat2([x, edge_indices])
        x = keras.ops.elu(x)

        # Attention pooling for graph representation
        graph_repr = self.attention_pool(x, training=training)

        # Classification
        predictions = self.classifier(graph_repr)
        return predictions

# Test the advanced model
node_features, edge_indices, _ = create_synthetic_graph(num_nodes=50, num_features=16)
node_features = keras.ops.convert_to_tensor(node_features)
edge_indices = keras.ops.convert_to_tensor(edge_indices)

advanced_model = AdvancedGraphModel(hidden_dim=24, num_heads=3, num_classes=3)
predictions = advanced_model(node_features, edge_indices)

print(f"Advanced model output shape: {predictions.shape}")
```

## Available Layers

### Graph Neural Network Layers

1. **GCNConv**: Graph Convolutional Network layer
   ```python
   from keras_geometric.layers import GCNConv
   gcn = GCNConv(output_dim=64, use_bias=True)
   ```

2. **GINConv**: Graph Isomorphism Network layer
   ```python
   from keras_geometric.layers import GINConv
   gin = GINConv(output_dim=64, aggregator="sum")
   ```

3. **GATv2Conv**: Graph Attention Network v2 layer
   ```python
   from keras_geometric.layers import GATv2Conv
   gat = GATv2Conv(output_dim=64, heads=4, use_bias=True)
   ```

4. **SAGEConv**: GraphSAGE layer
   ```python
   from keras_geometric.layers import SAGEConv
   sage = SAGEConv(output_dim=64, aggregator="mean")
   ```

### Pooling Layers

1. **GlobalPooling**: Simple global pooling (mean, max, sum)
   ```python
   from keras_geometric.layers.pooling import GlobalPooling
   pool = GlobalPooling(pooling="mean")
   ```

2. **AttentionPooling**: Attention-based pooling
   ```python
   from keras_geometric.layers.pooling import AttentionPooling
   att_pool = AttentionPooling(attention_dim=32)
   ```

3. **Set2Set**: Advanced attention pooling with LSTM
   ```python
   from keras_geometric.layers.pooling import Set2Set
   set2set = Set2Set(output_dim=32, processing_steps=3)
   ```

4. **BatchGlobalPooling**: Pooling for batched graphs
   ```python
   from keras_geometric.layers.pooling import BatchGlobalPooling
   batch_pool = BatchGlobalPooling(pooling="mean")
   ```

## Tips and Best Practices

### 1. Choosing the Right Layer

- **GCNConv**: Good starting point, simple and effective
- **GINConv**: Better for graph classification tasks
- **GATv2Conv**: When you need attention mechanisms
- **SAGEConv**: Good for large graphs with sampling

### 2. Model Architecture

```python
# Typical node classification architecture
x = GCNConv(hidden_dim)(inputs)
x = Activation("relu")(x)
x = Dropout(0.5)(x)
x = GCNConv(num_classes)(x)
outputs = Activation("softmax")(x)

# Typical graph classification architecture
x = GINConv(hidden_dim)(inputs)
x = Activation("relu")(x)
x = GlobalPooling("mean")(x)  # Graph-level representation
outputs = Dense(num_classes, activation="softmax")(x)
```

### 3. Common Patterns

- **Residual connections**: Add skip connections for deeper models
- **Normalization**: Use batch/layer normalization for stability
- **Dropout**: Apply dropout between layers for regularization
- **Multiple pooling**: Combine different pooling strategies

### 4. Backend Considerations

```python
# TensorFlow backend (default)
os.environ["KERAS_BACKEND"] = "tensorflow"

# PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

# JAX backend
os.environ["KERAS_BACKEND"] = "jax"
```

## What's Next?

Now that you understand the basics, explore:

1. **Real datasets**: Try the Cora dataset for citation networks
2. **Advanced architectures**: Combine multiple layer types
3. **Custom layers**: Build your own GNN layers
4. **Performance optimization**: Use backend-specific optimizations

## Troubleshooting

### Common Issues

1. **Shape mismatches**: Ensure edge_indices has shape `[2, num_edges]`
2. **Backend compatibility**: Set backend before importing keras_geometric
3. **Memory issues**: Use gradient checkpointing for large graphs

### Getting Help

- Check the [API documentation](../api_reference/index.md)
- Look at [examples](../../examples/) for more use cases
- Open an issue on GitHub for bugs or questions

Happy graph learning! 🎯📊
