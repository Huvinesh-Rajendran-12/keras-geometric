# Keras Geometric Examples

This directory contains examples demonstrating various features and use cases of Keras Geometric.

## Directory Structure

### 📚 Basic Examples (`basic/`)
Simple, introductory examples for each layer type:
- `simple_gcn_example.py` - Basic Graph Convolutional Network usage
- `simple_gin_example.py` - Basic Graph Isomorphism Network usage
- `simple_gatv2_example.py` - Basic Graph Attention Network v2 usage
- `simple_sage_example.py` - Basic GraphSAGE usage

### 🎯 Node Classification Demos (`node_classification_demos/`)
Complete examples for node classification tasks:
- `gcn_example.py` - Node classification with GCN on citation networks
- `gatv2_example.py` - Node classification with GATv2
- `graphsage_example.py` - Inductive learning with GraphSAGE

### 🚀 Advanced Examples (`advanced/`)
More complex examples showcasing advanced features:
- `pooling_example.py` - Graph pooling techniques for graph-level tasks

### 📂 Existing Tutorials (`node_classification/`)
Comprehensive tutorials with step-by-step explanations.

## Getting Started

1. Install Keras Geometric with your preferred backend:
   ```bash
   pip install keras-geometric[tensorflow]  # or pytorch, jax
   ```

2. Set your backend:
   ```bash
   export KERAS_BACKEND=tensorflow  # or torch, jax
   ```

3. Run any example:
   ```bash
   python basic/simple_gcn_example.py
   ```

## Example Categories by Use Case

### Learning the Basics
Start with examples in the `basic/` directory to understand:
- How to create graph data structures
- How to build simple GNN models
- How to perform forward passes

### Real-World Applications
Check `node_classification_demos/` for:
- Loading standard datasets (Cora, CiteSeer, PubMed)
- Building complete training pipelines
- Evaluating model performance

### Advanced Techniques
Explore `advanced/` for:
- Graph pooling mechanisms
- Custom message passing implementations
- Multi-task learning setups

## Contributing Examples

When adding new examples:
1. Place them in the appropriate category directory
2. Include clear comments explaining each step
3. Add a docstring at the top describing the example's purpose
4. Keep examples focused on demonstrating specific features
