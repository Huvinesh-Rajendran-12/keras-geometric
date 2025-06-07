# Keras Geometric Documentation

Welcome to the Keras Geometric documentation! Keras Geometric is a library built on Keras (version 3+) designed for geometric deep learning, with a primary focus on Graph Neural Networks (GNNs).

## What is Keras Geometric?

Keras Geometric provides modular building blocks to easily create and experiment with GNN architectures within the Keras ecosystem. The core philosophy is to offer a flexible and intuitive API, leveraging the power and simplicity of Keras for building complex graph-based models.

## Key Features

- **Flexible Message Passing:** A core `MessagePassing` layer that handles the fundamental logic of neighborhood aggregation, allowing for easy customization of message creation, aggregation, and update steps. Supports various aggregation methods (e.g., 'sum', 'mean', 'max').
- **Standard Graph Convolutions:** Ready-to-use implementations of popular graph convolution layers:
  - `GCNConv`: Graph Convolutional Network layer from Kipf & Welling (2017).
  - `GINConv`: Graph Isomorphism Network layer from Xu et al. (2019).
  - `GATv2Conv`: Graph Attention Network v2 layer from Brody et al. (2021).
  - `SAGEConv`: GraphSAGE layer from Hamilton et al. (2017).
- **Graph Pooling Operations:** Essential pooling layers for graph-level tasks:
  - `GlobalPooling`: Mean, max, and sum pooling for graph representations.
  - `AttentionPooling`: Learnable attention-based pooling.
  - `Set2Set`: Advanced LSTM-based attention pooling.
  - `BatchGlobalPooling`: Efficient pooling for batched graphs.
- **Seamless Keras Integration:** Designed as standard Keras layers, making them easy to integrate into `keras.Sequential` or functional API models.
- **Backend Agnostic:** Leverages Keras 3, allowing compatibility with different backends like TensorFlow, PyTorch, and JAX.
- **Dataset Handling:** Built-in support for common graph datasets and a flexible `GraphData` class for handling graph-structured data.

## Getting Started

- [Installation Guide](installation.md): Learn how to install Keras Geometric and its dependencies.
- [Getting Started](getting_started.md): A quick introduction to the core concepts and basic usage.
- [Tutorials](tutorials/index.md): Step-by-step guides for common GNN tasks.
- [API Reference](api_reference/index.md): Detailed documentation of classes and functions.
- [Examples](examples/index.md): Example notebooks demonstrating usage patterns.

## Citation

If you use Keras Geometric in your research, please cite:

```
@software{keras_geometric,
  author = {Author, A.},
  title = {Keras Geometric: A Graph Neural Network Library for Keras},
  url = {https://github.com/author/keras-geometric},
  version = {0.1.0},
  year = {2023},
}
```

## License

Keras Geometric is released under the MIT License. See the [LICENSE](https://github.com/author/keras-geometric/blob/main/LICENSE) file for details.
