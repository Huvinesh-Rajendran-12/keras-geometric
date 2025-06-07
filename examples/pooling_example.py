"""
Example demonstrating graph pooling operations for graph-level tasks.

This example shows how to use different pooling layers to create
graph-level representations from node features.
"""

import os

import keras
import numpy as np

# Set backend
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from keras_geometric.layers import GCNConv
from keras_geometric.layers.pooling import (
    AttentionPooling,
    BatchGlobalPooling,
    GlobalPooling,
    Set2Set,
)


def create_sample_graph(
    num_nodes=50, num_features=16, num_edges=100
) -> tuple[np.ndarray, np.ndarray]:
    """Create a sample graph for demonstration."""
    # Random node features
    node_features = np.random.randn(num_nodes, num_features).astype(np.float32)

    # Random edge connections
    edge_indices = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int32)

    return node_features, edge_indices


def demo_global_pooling() -> None:
    """Demonstrate GlobalPooling with different aggregation methods."""
    print("=" * 60)
    print("GLOBAL POOLING DEMO")
    print("=" * 60)

    # Create sample graph
    node_features, edge_indices = create_sample_graph()
    node_features = keras.ops.convert_to_tensor(node_features)
    edge_indices = keras.ops.convert_to_tensor(edge_indices)

    print(f"Input graph: {node_features.shape[0]} nodes, {edge_indices.shape[1]} edges")
    print(f"Node features shape: {node_features.shape}")

    # Test different pooling methods
    pooling_methods = ["mean", "max", "sum"]

    for method in pooling_methods:
        pooling_layer = GlobalPooling(pooling=method)
        graph_repr = pooling_layer(node_features)

        print(
            f"{method.capitalize():4s} pooling: {node_features.shape} -> {graph_repr.shape}"
        )

    print()


def demo_attention_pooling() -> None:
    """Demonstrate attention-based pooling."""
    print("=" * 60)
    print("ATTENTION POOLING DEMO")
    print("=" * 60)

    # Create sample graph
    node_features, edge_indices = create_sample_graph(num_nodes=30, num_features=24)
    node_features = keras.ops.convert_to_tensor(node_features)
    edge_indices = keras.ops.convert_to_tensor(edge_indices)

    print(f"Input graph: {node_features.shape[0]} nodes")
    print(f"Node features shape: {node_features.shape}")

    # Simple attention pooling
    att_pool = AttentionPooling(attention_dim=16)
    graph_repr = att_pool(node_features)
    print(f"AttentionPooling: {node_features.shape} -> {graph_repr.shape}")

    # Set2Set pooling (more advanced)
    set2set_pool = Set2Set(output_dim=12, processing_steps=3)
    graph_repr_s2s = set2set_pool(node_features)
    print(f"Set2Set:          {node_features.shape} -> {graph_repr_s2s.shape}")

    print()


def demo_batch_pooling() -> None:
    """Demonstrate batch pooling for multiple graphs."""
    print("=" * 60)
    print("BATCH POOLING DEMO")
    print("=" * 60)

    # Create batch of graphs with different sizes
    graph_sizes = [20, 35, 15]  # 3 graphs with different number of nodes
    total_nodes = sum(graph_sizes)
    num_features = 8

    # Create concatenated node features
    node_features = np.random.randn(total_nodes, num_features).astype(np.float32)

    # Create batch indices indicating which graph each node belongs to
    batch_indices = []
    for graph_id, size in enumerate(graph_sizes):
        batch_indices.extend([graph_id] * size)
    batch_indices = np.array(batch_indices, dtype=np.int32)

    # Convert to tensors
    node_features = keras.ops.convert_to_tensor(node_features)
    batch_indices = keras.ops.convert_to_tensor(batch_indices)

    print(f"Batch of {len(graph_sizes)} graphs:")
    print(f"Graph sizes: {graph_sizes}")
    print(f"Total nodes: {total_nodes}")
    print(f"Node features shape: {node_features.shape}")
    print(f"Batch indices shape: {batch_indices.shape}")

    # Apply batch pooling
    batch_pool = BatchGlobalPooling(pooling="mean")
    batch_graph_reprs = batch_pool([node_features, batch_indices])

    print(f"Batch pooling: -> {batch_graph_reprs.shape}")
    print(
        f"Each graph now has a {batch_graph_reprs.shape[1]}-dimensional representation"
    )

    print()


def demo_graph_classification_pipeline() -> None:
    """Demonstrate complete graph classification pipeline with pooling."""
    print("=" * 60)
    print("GRAPH CLASSIFICATION PIPELINE")
    print("=" * 60)

    # Create sample graph
    node_features, edge_indices = create_sample_graph(num_nodes=40, num_features=12)
    node_features = keras.ops.convert_to_tensor(node_features)
    edge_indices = keras.ops.convert_to_tensor(edge_indices)

    print(f"Input graph: {node_features.shape[0]} nodes, {edge_indices.shape[1]} edges")

    # Build a graph classification model
    hidden_dim = 32
    num_classes = 3

    # GNN layers for node embeddings
    gcn1 = GCNConv(hidden_dim, use_bias=True)
    gcn2 = GCNConv(hidden_dim, use_bias=True)

    # Pooling layer for graph representation
    pooling_layer = AttentionPooling(attention_dim=16)

    # Classification head
    classifier = keras.layers.Dense(num_classes, activation="softmax")

    # Forward pass
    print("\\nForward pass through graph classification model:")

    # Node-level processing
    x = gcn1([node_features, edge_indices])
    x = keras.ops.relu(x)
    print(f"After GCN1: {x.shape}")

    x = gcn2([x, edge_indices])
    x = keras.ops.relu(x)
    print(f"After GCN2: {x.shape}")

    # Graph-level pooling
    graph_repr = pooling_layer(x)
    print(f"After pooling: {graph_repr.shape}")

    # Classification
    predictions = classifier(graph_repr)
    print(f"Final predictions: {predictions.shape}")

    # Verify probabilities sum to 1
    prob_sum = keras.ops.sum(predictions)
    print(f"Probability sum: {keras.ops.convert_to_numpy(prob_sum):.6f}")

    print()


def demo_different_pooling_comparison() -> None:
    """Compare different pooling methods on the same graph."""
    print("=" * 60)
    print("POOLING METHODS COMPARISON")
    print("=" * 60)

    # Create sample graph
    node_features, edge_indices = create_sample_graph(num_nodes=25, num_features=16)
    node_features = keras.ops.convert_to_tensor(node_features)

    print(
        f"Input: {node_features.shape[0]} nodes with {node_features.shape[1]} features each"
    )

    # Apply GCN to get meaningful node embeddings first
    gcn = GCNConv(32, use_bias=True)
    node_embeddings = gcn([node_features, keras.ops.convert_to_tensor(edge_indices)])
    node_embeddings = keras.ops.relu(node_embeddings)

    print(f"After GCN: {node_embeddings.shape}")
    print("\\nApplying different pooling methods:")

    # Test different pooling methods
    pooling_layers = [
        ("Global Mean", GlobalPooling(pooling="mean")),
        ("Global Max", GlobalPooling(pooling="max")),
        ("Global Sum", GlobalPooling(pooling="sum")),
        ("Attention", AttentionPooling(attention_dim=16)),
        ("Set2Set", Set2Set(output_dim=16, processing_steps=2)),
    ]

    results = {}
    for name, layer in pooling_layers:
        graph_repr = layer(node_embeddings)
        results[name] = graph_repr
        print(f"{name:12s}: {node_embeddings.shape} -> {graph_repr.shape}")

    # Show some statistics
    print("\\nPooling output statistics:")
    for name, repr_tensor in results.items():
        repr_numpy = keras.ops.convert_to_numpy(repr_tensor)
        mean_val = np.mean(repr_numpy)
        std_val = np.std(repr_numpy)
        print(f"{name:12s}: mean={mean_val:6.3f}, std={std_val:6.3f}")

    print()


def main() -> None:
    """Run all pooling demonstrations."""
    print("KERAS GEOMETRIC POOLING EXAMPLES")
    print("This example demonstrates various graph pooling operations")
    print("for creating graph-level representations from node features.")
    print()

    # Run all demos
    demo_global_pooling()
    demo_attention_pooling()
    demo_batch_pooling()
    demo_graph_classification_pipeline()
    demo_different_pooling_comparison()

    print("=" * 60)
    print("All pooling examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
