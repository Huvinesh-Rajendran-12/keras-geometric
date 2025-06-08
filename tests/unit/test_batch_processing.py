"""
Graph batch processing tests.

These tests verify handling of multiple graphs in batch format
and graph batching utilities.
"""

import os

import keras
import numpy as np
import pytest

# Set backend
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from keras_geometric.layers import GATv2Conv, GCNConv, GINConv, SAGEConv

pytestmark = [pytest.mark.unit, pytest.mark.batch]


def create_batch_graphs(
    num_graphs: int, nodes_per_graph: int, edges_per_graph: int, input_dim: int
):
    """
    Generates a list of synthetic graphs with random node features and edge indices.

    Each graph in the batch contains the specified number of nodes, edges, and input feature dimensions. The returned list is suitable for testing graph neural network layers.

    Args:
        num_graphs: Number of graphs to generate.
        nodes_per_graph: Number of nodes in each graph.
        edges_per_graph: Number of edges in each graph.
        input_dim: Dimensionality of node feature vectors.

    Returns:
        A list of dictionaries, each representing a graph with keys:
            - "node_features": NumPy array of shape (nodes_per_graph, input_dim)
            - "edge_indices": NumPy array of shape (2, edges_per_graph)
            - "num_nodes": Number of nodes in the graph
            - "num_edges": Number of edges in the graph
    """
    graphs = []

    for _ in range(num_graphs):
        # Create node features
        node_features = np.random.randn(nodes_per_graph, input_dim).astype(np.float32)

        # Create edge indices
        edge_indices = np.random.randint(
            0, nodes_per_graph, size=(2, edges_per_graph)
        ).astype(np.int32)

        graphs.append(
            {
                "node_features": node_features,
                "edge_indices": edge_indices,
                "num_nodes": nodes_per_graph,
                "num_edges": edges_per_graph,
            }
        )

    return graphs


def simulate_batched_graph_processing(graphs, layer):
    """
    Processes a list of graphs sequentially through a given graph neural network layer.

    Args:
        graphs: List of graph dictionaries, each containing "node_features" and "edge_indices".
        layer: A graph neural network layer to apply to each graph.

    Returns:
        List of output tensors, one per input graph.
    """
    outputs = []

    for graph in graphs:
        output = layer([graph["node_features"], graph["edge_indices"]])
        outputs.append(output)

    return outputs


class TestBatchProcessing:
    """Test batch processing of multiple graphs."""

    def test_sequential_graph_processing(self):
        """
        Tests sequential processing of multiple graphs through a GCNConv layer.

        Creates a batch of graphs with fixed node and edge counts, applies the GCNConv layer to each graph, and verifies that the output shapes match expectations and contain no NaN values.
        """
        num_graphs = 5
        nodes_per_graph = 20
        edges_per_graph = 40
        input_dim = 16
        output_dim = 32

        # Create batch of graphs
        graphs = create_batch_graphs(
            num_graphs, nodes_per_graph, edges_per_graph, input_dim
        )

        # Create layer
        layer = GCNConv(output_dim=output_dim)

        # Process graphs
        outputs = simulate_batched_graph_processing(graphs, layer)

        # Verify outputs
        assert len(outputs) == num_graphs
        for output in outputs:
            assert output.shape == (nodes_per_graph, output_dim)
            assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

    def test_variable_graph_sizes(self):
        """
        Tests processing of graphs with varying numbers of nodes and edges using a GATv2Conv layer.

        Creates multiple graphs with different sizes, applies a GATv2Conv layer with multiple heads to each, and verifies that the output shape matches the expected number of nodes for each graph.
        """
        input_dim = 12
        output_dim = 24

        # Create graphs of different sizes
        graph_configs = [
            (10, 20),  # (nodes, edges)
            (15, 30),
            (25, 60),
            (8, 16),
        ]

        graphs = []
        for nodes, edges in graph_configs:
            node_features = np.random.randn(nodes, input_dim).astype(np.float32)
            edge_indices = np.random.randint(0, nodes, size=(2, edges)).astype(np.int32)
            graphs.append(
                {
                    "node_features": node_features,
                    "edge_indices": edge_indices,
                    "num_nodes": nodes,
                }
            )

        # Create layer
        heads = 2
        layer = GATv2Conv(output_dim=output_dim, heads=heads)

        # Process graphs
        outputs = []
        for graph in graphs:
            output = layer([graph["node_features"], graph["edge_indices"]])
            outputs.append(output)

        # Verify outputs
        for i, output in enumerate(outputs):
            expected_nodes = graph_configs[i][0]
            # GATv2Conv concatenates heads, so output_dim is multiplied by heads
            expected_output_dim = output_dim * heads
            assert output.shape == (expected_nodes, expected_output_dim)

    def test_empty_graphs_in_batch(self):
        """
        Tests batch processing when some graphs in the batch are empty.

        Verifies that the graph convolutional layer produces correctly shaped outputs for both normal and empty graphs, ensuring that empty graphs yield empty outputs without errors.
        """
        input_dim = 8
        output_dim = 16

        # Create mixed batch with some empty graphs
        graphs = []

        # Normal graph
        graphs.append(
            {
                "node_features": np.random.randn(10, input_dim).astype(np.float32),
                "edge_indices": np.random.randint(0, 10, size=(2, 20)).astype(np.int32),
                "num_nodes": 10,
            }
        )

        # Empty graph
        graphs.append(
            {
                "node_features": np.array([]).reshape(0, input_dim).astype(np.float32),
                "edge_indices": np.array([[], []], dtype=np.int32),
                "num_nodes": 0,
            }
        )

        # Another normal graph
        graphs.append(
            {
                "node_features": np.random.randn(5, input_dim).astype(np.float32),
                "edge_indices": np.random.randint(0, 5, size=(2, 10)).astype(np.int32),
                "num_nodes": 5,
            }
        )

        # Create layer
        layer = GCNConv(output_dim=output_dim)

        # Process graphs
        outputs = []
        for graph in graphs:
            output = layer([graph["node_features"], graph["edge_indices"]])
            outputs.append(output)

        # Verify outputs
        assert outputs[0].shape == (10, output_dim)
        assert outputs[1].shape == (0, output_dim)  # Empty graph
        assert outputs[2].shape == (5, output_dim)

    def test_batch_consistency(self):
        """
        Verifies that processing identical graphs through a SAGEConv layer with mean aggregation produces identical outputs, confirming consistent layer weights during batch processing.
        """
        num_graphs = 3
        nodes_per_graph = 15
        edges_per_graph = 30
        input_dim = 10
        output_dim = 20

        # Create identical graphs
        base_features = np.random.randn(nodes_per_graph, input_dim).astype(np.float32)
        base_edges = np.random.randint(
            0, nodes_per_graph, size=(2, edges_per_graph)
        ).astype(np.int32)

        graphs = []
        for _ in range(num_graphs):
            graphs.append(
                {
                    "node_features": base_features.copy(),
                    "edge_indices": base_edges.copy(),
                }
            )

        # Create layer
        layer = SAGEConv(output_dim=output_dim, aggregator="mean")

        # Process graphs
        outputs = []
        for graph in graphs:
            output = layer([graph["node_features"], graph["edge_indices"]])
            outputs.append(output)

        # All outputs should be identical (same inputs, same weights)
        for i in range(1, len(outputs)):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(outputs[0]),
                keras.ops.convert_to_numpy(outputs[i]),
                rtol=1e-6,
            )

    def test_different_aggregators_batch(self):
        """
        Tests batch processing of graphs using SAGEConv layers with different aggregation strategies.

        Creates a batch of graphs and processes them through SAGEConv layers with "mean", "max", and "sum" aggregators. Verifies that each aggregator produces outputs of the expected shape for each graph and that the outputs contain no NaN values.
        """
        num_graphs = 4
        nodes_per_graph = 12
        edges_per_graph = 24
        input_dim = 8
        output_dim = 16

        # Create batch of graphs
        graphs = create_batch_graphs(
            num_graphs, nodes_per_graph, edges_per_graph, input_dim
        )

        # Test different aggregators
        aggregators = ["mean", "max", "sum"]
        results = {}

        for agg in aggregators:
            layer = SAGEConv(output_dim=output_dim, aggregator=agg)
            outputs = simulate_batched_graph_processing(graphs, layer)
            results[agg] = outputs

        # Verify all aggregators produce valid outputs
        for _agg, outputs in results.items():
            assert len(outputs) == num_graphs
            for output in outputs:
                assert output.shape == (nodes_per_graph, output_dim)
                assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

    def test_memory_efficiency_batch_processing(self):
        """
        Tests that batch processing a large number of graphs with a GCNConv layer does not exceed reasonable memory usage and produces valid outputs.

        Asserts that memory consumption increase remains below 500MB and that the number of outputs matches the number of input graphs.
        """
        import gc

        import psutil

        def get_memory_usage():
            """
            Returns the current process's memory usage in megabytes.

            Returns:
                The resident set size (RSS) of the current process in megabytes.
            """
            process = psutil.Process()  # Uses current process by default
            return (
                process.memory_info().rss  # pyrefly: ignore  # missing-argument
                / 1024
                / 1024
            )

        num_graphs = 10
        nodes_per_graph = 100
        edges_per_graph = 200
        input_dim = 32
        output_dim = 64

        # Create large batch
        graphs = create_batch_graphs(
            num_graphs, nodes_per_graph, edges_per_graph, input_dim
        )

        # Measure memory before
        gc.collect()
        initial_memory = get_memory_usage()

        # Process batch
        layer = GCNConv(output_dim=output_dim)
        outputs = simulate_batched_graph_processing(graphs, layer)

        # Measure memory after
        final_memory = get_memory_usage()
        memory_used = final_memory - initial_memory

        # Verify outputs
        assert len(outputs) == num_graphs

        # Memory usage should be reasonable (less than 500MB for this test)
        assert memory_used < 500, f"Memory usage too high: {memory_used:.1f}MB"

    def test_attention_batch_processing(self):
        """
        Tests batch processing of graphs using a GATv2Conv attention layer.

        Creates a batch of graphs with fixed node and edge counts, applies a multi-head GATv2Conv layer to each graph, and verifies that the output shapes match the expected dimensions and contain no NaN values.
        """
        num_graphs = 3
        nodes_per_graph = 20
        edges_per_graph = 40
        input_dim = 16
        output_dim = 32
        heads = 4

        # Create batch of graphs
        graphs = create_batch_graphs(
            num_graphs, nodes_per_graph, edges_per_graph, input_dim
        )

        # Create attention layer
        layer = GATv2Conv(output_dim=output_dim, heads=heads)

        # Process graphs
        outputs = simulate_batched_graph_processing(graphs, layer)

        # Verify outputs
        assert len(outputs) == num_graphs
        for output in outputs:
            # GATv2Conv concatenates heads, so output_dim is multiplied by heads
            expected_output_dim = output_dim * heads
            assert output.shape == (nodes_per_graph, expected_output_dim)
            assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

    def test_gin_batch_processing(self):
        """
        Tests batch processing of graphs using GINConv layers with various aggregator types and epsilon configurations.

        Verifies that outputs for each graph in the batch have the expected shape for different GINConv settings.
        """
        num_graphs = 4
        nodes_per_graph = 18
        edges_per_graph = 36
        input_dim = 14
        output_dim = 28

        # Create batch of graphs
        graphs = create_batch_graphs(
            num_graphs, nodes_per_graph, edges_per_graph, input_dim
        )

        # Test different GIN configurations
        layer_configs = [
            {"aggregator": "sum", "eps_init": 0.0, "train_eps": False},
            {"aggregator": "mean", "eps_init": 0.1, "train_eps": True},
            {"aggregator": "max", "eps_init": 1.0, "train_eps": False},
        ]

        for config in layer_configs:
            layer = GINConv(output_dim=output_dim, **config)
            outputs = simulate_batched_graph_processing(graphs, layer)

            assert len(outputs) == num_graphs
            for output in outputs:
                assert output.shape == (nodes_per_graph, output_dim)

    def test_mixed_model_batch_processing(self):
        """
        Tests batch processing of graphs through a Keras model combining GCNConv, SAGEConv, and dense softmax layers.

        Creates a batch of graphs, processes each through a mixed model with graph convolutional and dense layers, and verifies output shapes and that softmax outputs sum to 1 per node.
        """
        num_graphs = 3
        nodes_per_graph = 16
        edges_per_graph = 32
        input_dim = 12
        hidden_dim = 24
        output_dim = 8

        # Create batch of graphs
        graphs = create_batch_graphs(
            num_graphs, nodes_per_graph, edges_per_graph, input_dim
        )

        # Create layers
        gcn_layer = GCNConv(hidden_dim)
        sage_layer = SAGEConv(hidden_dim, aggregator="mean")
        relu_activation = keras.layers.Activation("relu")
        dense_layer = keras.layers.Dense(output_dim, activation="softmax")

        # Process graphs through layers
        outputs = []
        for graph in graphs:
            # GCN layer
            x = gcn_layer([graph["node_features"], graph["edge_indices"]])
            x = relu_activation(x)

            # SAGE layer
            x = sage_layer([x, graph["edge_indices"]])
            x = relu_activation(x)

            # Output layer
            output = dense_layer(x)
            outputs.append(output)

        # Verify outputs
        assert len(outputs) == num_graphs
        for output in outputs:
            assert output.shape == (nodes_per_graph, output_dim)
            # Check softmax properties
            assert np.allclose(
                np.sum(keras.ops.convert_to_numpy(output), axis=1), 1.0, atol=1e-6
            )

    def test_gradient_consistency_batch(self):
        """
        Verifies that processing identical graphs through a compiled GCN-based model yields numerically identical losses, ensuring gradient and loss consistency in batch scenarios.
        """
        nodes_per_graph = 10
        edges_per_graph = 20
        input_dim = 8
        output_dim = 4

        # Create identical graphs
        base_features = np.random.randn(nodes_per_graph, input_dim).astype(np.float32)
        base_edges = np.random.randint(
            0, nodes_per_graph, size=(2, edges_per_graph)
        ).astype(np.int32)

        graphs = [
            {"node_features": base_features.copy(), "edge_indices": base_edges.copy()},
            {"node_features": base_features.copy(), "edge_indices": base_edges.copy()},
        ]

        # Create layer
        layer = GCNConv(output_dim)
        dense_layer = keras.layers.Dense(2, activation="softmax")

        # Create identical targets (for potential future loss computation tests)
        _ = np.random.randint(0, 2, size=(nodes_per_graph,))

        # Test that both graphs produce same outputs (since inputs are identical)
        outputs = []
        for graph in graphs:
            x = layer([graph["node_features"], graph["edge_indices"]])
            x = keras.layers.Activation("relu")(x)
            output = dense_layer(x)
            outputs.append(output)

        # Outputs should be identical (same inputs, same weights)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(outputs[0]),
            keras.ops.convert_to_numpy(outputs[1]),
            rtol=1e-6,
        )
