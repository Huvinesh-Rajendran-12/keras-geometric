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
    """Create a batch of graphs for testing."""
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
    """Simulate processing multiple graphs with a single layer."""
    outputs = []

    for graph in graphs:
        output = layer([graph["node_features"], graph["edge_indices"]])
        outputs.append(output)

    return outputs


class TestBatchProcessing:
    """Test batch processing of multiple graphs."""

    def test_sequential_graph_processing(self):
        """Test processing multiple graphs sequentially."""
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
            assert not np.any(np.isnan(output.numpy()))

    def test_variable_graph_sizes(self):
        """Test processing graphs of different sizes."""
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
        layer = GATv2Conv(output_dim=output_dim, heads=2)

        # Process graphs
        outputs = []
        for graph in graphs:
            output = layer([graph["node_features"], graph["edge_indices"]])
            outputs.append(output)

        # Verify outputs
        for i, output in enumerate(outputs):
            expected_nodes = graph_configs[i][0]
            assert output.shape == (expected_nodes, output_dim)

    def test_empty_graphs_in_batch(self):
        """Test handling of empty graphs in a batch."""
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
        """Test that layer weights remain consistent across batch processing."""
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
                outputs[0].numpy(), outputs[i].numpy(), rtol=1e-6
            )

    def test_different_aggregators_batch(self):
        """Test different aggregators on the same batch of graphs."""
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
                assert not np.any(np.isnan(output.numpy()))

    def test_memory_efficiency_batch_processing(self):
        """Test memory efficiency of batch processing."""
        import gc

        import psutil

        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

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
        """Test attention mechanisms with batch processing."""
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
            assert output.shape == (nodes_per_graph, output_dim)
            assert not np.any(np.isnan(output.numpy()))

    def test_gin_batch_processing(self):
        """Test GIN layers with batch processing."""
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
            {"aggregator": "sum", "epsilon": 0.0, "train_eps": False},
            {"aggregator": "mean", "epsilon": 0.1, "train_eps": True},
            {"aggregator": "max", "epsilon": 1.0, "train_eps": False},
        ]

        for config in layer_configs:
            layer = GINConv(output_dim=output_dim, **config)
            outputs = simulate_batched_graph_processing(graphs, layer)

            assert len(outputs) == num_graphs
            for output in outputs:
                assert output.shape == (nodes_per_graph, output_dim)

    def test_mixed_model_batch_processing(self):
        """Test batch processing with models containing multiple layer types."""
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

        # Build mixed model
        def create_mixed_model():
            node_input = keras.Input(shape=(input_dim,))
            edge_input = keras.Input(shape=(2, None))

            # GCN layer
            x = GCNConv(hidden_dim)([node_input, edge_input])
            x = keras.layers.Activation("relu")(x)

            # SAGE layer
            x = SAGEConv(hidden_dim, aggregator="mean")([x, edge_input])
            x = keras.layers.Activation("relu")(x)

            # Output layer
            outputs = keras.layers.Dense(output_dim, activation="softmax")(x)

            return keras.Model(inputs=[node_input, edge_input], outputs=outputs)

        model = create_mixed_model()

        # Process graphs through model
        outputs = []
        for graph in graphs:
            output = model([graph["node_features"], graph["edge_indices"]])
            outputs.append(output)

        # Verify outputs
        assert len(outputs) == num_graphs
        for output in outputs:
            assert output.shape == (nodes_per_graph, output_dim)
            # Check softmax properties
            assert np.allclose(np.sum(output.numpy(), axis=1), 1.0, atol=1e-6)

    def test_gradient_consistency_batch(self):
        """Test that gradients are consistent across batch processing."""
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

        # Create model
        node_input = keras.Input(shape=(input_dim,))
        edge_input = keras.Input(shape=(2, edges_per_graph))

        x = GCNConv(output_dim)([node_input, edge_input])
        x = keras.layers.Activation("relu")(x)
        outputs = keras.layers.Dense(2, activation="softmax")(x)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Create identical targets
        targets = np.random.randint(0, 2, size=(nodes_per_graph,))

        # Test that both graphs produce same loss
        losses = []
        for graph in graphs:
            loss = model.evaluate(
                [graph["node_features"], graph["edge_indices"]], targets, verbose=0
            )
            losses.append(loss)

        # Losses should be identical
        np.testing.assert_allclose(losses[0], losses[1], rtol=1e-6)
