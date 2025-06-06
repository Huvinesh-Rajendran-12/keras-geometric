"""
Performance tests for large graphs and memory usage.

These tests verify that GNN layers can handle large graphs efficiently
and measure memory usage patterns.
"""

import gc
import os
import time

import keras
import numpy as np
import psutil
import pytest

# Set backend
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from keras_geometric.layers import GATv2Conv, GCNConv, SAGEConv

pytestmark = [pytest.mark.performance, pytest.mark.slow]


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_large_graph(
    num_nodes: int, avg_degree: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Create a large random graph."""
    num_edges = num_nodes * avg_degree

    # Create random edges
    edge_indices = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int32)

    # Create node features
    input_dim = 64
    node_features = np.random.randn(num_nodes, input_dim).astype(np.float32)

    return node_features, edge_indices


class TestLargeGraphPerformance:
    """Test performance with large graphs."""

    def setup_method(self):
        """Setup before each test."""
        gc.collect()
        self.initial_memory = get_memory_usage()

    def teardown_method(self):
        """Cleanup after each test."""
        gc.collect()

    @pytest.mark.slow
    def test_gcn_large_graph_performance(self):
        """Test GCNConv performance with large graphs."""
        num_nodes = 10000
        output_dim = 128

        # Create large graph
        node_features, edge_indices = create_large_graph(num_nodes)

        # Create layer
        layer = GCNConv(output_dim=output_dim)

        # Measure forward pass time
        start_time = time.time()
        output = layer([node_features, edge_indices])
        forward_time = time.time() - start_time

        # Verify output
        assert output.shape == (num_nodes, output_dim)
        assert not np.any(np.isnan(output.numpy()))

        # Performance assertions
        assert (
            forward_time < 5.0
        ), f"Forward pass took {forward_time:.2f}s, should be < 5s"

        memory_used = get_memory_usage() - self.initial_memory
        assert memory_used < 2000, f"Memory usage {memory_used:.1f}MB too high"

    @pytest.mark.slow
    def test_gatv2_large_graph_performance(self):
        """Test GATv2Conv performance with large graphs."""
        num_nodes = 5000  # Smaller due to attention complexity
        output_dim = 64
        heads = 4

        # Create large graph
        node_features, edge_indices = create_large_graph(num_nodes)

        # Create layer
        layer = GATv2Conv(output_dim=output_dim, heads=heads)

        # Measure forward pass time
        start_time = time.time()
        output = layer([node_features, edge_indices])
        forward_time = time.time() - start_time

        # Verify output
        assert output.shape == (num_nodes, output_dim)
        assert not np.any(np.isnan(output.numpy()))

        # Performance assertions (attention is more expensive)
        assert (
            forward_time < 10.0
        ), f"Forward pass took {forward_time:.2f}s, should be < 10s"

        memory_used = get_memory_usage() - self.initial_memory
        assert memory_used < 3000, f"Memory usage {memory_used:.1f}MB too high"

    @pytest.mark.slow
    @pytest.mark.parametrize("aggregator", ["mean", "max", "sum"])
    def test_sage_large_graph_aggregators(self, aggregator: str):
        """Test SAGEConv with different aggregators on large graphs."""
        num_nodes = 8000
        output_dim = 96

        # Create large graph
        node_features, edge_indices = create_large_graph(num_nodes)

        # Create layer
        layer = SAGEConv(output_dim=output_dim, aggregator=aggregator)

        # Measure forward pass time
        start_time = time.time()
        output = layer([node_features, edge_indices])
        forward_time = time.time() - start_time

        # Verify output
        assert output.shape == (num_nodes, output_dim)
        assert not np.any(np.isnan(output.numpy()))

        # Performance assertions
        assert (
            forward_time < 8.0
        ), f"Forward pass took {forward_time:.2f}s, should be < 8s"

    def test_memory_scaling_by_nodes(self):
        """Test how memory usage scales with number of nodes."""
        output_dim = 32
        memory_measurements = []

        node_counts = [1000, 2000, 4000]

        for num_nodes in node_counts:
            gc.collect()
            initial_mem = get_memory_usage()

            # Create graph
            node_features, edge_indices = create_large_graph(num_nodes, avg_degree=5)

            # Create and run layer
            layer = GCNConv(output_dim=output_dim)
            output = layer([node_features, edge_indices])

            final_mem = get_memory_usage()
            memory_used = final_mem - initial_mem
            memory_measurements.append((num_nodes, memory_used))

            # Cleanup
            del node_features, edge_indices, layer, output
            gc.collect()

        # Check that memory scaling is reasonable (should be roughly linear)
        mem_per_node_1 = memory_measurements[0][1] / memory_measurements[0][0]
        mem_per_node_2 = memory_measurements[-1][1] / memory_measurements[-1][0]

        # Memory per node shouldn't increase dramatically
        assert mem_per_node_2 < mem_per_node_1 * 3, "Memory scaling is too steep"

    def test_memory_scaling_by_edges(self):
        """Test how memory usage scales with number of edges."""
        num_nodes = 2000
        output_dim = 32
        memory_measurements = []

        edge_densities = [5, 10, 20]  # Average degree

        for avg_degree in edge_densities:
            gc.collect()
            initial_mem = get_memory_usage()

            # Create graph
            node_features, edge_indices = create_large_graph(num_nodes, avg_degree)

            # Create and run layer
            layer = GCNConv(output_dim=output_dim)
            output = layer([node_features, edge_indices])

            final_mem = get_memory_usage()
            memory_used = final_mem - initial_mem
            memory_measurements.append((avg_degree, memory_used))

            # Cleanup
            del node_features, edge_indices, layer, output
            gc.collect()

        # Memory should increase with edge density but not excessively
        assert memory_measurements[-1][1] < memory_measurements[0][1] * 5

    @pytest.mark.slow
    def test_multilayer_large_graph_performance(self):
        """Test performance of multi-layer models on large graphs."""
        num_nodes = 6000
        hidden_dim = 64
        num_layers = 3

        # Create large graph
        node_features, edge_indices = create_large_graph(num_nodes)

        # Build multi-layer model
        layers = []
        for _ in range(num_layers):
            layer = GCNConv(hidden_dim)
            layers.append(layer)

        # Measure forward pass time
        start_time = time.time()

        x = node_features
        for layer in layers:
            x = layer([x, edge_indices])

        forward_time = time.time() - start_time

        # Verify output
        assert x.shape == (num_nodes, hidden_dim)
        assert not np.any(np.isnan(x.numpy()))

        # Performance assertions
        assert forward_time < 15.0, f"Multi-layer forward pass took {forward_time:.2f}s"

    def test_batch_processing_simulation(self):
        """Test processing multiple small graphs (simulating batching)."""
        num_graphs = 10
        nodes_per_graph = 500
        output_dim = 32

        # Create multiple small graphs
        graphs = []
        for _ in range(num_graphs):
            node_features, edge_indices = create_large_graph(
                nodes_per_graph, avg_degree=8
            )
            graphs.append((node_features, edge_indices))

        # Create layer
        layer = GCNConv(output_dim=output_dim, activation="relu")

        # Process each graph
        start_time = time.time()

        outputs = []
        for node_features, edge_indices in graphs:
            output = layer([node_features, edge_indices])
            outputs.append(output)

        total_time = time.time() - start_time

        # Verify outputs
        assert len(outputs) == num_graphs
        for output in outputs:
            assert output.shape == (nodes_per_graph, output_dim)
            assert not np.any(np.isnan(output.numpy()))

        # Performance assertion
        avg_time_per_graph = total_time / num_graphs
        assert (
            avg_time_per_graph < 1.0
        ), f"Average time per graph: {avg_time_per_graph:.3f}s"

    def test_gradient_computation_large_graph(self):
        """Test gradient computation performance on large graphs."""
        num_nodes = 5000
        hidden_dim = 64
        num_classes = 10

        # Create large graph
        node_features, edge_indices = create_large_graph(num_nodes)

        # Build model
        node_input = keras.Input(shape=(node_features.shape[1],))
        edge_input = keras.Input(shape=(2, edge_indices.shape[1]))

        x = GCNConv(hidden_dim)([node_input, edge_input])
        x = keras.layers.Activation("relu")(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=[node_input, edge_input], outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Create targets
        targets = np.random.randint(0, num_classes, size=(num_nodes,))

        # Measure gradient computation time
        start_time = time.time()

        model.fit([node_features, edge_indices], targets, epochs=1, verbose=0)

        gradient_time = time.time() - start_time

        # Performance assertion
        assert gradient_time < 20.0, f"Gradient computation took {gradient_time:.2f}s"

    @pytest.mark.slow
    def test_very_sparse_graph_performance(self):
        """Test performance with very sparse graphs."""
        num_nodes = 20000
        avg_degree = 3  # Very sparse
        output_dim = 64

        # Create sparse graph
        node_features, edge_indices = create_large_graph(num_nodes, avg_degree)

        # Create layer
        layer = GCNConv(output_dim=output_dim)

        # Measure performance
        start_time = time.time()
        output = layer([node_features, edge_indices])
        forward_time = time.time() - start_time

        # Verify output
        assert output.shape == (num_nodes, output_dim)

        # Sparse graphs should be fast
        assert forward_time < 3.0, f"Sparse graph processing took {forward_time:.2f}s"

    @pytest.mark.slow
    def test_dense_graph_performance(self):
        """Test performance with dense graphs."""
        num_nodes = 1000
        avg_degree = 100  # Very dense
        output_dim = 64

        # Create dense graph
        node_features, edge_indices = create_large_graph(num_nodes, avg_degree)

        # Create layer
        layer = GCNConv(output_dim=output_dim)

        # Measure performance
        start_time = time.time()
        output = layer([node_features, edge_indices])
        forward_time = time.time() - start_time

        # Verify output
        assert output.shape == (num_nodes, output_dim)

        # Dense graphs will be slower but should still be reasonable
        assert forward_time < 8.0, f"Dense graph processing took {forward_time:.2f}s"
