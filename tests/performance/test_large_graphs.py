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
    """
    Returns the current process memory usage in megabytes.
    Returns:
        The resident set size (RSS) of the current process in MB.
    """
    process = psutil.Process(os.getpid())  # pyrefly: ignore  # missing-argument
    return process.memory_info().rss / 1024 / 1024


def create_large_graph(
    num_nodes: int, avg_degree: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a random graph with the specified number of nodes and average degree.

    The function returns node features as a float32 NumPy array of shape (num_nodes, 64) and edge indices as an int32 NumPy array of shape (2, num_edges), where num_edges = num_nodes * avg_degree.
    """
    num_edges = num_nodes * avg_degree

    # Create random edges
    edge_indices = np.random.randint(0, num_nodes, size=(2, num_edges)).astype(np.int32)

    # Create node features
    input_dim = 64
    node_features = np.random.randn(num_nodes, input_dim).astype(np.float32)

    return node_features, edge_indices


class TestLargeGraphPerformance:
    """Test performance with large graphs."""

    def __init__(self):
        """
        Initializes the test class and sets the initial memory usage attribute to 0.0.
        """
        self.initial_memory: float = (
            0.0  # pyrefly: ignore  # implicitly-defined-attribute
        )

    def setup_method(self):
        """
        Prepares the test environment by running garbage collection and recording initial memory usage before each test.
        """
        gc.collect()
        self.initial_memory = get_memory_usage()

    def teardown_method(self):
        """
        Performs cleanup after each test by triggering garbage collection.
        """
        gc.collect()

    @pytest.mark.slow
    def test_gcn_large_graph_performance(self):
        """
        Evaluates the forward pass performance of a GCNConv layer on a large graph with 10,000 nodes.

        Asserts that the output shape is correct, contains no NaN values, the forward pass completes in under 5 seconds, and memory usage increase remains below 2000 MB.
        """
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
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

        # Performance assertions
        assert forward_time < 5.0, (
            f"Forward pass took {forward_time:.2f}s, should be < 5s"
        )

        memory_used = get_memory_usage() - self.initial_memory
        assert memory_used < 2000, f"Memory usage {memory_used:.1f}MB too high"

    @pytest.mark.slow
    def test_gatv2_large_graph_performance(self):
        """
        Evaluates the performance of the GATv2Conv layer on a large graph with 5,000 nodes.

        Measures the forward pass execution time and memory usage, asserting that the output shape is correct, contains no NaN values, the forward pass completes in under 10 seconds, and memory usage increase remains below 3,000 MB.
        """
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
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

        # Performance assertions (attention is more expensive)
        assert forward_time < 10.0, (
            f"Forward pass took {forward_time:.2f}s, should be < 10s"
        )

        memory_used = get_memory_usage() - self.initial_memory
        assert memory_used < 3000, f"Memory usage {memory_used:.1f}MB too high"

    @pytest.mark.slow
    @pytest.mark.parametrize("aggregator", ["mean", "max", "sum"])
    def test_sage_large_graph_aggregators(self, aggregator: str):
        """
        Evaluates the performance of the SAGEConv layer with a specified aggregator on a large graph.

        Runs a forward pass of SAGEConv using the given aggregator on an 8,000-node graph, verifying output shape, absence of NaNs, and that the forward pass completes in under 8 seconds.
        """
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
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

        # Performance assertions
        assert forward_time < 8.0, (
            f"Forward pass took {forward_time:.2f}s, should be < 8s"
        )

    def test_memory_scaling_by_nodes(self):
        """
        Evaluates memory usage scaling as the number of nodes increases in a GCN layer.

        Runs the GCNConv layer on graphs with varying node counts and measures memory usage before and after the forward pass. Asserts that memory usage per node does not increase more than threefold between the smallest and largest graph, indicating reasonable scaling behavior.
        """
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
        """
        Evaluates memory usage scaling with increasing edge density in a GCN layer.

        Creates graphs with a fixed number of nodes and varying average degrees, measures memory usage before and after a forward pass through a GCNConv layer, and asserts that memory usage for the densest graph does not exceed five times that of the sparsest.
        """
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
        """
        Evaluates the forward pass performance of a three-layer GCN model on a large graph.

        Creates a graph with 6,000 nodes, constructs a model with three stacked GCNConv layers, and measures the total forward pass time. Asserts that the output shape is correct, contains no NaNs, and that the forward pass completes in under 15 seconds.
        """
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
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(x)))

        # Performance assertions
        assert forward_time < 15.0, f"Multi-layer forward pass took {forward_time:.2f}s"

    def test_batch_processing_simulation(self):
        """
        Tests the performance of processing multiple small graphs sequentially to simulate batch processing.

        Creates and processes 10 small graphs using a GCNConv layer with ReLU activation, measuring total and average processing time. Verifies output shapes, absence of NaNs, and asserts that the average processing time per graph is under 1 second.
        """
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
            assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

        # Performance assertion
        avg_time_per_graph = total_time / num_graphs
        assert avg_time_per_graph < 1.0, (
            f"Average time per graph: {avg_time_per_graph:.3f}s"
        )

    def test_gradient_computation_large_graph(self):
        """
        Measures the training time for one epoch of gradient computation on a large graph using a Keras model with a GCN layer.

        Asserts that the gradient computation completes within 20 seconds for a graph with 5,000 nodes and 10 output classes.
        """
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
        """
        Evaluates the forward pass performance of a GCN layer on a very sparse large graph.

        Creates a graph with 20,000 nodes and low average degree, runs a GCN layer, and asserts that the output shape is correct and the forward pass completes in under 3 seconds.
        """
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
        """
        Evaluates the forward pass performance of a GCN layer on a dense graph.

        Creates a graph with 1,000 nodes and high average degree, runs a GCNConv layer, and asserts that the output shape is correct and the forward pass completes in under 8 seconds.
        """
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
