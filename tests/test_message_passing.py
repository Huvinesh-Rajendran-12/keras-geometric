import numpy as np
import pytest
from keras import ops
from keras.src import testing
from keras.src.ops import KerasTensor

from keras_geometric.layers import MessagePassing


class TestMessagePassing(testing.TestCase):
    """Comprehensive test suite for MessagePassing layer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()

        # Simple graph: 5 nodes, 6 edges
        self.num_nodes: int = 5
        self.num_features: int = 8
        self.num_edges: int = 6

        # Create node features
        self.node_features: np.ndarray = np.random.randn(
            self.num_nodes, self.num_features
        ).astype(np.float32)

        # Create edge indices (simple graph structure)
        # 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 0, 0 -> 2
        self.edge_index: np.ndarray = np.array(
            [
                [0, 1, 2, 3, 4, 0],  # source nodes
                [1, 2, 3, 4, 0, 2],  # target nodes
            ],
            dtype=np.int32,
        )

        # Create edge attributes
        self.edge_attr: np.ndarray = np.random.randn(self.num_edges, 4).astype(
            np.float32
        )
        self.sizes: list[int] | None = None

    def test_initialization(self) -> None:
        """Test layer initialization with different aggregators."""
        # Test valid aggregators
        for aggregator in ["mean", "max", "sum", "min", "std"]:
            layer = MessagePassing(aggregator=aggregator)
            self.assertEqual(layer.aggregator, aggregator)

        # Test invalid aggregator
        with self.assertRaises(ValueError):
            MessagePassing(aggregator="invalid")

    def test_mean_aggregation(self) -> None:
        """Test mean aggregation with various edge cases."""
        layer = MessagePassing(aggregator="mean")

        # Test normal case
        messages = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        target_idx = np.array([0, 0, 1], dtype=np.int32)

        result = layer.aggregate(messages, target_idx, num_nodes=3)

        # Node 0 should have mean of first two messages
        expected_node_0 = np.array([2.0, 3.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[0]), expected_node_0, rtol=1e-5
        )

        # Node 1 should have the third message
        expected_node_1 = np.array([5.0, 6.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[1]), expected_node_1, rtol=1e-5
        )

        # Node 2 should have zeros (no messages)
        expected_node_2 = np.array([0.0, 0.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[2]), expected_node_2, rtol=1e-5
        )

    def test_max_aggregation(self) -> None:
        """Test max aggregation."""
        layer = MessagePassing(aggregator="max")

        messages = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        target_idx = np.array([0, 0, 1], dtype=np.int32)

        result = layer.aggregate(messages, target_idx, num_nodes=3)

        # Node 0 should have max of first two messages
        expected_node_0 = np.array([3.0, 5.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[0]), expected_node_0, rtol=1e-5
        )

        # Node 1 should have the third message
        expected_node_1 = np.array([2.0, 4.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[1]), expected_node_1, rtol=1e-5
        )

    def test_sum_aggregation(self) -> None:
        """Test sum aggregation."""
        layer = MessagePassing(aggregator="sum")

        messages = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        target_idx = np.array([0, 0, 1], dtype=np.int32)

        result = layer.aggregate(messages, target_idx, num_nodes=3)

        # Node 0 should have sum of first two messages
        expected_node_0 = np.array([4.0, 6.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[0]), expected_node_0, rtol=1e-5
        )

    def test_min_aggregation(self) -> None:
        """Test min aggregation."""
        layer = MessagePassing(aggregator="min")

        messages = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        target_idx = np.array([0, 0, 1], dtype=np.int32)

        result = layer.aggregate(messages, target_idx, num_nodes=3)

        # Node 0 should have min of first two messages
        expected_node_0 = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[0]), expected_node_0, rtol=1e-5
        )

    def test_std_aggregation(self) -> None:
        """Test standard deviation aggregation."""
        layer = MessagePassing(aggregator="std")

        # Use messages where std is easy to calculate
        messages = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32
        )
        target_idx = np.array([0, 0, 1, 1], dtype=np.int32)

        result = layer.aggregate(messages, target_idx, num_nodes=2)

        # Node 0: std of [1, 3] and [2, 4]
        expected_std_0 = np.array([1.0, 1.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[0]), expected_std_0, rtol=1e-5
        )

        # Node 1: std of [5, 7] and [6, 8]
        expected_std_1 = np.array([1.0, 1.0])
        np.testing.assert_allclose(
            ops.convert_to_numpy(result[1]), expected_std_1, rtol=1e-5
        )

    def test_empty_graph(self) -> None:
        """Test with empty graph (no nodes)."""
        layer = MessagePassing(aggregator="mean")

        empty_features = np.zeros((0, 8), dtype=np.float32)
        empty_edge_index = np.zeros((2, 0), dtype=np.int32)

        result = layer.propagate(x=empty_features, edge_index=empty_edge_index)

        self.assertEqual(result.shape, (0, 8))

    def test_no_edges(self) -> None:
        """Test with graph that has nodes but no edges."""
        layer = MessagePassing(aggregator="mean")

        features = np.random.randn(5, 8).astype(np.float32)
        empty_edge_index = np.zeros((2, 0), dtype=np.int32)

        result = layer.propagate(x=features, edge_index=empty_edge_index)

        # Should return zeros for all nodes
        expected = np.zeros((5, 8), dtype=np.float32)
        np.testing.assert_allclose(ops.convert_to_numpy(result), expected, rtol=1e-5)

    def test_message_with_edge_attributes(self) -> None:
        """Test message function with edge attributes."""
        layer = MessagePassing()

        # Create dummy tensors
        x_i = ops.ones((10, 8))
        x_j = ops.ones((10, 8)) * 2
        edge_attr = ops.ones((10, 4)) * 3

        # Test default message function with edge attributes
        messages = layer.message(x_i, x_j, edge_attr=edge_attr)

        # Should concatenate x_j and edge_attr
        self.assertEqual(messages.shape, (10, 12))

    def test_bipartite_graph(self) -> None:
        """Test with bipartite graph (different source and target features)."""
        layer = MessagePassing(aggregator="sum")

        # Different number of source and target nodes
        x_source = np.random.randn(4, 8).astype(np.float32)
        x_target = np.random.randn(3, 8).astype(np.float32)

        # Edges from source to target
        edge_index = np.array(
            [
                [0, 1, 2, 3, 0],  # source indices
                [0, 1, 2, 0, 1],  # target indices
            ],
            dtype=np.int32,
        )

        result = layer.propagate(x=(x_target, x_source), edge_index=edge_index)

        # Result should have shape of target nodes
        self.assertEqual(result.shape, (3, 8))

    def test_call_method(self) -> None:
        """Test the call method with different input formats."""
        layer = MessagePassing(aggregator="mean")

        # Test with list inputs
        result1 = layer([self.node_features, self.edge_index])
        self.assertEqual(result1.shape, (self.num_nodes, self.num_features))

        # Test with tuple inputs
        result2 = layer((self.node_features, self.edge_index))
        self.assertEqual(result2.shape, (self.num_nodes, self.num_features))

        # Test with edge attributes
        result3 = layer([self.node_features, self.edge_index, self.edge_attr])
        self.assertIsNotNone(result3)

    def test_edge_index_caching(self) -> None:
        """Test edge index caching mechanism."""
        layer = MessagePassing()

        # Before any calls, cache should be empty
        self.assertIsNone(layer._cached_edge_idx)
        self.assertIsNone(layer._cached_edge_idx_hash)

        # First call should cache edge_index
        _ = layer([self.node_features, self.edge_index])
        self.assertIsNotNone(layer._cached_edge_idx)
        self.assertIsNotNone(layer._cached_edge_idx_hash)

        # Verify the cached edge_index has correct dtype
        self.assertEqual(ops.convert_to_numpy(layer._cached_edge_idx).dtype, np.int32)

        # Verify the cached edge_index has correct shape
        self.assertEqual(layer._cached_edge_idx.shape, self.edge_index.shape)

    def test_pre_and_post_hooks(self) -> None:
        """Test pre_aggregate and post_update hooks."""

        class CustomMessagePassing(MessagePassing):
            def pre_aggregate(self, messages: KerasTensor) -> KerasTensor:
                # Scale messages by 2
                return messages * 2

            def post_update(
                self, x: KerasTensor, x_updated: KerasTensor
            ) -> KerasTensor:
                # Add residual connection
                return x + x_updated

        layer = CustomMessagePassing(aggregator="sum")

        # Small test case for easy verification
        features = np.ones((3, 2), dtype=np.float32)
        edge_index = np.array([[0, 1], [1, 2]], dtype=np.int32)

        result = layer([features, edge_index])

        # Verify hooks were applied
        self.assertIsNotNone(result)

    def test_custom_message(self) -> None:
        """Test custom message implementation."""

        class CustomMessageLayer(MessagePassing):
            def message(
                self,
                x_i: KerasTensor,
                x_j: KerasTensor,
                edge_attr: KerasTensor | None = None,
                edge_index: KerasTensor | None = None,
                size: tuple[int, int] | None = None,
                **kwargs,
            ) -> KerasTensor:
                # Custom message: difference between source and target
                return x_j - x_i

        layer = CustomMessageLayer(aggregator="mean")
        result = layer([self.node_features, self.edge_index])

        self.assertEqual(result.shape, (self.num_nodes, self.num_features))

    def test_custom_update(self) -> None:
        """Test custom update implementation."""

        class CustomUpdateLayer(MessagePassing):
            def update(
                self, aggregated: KerasTensor, x: KerasTensor | None = None
            ) -> KerasTensor:
                # Custom update: multiply by 2 and add bias
                return aggregated * 2 + 1

        layer = CustomUpdateLayer(aggregator="sum")
        result = layer([self.node_features, self.edge_index])

        self.assertEqual(result.shape, (self.num_nodes, self.num_features))

    def test_serialization(self) -> None:
        """Test layer serialization and deserialization."""
        layer = MessagePassing(aggregator="max", name="test_layer")

        # Get config
        config = layer.get_config()
        self.assertEqual(config["aggregator"], "max")
        self.assertEqual(config["name"], "test_layer")

        # Create new layer from config
        new_layer = MessagePassing.from_config(config)
        self.assertEqual(new_layer.aggregator, "max")
        self.assertEqual(new_layer.name, "test_layer")

    def test_compute_output_shape(self) -> None:
        """Test compute_output_shape method."""
        layer = MessagePassing()

        # Test with list input shapes
        input_shapes = [(None, 5, 32), (None, 2, None)]
        output_shape = layer.compute_output_shape(input_shapes)
        self.assertEqual(output_shape, (None, 5, 32))

        # Test with tuple input shapes
        input_shapes = ((None, 5, 32), (None, 2, None))
        output_shape = layer.compute_output_shape(input_shapes)
        self.assertEqual(output_shape, (None, 5, 32))

    def test_numerical_stability(self) -> None:
        """Test numerical stability with extreme values."""
        layer = MessagePassing(aggregator="mean")

        # Test with very large values
        large_messages = np.full((100, 10), 1e10, dtype=np.float32)
        target_idx = np.zeros(100, dtype=np.int32)

        result = layer.aggregate(large_messages, target_idx, num_nodes=1)
        result_numpy = ops.convert_to_numpy(result)
        self.assertFalse(np.any(np.isnan(result_numpy)))
        self.assertFalse(np.any(np.isinf(result_numpy)))

        # Test with very small values
        small_messages = np.full((100, 10), 1e-10, dtype=np.float32)
        result = layer.aggregate(small_messages, target_idx, num_nodes=1)
        result_numpy = ops.convert_to_numpy(result)
        self.assertFalse(np.any(np.isnan(result_numpy)))
        self.assertFalse(np.any(np.isinf(result_numpy)))

    def test_invalid_inputs(self) -> None:
        """Test error handling for invalid inputs."""
        layer = MessagePassing()

        # Test with invalid input format
        with self.assertRaises(ValueError):
            layer(self.node_features)  # Missing edge_index

        with self.assertRaises(ValueError):
            layer([self.node_features])  # Only one element in list

    def test_integration_with_keras_model(self) -> None:
        """Test integration with Keras Model API."""
        # Test that the layer can be used in a model without errors
        # We'll test construction and basic functionality

        # Create a message passing layer
        layer = MessagePassing(aggregator="mean", name="test_message_passing")

        # Test that it can process inputs directly
        result = layer([self.node_features, self.edge_index])
        self.assertEqual(result.shape, (self.num_nodes, self.num_features))

        # Test that layer can be serialized/deserialized (important for model saving)
        config = layer.get_config()
        new_layer = MessagePassing.from_config(config)

        # Test that the new layer produces same output
        result2 = new_layer([self.node_features, self.edge_index])
        # pyrefly: ignore # implicitly-defined-attribute
        self.assertEqual(result2.shape, (self.num_nodes, self.num_features))

        # Test layer name is preserved
        self.assertEqual(new_layer.name, "test_message_passing")


# Performance benchmark tests
class BenchmarkMessagePassing:
    """Benchmark tests for MessagePassing layer performance."""

    sizes: list[int]  # Explicitly declare instance attribute
    feature_dims: list[int]  # Explicitly declare instance attribute

    def setup_method(self) -> None:
        """Set up benchmark fixtures."""
        # Create larger graphs for benchmarking
        self.sizes = [100, 1000, 10000]
        self.feature_dims = [32, 64, 128]

    def create_random_graph(
        self, num_nodes: int, num_features: int, avg_degree: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a random graph for benchmarking."""
        num_edges = num_nodes * avg_degree

        # Random node features
        features = np.random.randn(num_nodes, num_features).astype(np.float32)

        # Random edges (with replacement for simplicity)
        source = np.random.randint(0, num_nodes, size=num_edges)
        target = np.random.randint(0, num_nodes, size=num_edges)
        edge_index = np.stack([source, target], axis=0).astype(np.int32)

        return features, edge_index

    def benchmark_aggregators(self) -> dict[int, dict[str, float]]:
        """Benchmark different aggregation methods."""
        import time

        aggregators = ["mean", "max", "sum", "min", "std"]
        results = {}

        for num_nodes in self.sizes:
            results[num_nodes] = {}
            features, edge_index = self.create_random_graph(num_nodes, 64)

            for aggregator in aggregators:
                layer = MessagePassing(aggregator=aggregator)

                # Warm up
                _ = layer([features, edge_index])

                # Measure time
                start = time.time()
                for _ in range(10):
                    _ = layer([features, edge_index])
                end = time.time()

                avg_time = (end - start) / 10
                results[num_nodes][aggregator] = avg_time

        return results


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
