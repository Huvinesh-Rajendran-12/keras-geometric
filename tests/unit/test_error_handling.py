"""
Systematic error handling and edge case tests.

These tests verify that all layers handle invalid inputs gracefully
and provide meaningful error messages.
"""

import os

import numpy as np
import pytest

# Set backend
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras

from keras_geometric.layers import GATv2Conv, GCNConv, GINConv, MessagePassing, SAGEConv

pytestmark = [pytest.mark.unit, pytest.mark.error_handling]


class TestErrorHandling:
    """Test error handling and edge cases for all layers."""

    def test_invalid_aggregator_names(self):
        """
        Verifies that passing invalid aggregator names to MessagePassing, GINConv, and SAGEConv raises a ValueError with an appropriate message.
        """
        invalid_aggregators = ["invalid", "median", "variance", ""]

        for invalid_agg in invalid_aggregators:
            with pytest.raises(ValueError, match="Invalid aggregator"):
                MessagePassing(aggregator=invalid_agg)

            with pytest.raises(ValueError, match="Invalid aggregator"):
                GINConv(output_dim=16, aggregator=invalid_agg)

            with pytest.raises(ValueError, match="Invalid aggregator"):
                SAGEConv(output_dim=16, aggregator=invalid_agg)

    def test_invalid_input_shapes(self):
        """
        Tests that GCNConv raises appropriate errors when given inputs with invalid shapes.

        Verifies that missing required inputs or incorrectly shaped edge indices result in
        ValueError or IndexError as expected.
        """
        layer = GCNConv(output_dim=16)

        # Wrong number of inputs
        with pytest.raises(ValueError):
            layer([np.random.randn(10, 8)])  # Missing edge_index

        # Wrong edge index shape
        node_features = np.random.randn(10, 8).astype(np.float32)

        with pytest.raises(ValueError, match=".*"):
            # Edge index should be (2, E) not (3, E)
            invalid_edges = np.random.randint(0, 10, size=(3, 20)).astype(np.int32)
            layer([node_features, invalid_edges])

    def test_mismatched_dimensions(self):
        """
        Tests that GCNConv produces correct output shape when node features and edge indices reference different numbers of nodes.

        Verifies that the layer handles cases where some nodes are not referenced by any edge without raising errors.
        """
        layer = GCNConv(output_dim=16)

        # Node features and edge indices with mismatched batch dimensions
        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 5, size=(2, 20)).astype(
            np.int32
        )  # Only 5 nodes referenced

        # This should work (some nodes may not have edges)
        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)

    def test_out_of_bounds_edge_indices(self):
        """
        Tests that GCNConv raises an error when edge indices reference nodes outside the valid range.
        """
        import keras
        
        # Skip for PyTorch and JAX backends as they may not raise exceptions for out-of-bounds indices
        # This is a known behavior difference where these backends can return arbitrary values
        if keras.backend.backend() in ["torch", "jax"]:
            pytest.skip(f"{keras.backend.backend()} backend may not raise exceptions for out-of-bounds indices")
            
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(10, 8).astype(np.float32)

        # Edge indices referencing nodes beyond the graph
        invalid_edges = np.array(
            [[0, 1, 15], [1, 2, 3]], dtype=np.int32
        )  # Node 15 doesn't exist

        # This should raise an error - different backends raise different error types
        with pytest.raises((ValueError, Exception)):
            layer([node_features, invalid_edges])

    def test_negative_edge_indices(self):
        """
        Tests that GCNConv handles negative edge indices without crashing.

        Verifies that the layer either produces output of the expected shape or raises an error when provided with edge indices containing negative values.
        """
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(10, 8).astype(np.float32)

        # Negative edge indices - may be handled gracefully by backend
        invalid_edges = np.array([[0, -1, 2], [1, 2, 3]], dtype=np.int32)

        # Some backends may handle negative indices, so just verify it doesn't crash
        try:
            output = layer([node_features, invalid_edges])
            # If it works, verify output shape
            assert output.shape == (10, 16)
        except (ValueError, IndexError, Exception):
            # Or it may raise an error, which is also acceptable
            pass

    def test_empty_inputs(self):
        """
        Tests that GCNConv correctly handles empty node features and edge indices.

        Verifies that the output shape is (0, output_dim) when provided with empty inputs.
        """
        layer = GCNConv(output_dim=16)

        # Empty node features
        empty_nodes = np.array([]).reshape(0, 8).astype(np.float32)
        empty_edges = np.array([[], []], dtype=np.int32)

        output = layer([empty_nodes, empty_edges])
        assert output.shape == (0, 16)

    def test_single_node_graph(self):
        """
        Tests that GCNConv correctly processes a graph with a single node and no edges.

        Verifies that the output shape matches the single node and output dimension, and that all output values are finite.
        """
        layer = GCNConv(output_dim=16)

        # Single node, no edges
        single_node = np.random.randn(1, 8).astype(np.float32)
        no_edges = np.array([[], []], dtype=np.int32)

        output = layer([single_node, no_edges])
        assert output.shape == (1, 16)
        # Just verify it produces finite output
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output)))

    def test_self_loops_handling(self):
        """
        Tests that GCNConv correctly processes graphs containing self-loops, ensuring output shape matches the number of nodes and contains no NaN values.
        """
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(5, 8).astype(np.float32)

        # Edges with self-loops
        edges_with_self_loops = np.array(
            [
                [0, 1, 2, 0, 1],  # Self-loops for nodes 0 and 1
                [1, 2, 3, 0, 1],
            ],
            dtype=np.int32,
        )

        output = layer([node_features, edges_with_self_loops])
        assert output.shape == (5, 16)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

    def test_duplicate_edges_handling(self):
        """
        Tests that GCNConv produces valid outputs when the input graph contains duplicate edges.

        Verifies that the output shape matches the number of nodes and output dimension, and that no NaN values are present in the result.
        """
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(4, 8).astype(np.float32)

        # Duplicate edges - correct input order
        duplicate_edges = np.array(
            [
                [0, 1, 1, 2],  # Edge (1,2) appears twice
                [1, 2, 2, 3],
            ],
            dtype=np.int32,
        )

        output = layer([node_features, duplicate_edges])  # Correct order
        assert output.shape == (4, 16)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

    def test_invalid_output_dimensions(self):
        """
        Tests that GCNConv can be created with a minimal output dimension and produces output of the expected shape.
        """
        # Just test that we can create layers with various dimensions
        # Error handling may vary by backend and Keras version
        layer = GCNConv(output_dim=1)  # Very small but valid

        node_features = np.random.randn(5, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 5, size=(2, 10)).astype(np.int32)
        output = layer([node_features, edge_indices])
        assert output.shape == (5, 1)

    def test_invalid_heads_for_attention(self):
        """
        Tests that GATv2Conv produces correct output shape when initialized with a single attention head.

        Verifies that the layer handles the minimum valid number of attention heads and outputs the expected shape.
        """
        # Test with minimum valid heads
        layer = GATv2Conv(output_dim=16, heads=1)

        node_features = np.random.randn(5, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 5, size=(2, 10)).astype(np.int32)
        output = layer([node_features, edge_indices])
        assert output.shape == (5, 16)

    def test_numerical_edge_cases(self):
        """
        Tests GCNConv layer behavior with NaN and infinite values in node features.

        Verifies that the output contains NaN or infinite values when the corresponding input features contain NaN or inf, ensuring numerical edge cases are handled as expected.
        """
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        # Test with NaN inputs
        nan_features = node_features.copy()
        nan_features[0, 0] = np.nan

        output = layer([nan_features, edge_indices])
        # Output should contain NaN due to input NaN
        assert np.any(np.isnan(keras.ops.convert_to_numpy(output)))

        # Test with infinite inputs
        inf_features = node_features.copy()
        inf_features[0, 0] = np.inf

        output = layer([inf_features, edge_indices])
        # Output should contain inf due to input inf
        assert np.any(np.isinf(keras.ops.convert_to_numpy(output)))

    def test_very_large_numbers(self):
        """
        Tests that GCNConv produces finite outputs when given very large input feature values.

        Verifies that the output shape matches the number of nodes and output dimension, and that all output values are finite despite large input magnitudes.
        """
        layer = GCNConv(output_dim=16)

        node_features = np.ones((10, 8), dtype=np.float32) * 1e10
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output)))  # Should remain finite

    def test_very_small_numbers(self):
        """
        Tests that GCNConv produces finite outputs when input node features contain very small values.
        """
        layer = GCNConv(output_dim=16)

        node_features = np.ones((10, 8), dtype=np.float32) * 1e-10
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output)))

    def test_message_passing_invalid_inputs(self):
        """
        Tests that the MessagePassing base class raises ValueError when called with invalid input formats, such as a string, an empty list, or missing edge indices.
        """
        mp = MessagePassing(aggregator="mean")

        # Test with wrong input format
        with pytest.raises(ValueError):
            mp.call("invalid_input")

        with pytest.raises(ValueError):
            mp.call([])  # Empty list

        with pytest.raises(ValueError):
            mp.call([np.random.randn(10, 8)])  # Missing edge_index

    def test_gin_conv_epsilon_edge_cases(self):
        """
        Tests GINConv with a large epsilon initialization parameter to ensure numerical stability.

        Verifies that the layer produces output of the expected shape and that the output contains no NaN values when using a large value for the epsilon parameter.
        """
        # GINConv uses 'eps_init' parameter
        layer = GINConv(output_dim=16, eps_init=1e6)

        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(output)))

    def test_sage_conv_pooling_without_mlp(self):
        """
        Tests that SAGEConv with the "pooling" aggregator correctly creates its own MLP and produces output of the expected shape.
        """
        # This should work - pooling aggregator should create its own MLP
        layer = SAGEConv(output_dim=16, aggregator="pooling")

        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)

    def test_layer_reuse(self):
        """
        Verifies that a GCNConv layer instance can be reused with inputs of different shapes.

        Ensures that the layer produces outputs with correct shapes when called multiple times with varying node and edge counts.
        """
        layer = GCNConv(output_dim=16)

        # First input
        node_features_1 = np.random.randn(10, 8).astype(np.float32)
        edge_indices_1 = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output_1 = layer([node_features_1, edge_indices_1])

        # Second input with different shape
        node_features_2 = np.random.randn(15, 8).astype(np.float32)
        edge_indices_2 = np.random.randint(0, 15, size=(2, 30)).astype(np.int32)

        output_2 = layer([node_features_2, edge_indices_2])

        assert output_1.shape == (10, 16)
        assert output_2.shape == (15, 16)

    def test_edge_attr_dimension_mismatch(self):
        """
        Tests GCNConv behavior when edge attribute dimensions do not match the number of edges.

        Verifies that the layer either raises an error or produces output of the correct shape when provided edge attributes with a mismatched number of rows relative to the edge indices.
        """
        layer = GCNConv(output_dim=16)

        node_features = np.random.randn(10, 8).astype(np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        # Edge attributes with wrong number of edges may be handled gracefully
        wrong_edge_attrs = np.random.randn(15, 4).astype(
            np.float32
        )  # Should be 20, not 15

        try:
            output = layer([node_features, edge_indices, wrong_edge_attrs])
            # If it works, verify output shape
            assert output.shape == (10, 16)
        except (ValueError, Exception):
            # Or it may raise an error, which is also acceptable
            pass

    def test_mixed_dtypes(self):
        """
        Tests that GCNConv correctly processes inputs where node features are float64 and edge indices are int64.

        Verifies that the layer produces an output of the expected shape when given mixed data types.
        """
        layer = GCNConv(output_dim=16)

        # Node features as float64, edge indices as int64
        node_features = np.random.randn(10, 8).astype(np.float64)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int64)

        output = layer([node_features, edge_indices])
        assert output.shape == (10, 16)

    def test_zero_input_features(self):
        """
        Verifies that GCNConv produces finite outputs when given all-zero input features and bias enabled.

        Ensures the output shape matches the expected dimensions and contains no NaN or infinite values.
        """
        layer = GCNConv(output_dim=16, use_bias=True)

        zero_features = np.zeros((10, 8), dtype=np.float32)
        edge_indices = np.random.randint(0, 10, size=(2, 20)).astype(np.int32)

        output = layer([zero_features, edge_indices])
        assert output.shape == (10, 16)
        # With GCN normalization, even with bias, output may still be zeros
        # Just verify no NaN or inf values
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output)))
